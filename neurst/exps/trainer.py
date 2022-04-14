# Copyright 2020 ByteDance Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from distutils.version import LooseVersion

import tensorflow as tf
from absl import logging

from neurst.criterions import Criterion, build_criterion
from neurst.data.dataset_utils import map_data_for_keras
from neurst.data.datasets.multiple_dataset import MultipleDataset
from neurst.exps import BaseExperiment, register_exp
from neurst.models.model_utils import summary_model_variables
from neurst.optimizers import OPTIMIZER_REGISTRY_NAME, build_optimizer, controlling_optimizer
from neurst.optimizers.schedules import LR_SCHEDULE_REGISTRY_NAME, build_lr_schedule
from neurst.sparsity.pruning_optimizer import create_pruning_optimizer
from neurst.sparsity.pruning_schedule import PolynomialDecay, PruningSchedule, build_pruning_schedule
from neurst.training import (CustomCheckpointCallback, LearningRateScheduler, MetricReductionCallback, Validator,
                             build_validator, training_utils)
from neurst.training.gradaccum_keras_model import GradAccumKerasModel
from neurst.utils import compat
from neurst.utils.checkpoints import restore_checkpoint_if_possible, restore_checkpoint_if_possible_v2
from neurst.utils.flags_core import Flag, ModuleFlag
from neurst.utils.misc import flatten_string_list


@register_exp("train")
class Trainer(BaseExperiment):
    """ Trainer for all tasks. """

    def __init__(self, args, **kwargs):
        """ Initializes a util class for training neural models. """
        super(Trainer, self).__init__(**kwargs)
        self._tb_log_dir = args["tb_log_dir"]
        self._train_steps = args["train_steps"]
        self._train_epochs = args["train_epochs"]
        self._summary_steps = args["summary_steps"]
        self._save_checkpoint_steps = args["save_checkpoint_steps"]
        self._checkpoints_max_to_keep = args["checkpoints_max_to_keep"]
        self._initial_global_step = args["initial_global_step"]
        self._pretrain_variable_pattern = args["pretrain_variable_pattern"]
        if args["pretrain_model"] and isinstance(args["pretrain_model"][0], dict):
            self._pretrain_v2 = True
            self._pretrain_model = args["pretrain_model"]
            if self._pretrain_variable_pattern:
                logging.info("Using pretrain model v2 and ignoring pretrain_variable_pattern: "
                             f"{self._pretrain_variable_pattern}")
        else:
            self._pretrain_v2 = False
            self._pretrain_model = flatten_string_list(args["pretrain_model"])
            if self._pretrain_model:
                if self._pretrain_variable_pattern is None:
                    self._pretrain_variable_pattern = [None] * len(self._pretrain_model)
                elif isinstance(self._pretrain_variable_pattern, str):
                    self._pretrain_variable_pattern = [self._pretrain_variable_pattern]
            assert ((self._pretrain_model is None and self._pretrain_variable_pattern is None)
                    or len(self._pretrain_model) == len(self._pretrain_variable_pattern)
                    or len(self._pretrain_model) == 1), (
                "`pretrain_variable_pattern` must match with `pretrain_model`.")
            if self._pretrain_model is not None and self._pretrain_variable_pattern is None:
                self._pretrain_variable_pattern = [None] * len(self._pretrain_model)
        self._update_cycle = args["update_cycle"]
        self._clip_value = args["clip_value"]
        self._clip_norm = args["clip_norm"]
        self._hvd_backend = self.strategy if self.strategy in ["byteps", "horovod"] else None
        with training_utils.get_strategy_scope(self.strategy):
            self._criterion = build_criterion(args)
            self._criterion.set_model(self.model)
            self._lr_schedule_args = args
            if compat.IS_PREV_TF_2_4_0:
                self._optimizer = build_optimizer(args)
            else:
                self._optimizer = build_optimizer(args, clipnorm=self._clip_norm, clipvalue=self._clip_value)
            assert self._optimizer is not None, "optimizer parameters must be provided for training."
            self._optimizer = controlling_optimizer(self._optimizer, args["optimizer_controller"],
                                                    args["optimizer_controller_args"])
        self._validator = build_validator(args)
        self._experimental_count_batch_num = args["experimental_count_batch_num"]
        self._freeze_variables = args["freeze_variables"]
        self._pruning_schedule = build_pruning_schedule(args)
        self._pruning_variable_pattern = args["pruning_variable_pattern"]
        self._nopruning_variable_pattern = args["nopruning_variable_pattern"]

    @staticmethod
    def class_or_method_args():
        return [
            ModuleFlag(Criterion.REGISTRY_NAME, help="The criterion for training or evaluation."),
            ModuleFlag(OPTIMIZER_REGISTRY_NAME, help="The optimizer for training."),
            ModuleFlag(LR_SCHEDULE_REGISTRY_NAME, help="The learning schedule for training."),
            ModuleFlag(Validator.REGISTRY_NAME, help="The validation process while training."),
            ModuleFlag(PruningSchedule.REGISTRY_NAME, help="The schedule for weight weight_pruning.",
                       default=PolynomialDecay.__name__),
            Flag("tb_log_dir", dtype=Flag.TYPE.STRING, default=None,
                 help="The path to store tensorboard summary, or `model_dir`/train by default."),
            Flag("train_epochs", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The number of times that the training process scans the whole data."),
            Flag("train_steps", dtype=Flag.TYPE.INTEGER, default=10000000,
                 help="The maximum steps for training loop."),
            Flag("summary_steps", dtype=Flag.TYPE.INTEGER, default=200,
                 help="Doing summary(logging & tensorboard) this every steps."),
            Flag("save_checkpoint_steps", dtype=Flag.TYPE.INTEGER, default=1000,
                 help="Saving checkpoints this every steps."),
            Flag("checkpoints_max_to_keep", dtype=Flag.TYPE.INTEGER, default=8,
                 help="The maximum checkpoints to be kept."),
            Flag("initial_global_step", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The manually specified initial global step."),
            Flag("pretrain_model", dtype=Flag.TYPE.STRING, default=None, multiple=True,
                 help="The path to a pretrained model directory(a seq2seq model, bert model, etc.). "
                      "(V2) Or a json/yaml-like dict string indicating pretrained models from "
                      "either neurst checkpoints or publicly available models converted "
                      "by neurst.utils.converters. Each entry has the elements: "
                      "path, model_name, from_prefix, to_prefix, name_pattern. "
                      "Multiple pretrain models are also available."),
            Flag("pretrain_variable_pattern", dtype=Flag.TYPE.STRING, default=None, multiple=True,
                 help="One can restore specified variables in the `pretrain_model` by this regular expression."
                      "Multiple pattern are also available, but must match to `pretrain_model`."),
            Flag("update_cycle", dtype=Flag.TYPE.INTEGER, default=1,
                 help="Training step with this many batches (Gradient Accumulation)."),
            Flag("clip_value", dtype=Flag.TYPE.FLOAT, default=None, help="Gradient clipping by value."),
            Flag("clip_norm", dtype=Flag.TYPE.FLOAT, default=None, help="Gradient clipping by norm."),
            Flag("experimental_count_batch_num", dtype=Flag.TYPE.BOOLEAN, default=None,
                 help="Pre-scan the dataset for training and count the number of batches."),
            Flag("freeze_variables", dtype=Flag.TYPE.STRING, default=None,
                 help="Variables whose names are matched with this regex will be freezed."),
            Flag("pruning_variable_pattern", dtype=Flag.TYPE.STRING, default=None,
                 help="The regular expression that indicates the variables will be pruned."),
            Flag("nopruning_variable_pattern", dtype=Flag.TYPE.STRING, default=None,
                 help="The regular expression that indicates the variables will NOT be pruned "
                      "(will take effect if `pruning_variable_pattern`=None)."),
            Flag("optimizer_controller", dtype=Flag.TYPE.STRING, default=None,
                 help="An optimizer wrapper controlling the specific operations during model training."),
            Flag("optimizer_controller_args", dtype=Flag.TYPE.STRING, default=None,
                 help="A dict of parameters for optimizer controller."),
        ]

    def _restore_ckpt_or_pretrain(self):
        """ restoring checkpoint from model_dir or pretrain_model dir. """
        stat = restore_checkpoint_if_possible(self.model, self.model_dir)
        continue_training_from = None
        if stat:
            logging.info(f"Successfully restoring checkpoint from model_dir={self.model_dir}")
            continue_training_from = self.model_dir
        else:
            logging.info(f"No checkpoint restored from model_dir={self.model_dir}")
            if self._pretrain_model:
                if self._pretrain_v2:
                    for pt in self._pretrain_model:
                        logging.info(f"Trying to restore from pretrain_model={pt}")
                        logging.info("NOTE THAT, one must first check the variable names in this checkpoint, "
                                     "otherwise no variables will be restored.")
                        restore_checkpoint_if_possible_v2(self.model, **pt)
                else:
                    for pt, pt_varname in zip(self._pretrain_model, self._pretrain_variable_pattern):
                        logging.info(f"Trying to restore from pretrain_model={pt}")
                        logging.info("NOTE THAT, one must first check the variable names in this checkpoint, "
                                     "otherwise no variables will be restored.")
                        restore_checkpoint_if_possible(self.model, pt, var_name_pattern=pt_varname)
                        if len(self._pretrain_model) == 1 and pt_varname is None:
                            continue_training_from = pt

        if self._initial_global_step is None and continue_training_from:
            _step = compat.hack_global_step(continue_training_from)
            if _step:
                compat.register_initial_step(_step or 0)  # must do this before creating optimizer and training
                logging.info(f"Restored initial global step={_step}")
        else:
            compat.register_initial_step(self._initial_global_step or 0)

    def run(self):
        """ Training a neural model.

        Step 1: Create training model
        Step 2: Restore checkpoint/pretrain model/global_step if exists.
        Step 3: Fetch training data.
        Step 5: Fetch training training.
        Step 6: TRAIN!!!
        """
        if self._hvd_backend == "horovod":
            import horovod.tensorflow.keras as hvd
        elif self._hvd_backend == "byteps":
            import byteps.tensorflow.keras as hvd

        tfds = training_utils.build_datasets(compat.ModeKeys.TRAIN, self.strategy,
                                             self.custom_dataset, self.task)
        if isinstance(self.custom_dataset, MultipleDataset):
            _tfds = None
            for _, ds in tfds.items():
                if _tfds is None:
                    _tfds = ds
                else:
                    _tfds = _tfds.concatenate(ds)
            tfds = _tfds
        tfds = tfds.prefetch(tf.data.experimental.AUTOTUNE)
        # Step 1: create a model
        with training_utils.get_strategy_scope(self.strategy):
            inps = self.task.create_inputs(compat.ModeKeys.TRAIN)
            formatted_inps = self.task.example_to_input(inps, compat.ModeKeys.TRAIN)
            model_out = self.model(formatted_inps, is_training=True)
            for metric_layer in self.task.build_metric_layer():
                model_out = metric_layer([formatted_inps, model_out])
            if (LooseVersion(tf.__version__) < LooseVersion("2.3")
                or LooseVersion(tf.__version__) >= LooseVersion("2.5")):
                logging.info(f"Warning: Need further check on AccumgradKerasModel when TF version={tf.__version__}. "
                             f"Here we ignore update_cycle={self._update_cycle}, "
                             f"clip_value={self._clip_value}, clip_norm={self._clip_norm}.")
                keras_model = tf.keras.Model(inps, model_out)
            elif compat.IS_PREV_TF_2_4_0:
                from neurst.training.gradaccum_keras_model import TF23GradAccumKerasModel
                keras_model = TF23GradAccumKerasModel(inps, model_out,
                                                      update_cycle=self._update_cycle,
                                                      clip_value=self._clip_value,
                                                      clip_norm=self._clip_norm,
                                                      freeze_variables=self._freeze_variables)
            else:
                keras_model = GradAccumKerasModel(inps, model_out,
                                                  update_cycle=self._update_cycle,
                                                  clip_value=self._clip_value,
                                                  clip_norm=self._clip_norm,
                                                  freeze_variables=self._freeze_variables)

            loss = self._criterion.reduce_loss(formatted_inps, model_out)
            if compat.is_tf_tensor(loss) or isinstance(loss, (list, tuple)):
                keras_model.add_loss(loss)
            elif isinstance(loss, dict):
                for _name, _loss in loss.items():
                    keras_model.add_loss(_loss)
                    keras_model.add_metric(_loss, name=_name + "_mean", aggregation="mean")
            else:
                raise ValueError("criterion.reduce_loss returns "
                                 "unsupported value of type: {}".format(type(loss)))
            self._restore_ckpt_or_pretrain()
            self._lr_schedule = build_lr_schedule(self._lr_schedule_args)
            if self._pruning_schedule is not None:
                self._optimizer = create_pruning_optimizer(self._optimizer, self.model, self._pruning_schedule,
                                                           pruning_variable_pattern=self._pruning_variable_pattern,
                                                           nopruning_variable_pattern=self._nopruning_variable_pattern,
                                                           keep_prune_property=True)
            self._optimizer = training_utils.handle_fp16_and_distributed_optimizer(
                self._optimizer, self._lr_schedule, self._hvd_backend)
            if self._hvd_backend is None:
                keras_model.compile(self._optimizer)
            else:
                # NOTE: we already add Horovod DistributedOptimizer in `_handle_fp16_and_distributed_optimizer`.
                # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
                # uses hvd.DistributedOptimizer() to compute gradients.
                keras_model.compile(self._optimizer, experimental_run_tf_function=False)
            keras_model.summary()
            summary_model_variables(self.model, self._freeze_variables)
        # initialize the checkpoint manager
        _ = compat.get_saver_or_default(self.model, self.model_dir, max_to_keep=self._checkpoints_max_to_keep)
        # build training training
        if not self._tb_log_dir:
            self._tb_log_dir = os.path.join(self.model_dir, "train")

        training_callbacks = [MetricReductionCallback(self.strategy, self._summary_steps, self._tb_log_dir,
                                                      device="GPU:0", lr_schedule=self._lr_schedule)]
        if self._hvd_backend is None or hvd.rank() == 0:
            training_callbacks.append(
                CustomCheckpointCallback(self.task.model_configs(self.model),
                                         save_checkpoint_steps=self._save_checkpoint_steps))
            if self._validator is not None:
                training_callbacks.append(self._validator.build(self.strategy, self.task, self.model))
        if self._hvd_backend is not None:
            # Horovod: average metrics among workers at the end of every epoch.
            #
            # Note: This callback must be in the list before the ReduceLROnPlateau,
            # TensorBoard or other metrics-based training.
            # NOTE!!! HERE we already integrate the metric averaging behaviour into the MetricReductionCallback.
            # training_callbacks.insert(0, hvd.callbacks.MetricAverageCallback(device="GPU:0"))

            # Horovod: broadcast initial variable states from rank 0 to all other processes.
            # This is necessary to ensure consistent initialization of all workers when
            # training is started with random weights or restored from a checkpoint.
            training_callbacks.insert(0, hvd.callbacks.BroadcastGlobalVariablesCallback(0, device="GPU:0"))
            if self._lr_schedule is not None:
                training_callbacks.append(LearningRateScheduler(self._lr_schedule))

        if self._experimental_count_batch_num:
            logging.info("Scanning the dataset......")
            iterator = iter(training_utils.maybe_distribution_dataset(self.strategy, tfds))
            cnt = 0
            for _ in iterator:
                cnt += 1
            logging.info(f"Total {cnt} batches per EPOCH.")

        if self._train_epochs:
            logging.info(f"Training for {self._train_epochs} epochs.")
            history = keras_model.fit(
                map_data_for_keras(tfds),
                initial_epoch=0,
                epochs=self._train_epochs,
                verbose=2,
                callbacks=training_callbacks)
        else:
            logging.info(f"Training for {self._train_steps} steps.")
            history = keras_model.fit(
                map_data_for_keras(tfds.repeat()),
                initial_epoch=0,
                epochs=1,
                steps_per_epoch=self._train_steps,
                verbose=2,
                callbacks=training_callbacks)

        logging.info(history.history)
