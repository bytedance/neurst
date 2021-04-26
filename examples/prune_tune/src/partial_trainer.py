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
import pickle
from distutils.version import LooseVersion

import tensorflow as tf
from absl import logging

from examples.prune_tune.src.partial_tuning_optimizer import create_partial_tuning_optimizer
from neurst.data.dataset_utils import map_data_for_keras
from neurst.data.datasets.multiple_dataset import MultipleDataset
from neurst.exps import register_exp
from neurst.exps.trainer import Trainer
from neurst.models.model_utils import summary_model_variables
from neurst.optimizers.schedules import build_lr_schedule
from neurst.sparsity.pruning_optimizer import create_pruning_optimizer
from neurst.training import CustomCheckpointCallback, LearningRateScheduler, MetricReductionCallback, training_utils
from neurst.training.gradaccum_keras_model import GradAccumKerasModel
from neurst.utils import compat
from neurst.utils.flags_core import Flag


@register_exp("prune_tune_train")
class PruneTuneTrainer(Trainer):
    """ Trainer for all tasks. """

    def __init__(self, args, **kwargs):
        """ Initializes a util class for training neural models. """
        super(PruneTuneTrainer, self).__init__(args, **kwargs)
        if args["mask_pkl"]:
            with tf.io.gfile.GFile(args["mask_pkl"], 'rb') as f:
                self.load_mask = pickle.load(f)
        else:
            self.mask_dir = os.path.join(self.model_dir, "mask.pkl")
            self.load_mask = None
        self._partial_tuning = args["partial_tuning"]

    @staticmethod
    def class_or_method_args():
        this_args = super(PruneTuneTrainer, PruneTuneTrainer).class_or_method_args()
        this_args.extend(
            [Flag("partial_tuning", dtype=Flag.TYPE.BOOLEAN, default=False,
                  help="Train partial weights according to mask"),
             Flag("mask_pkl", dtype=Flag.TYPE.STRING, default=None,
                  help="The file to the masks")])
        return this_args

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
            if self._partial_tuning is True:
                self._optimizer = create_partial_tuning_optimizer(self._optimizer, self.model, self.load_mask)
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

        history = keras_model.fit(
            map_data_for_keras(tfds.repeat()),
            initial_epoch=0,
            epochs=1,
            steps_per_epoch=self._train_steps,  # * args["update_cycle"],
            verbose=2,
            callbacks=training_callbacks)
        logging.info(history.history)

        if self._pruning_schedule is not None:
            mask = [
                tf.Variable(
                    (tf.cast(tf.math.not_equal(weight, 0.), weight.dtype.base_dtype)),
                    dtype=weight.dtype.base_dtype,
                    trainable=False,
                    synchronization=tf.VariableSynchronization.ON_READ,
                    aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA) for weight in keras_model.trainable_weights
            ]
            # np.save(self.mask_dir, mask)
            with open(self.mask_dir, 'wb') as f:
                pickle.dump(mask, f)

        if self._partial_tuning is True:
            mask = self.load_mask
            # np.save(self.mask_dir, mask)
            saved_mask_dir = os.path.join(self.model_dir, "mask.pkl")
            with open(saved_mask_dir, 'wb') as f:
                pickle.dump(mask, f)
