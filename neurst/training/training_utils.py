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
import signal
import sys
import time
from collections import namedtuple
from typing import Tuple

import tensorflow as tf
from absl import logging

from neurst.criterions import Criterion
from neurst.data.datasets import Dataset
from neurst.data.datasets.multiple_dataset import MultipleDataset
from neurst.training import distribution_utils
from neurst.training.hvd_utils import HorovodDistributedLossScaleOptimizer
from neurst.training.revised_dynamic_loss_scale import RevisedDynamicLossScale
from neurst.utils import compat
from neurst.utils.checkpoints import AverageCheckpointSaver, KeepBestCheckpointSaver
from neurst.utils.configurable import ModelConfigs
from neurst.utils.misc import DummyContextManager, to_numpy_or_python_type

GPU_EFFICIENT_LEVEL = namedtuple("gpu_efficient_level",
                                 "LEVEL0 LEVEL1 LEVEL2 LEVEL3 LEVEL4 LEVEL5")(0, 1, 2, 3, 4, 5)

EFFICIENT_MULTIPLIER = {
    GPU_EFFICIENT_LEVEL.LEVEL0: 8,
    GPU_EFFICIENT_LEVEL.LEVEL1: 8,
    GPU_EFFICIENT_LEVEL.LEVEL2: 16,
    GPU_EFFICIENT_LEVEL.LEVEL3: 32,
    GPU_EFFICIENT_LEVEL.LEVEL4: 64,
    GPU_EFFICIENT_LEVEL.LEVEL5: 8
}


def minimal_multiple(val, factor):
    if val % factor == 0:
        return val
    return int((val // factor + 1) * factor)


def maximum_lower_multiple(val, factor):
    multiple = minimal_multiple(val, factor)
    if (multiple - val) * 1. / factor > 0.25:
        cand = multiple - factor
        if cand <= 0:
            return val
        elif (val - cand) * 1. / factor >= 0.25:
            return val
        else:
            return cand
    return multiple


def startup_env(dtype="float16",
                enable_check_numerics=False,
                enable_xla=False):
    """ Startup program environments. """
    if dtype not in ["float32", "float16"]:
        raise ValueError(f"Not supported dtype={dtype} (now only accept float32 and float16)")
    if dtype == "float16":
        logging.info("Using float16 as computation dtype.")
        if compat.IS_PREV_TF_2_4_0:
            from tensorflow.keras.mixed_precision import experimental as mixed_precision
            policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_policy(policy)
        else:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        compat.register_computation_dtype("float16", -6.e4)
    if enable_check_numerics:
        logging.info("Enable checking numerics.")
        tf.debugging.enable_check_numerics()
    if enable_xla:
        tf.config.optimizer.set_jit(True)
        # it causes OOM and performance reression
        tf.config.optimizer.set_experimental_options({"pin_to_host_optimization": False})


def is_third_party_allreduce(strategy):
    return strategy in ["byteps", "horovod"]


def handle_distribution_strategy(distribution_strategy):
    """ Create distribution strategy. """
    strategy = None
    if distribution_strategy:
        strategy = distribution_strategy
        if isinstance(distribution_strategy, dict):
            strategy = distribution_strategy.get("distribution_strategy", None)
        if isinstance(distribution_strategy, str):
            strategy = distribution_strategy.lower()
        if is_third_party_allreduce(strategy):
            if strategy == "horovod":
                import horovod.tensorflow.keras as hvd
            else:
                import byteps.tensorflow.keras as hvd
            logging.info("import {} as hvd backend.".format(strategy))
            hvd.init()
            # Horovod: pin GPU to be used to process local rank (one GPU per process)
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if gpus:
                tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
            compat.register_distributed_worker_setting(hvd.rank(), hvd.size(), strategy)
            if hvd.rank() != 0:
                logging.set_verbosity(logging.ERROR)
        else:
            if isinstance(distribution_strategy, str):
                strategy = distribution_utils.get_distribution_strategy(
                    distribution_strategy=distribution_strategy)
            elif isinstance(distribution_strategy, dict):
                strategy = distribution_utils.get_distribution_strategy(**distribution_strategy)

    if strategy is None:
        logging.info("No distribution strategy was used.")
    else:
        try:
            logging.info("Using distribution strategy: {} with num_replicas_in_sync={}".format(
                strategy, strategy.num_replicas_in_sync))
        except Exception:
            pass
    return strategy


def get_strategy_scope(strategy):
    """ Returns the distributed strategy context. """
    try:
        return strategy.scope()
    except AttributeError:
        return DummyContextManager()


def get_num_replicas_in_sync(strategy):
    """ Returns the number of replicas in sync. """
    try:
        return strategy.num_replicas_in_sync
    except AttributeError:
        return 1


_SINGLE_DS_NAME = ""


def build_datasets(mode,
                   strategy,
                   custom_dataset: Dataset,
                   task,
                   cache=False,
                   args=None):
    """ Builds datasets and returns datasets. """
    if isinstance(custom_dataset, MultipleDataset):
        ret_ds = {
            name: task.create_and_batch_tfds(
                d, mode, num_replicas_in_sync=get_num_replicas_in_sync(strategy), args=args)
            for name, d in custom_dataset.datasets.items()
        }
        if cache:
            ret_ds = {k: v.cache() for k, v in ret_ds.items()}
    else:
        ret_ds = task.create_and_batch_tfds(
            custom_dataset, mode, num_replicas_in_sync=get_num_replicas_in_sync(strategy), args=args)
        if cache:
            ret_ds = ret_ds.cache()
    return ret_ds


def maybe_distribution_dataset(strategy, dataset):
    try:
        return strategy.experimental_distribute_dataset(dataset)
    except AttributeError:
        return dataset


def make_predictions(strategy,
                     model: tf.keras.models.Model,
                     tfds,
                     custom_dataset,
                     map_func=None,
                     eagerly=False):
    """ Makes predictions.
        The `tf_ds` are created by `build_datasets` fn.

    Args:
        strategy: The distritbution strategy.
        model: The keras model.
        tfds: A dataset or a list of datasets returned by `build_datasets`.
        custom_dataset: A Dataset object.
        map_func: A callable function that tasks custom dataset and eval result as inputs
            and converters each eval result.
        eagerly: Whether to run eagerly.

    Returns: A dict of evaluation results for each dataset
        or the evaluation result for single dataset.
    """
    if not isinstance(custom_dataset, MultipleDataset):
        tfds = {_SINGLE_DS_NAME: tfds}
    results = {}
    with get_strategy_scope(strategy):
        if eagerly:
            def predict_fn(iterator):
                data = next(iterator)
                return model(data)
        else:
            predict_fn = model.make_predict_function()
        for name, ds in tfds.items():
            assert isinstance(ds, tf.data.Dataset), (
                "Unsupported type of dataset({}): {}".format(ds, type(ds)))
            if eagerly:
                iterator = iter(ds.prefetch(tf.data.experimental.AUTOTUNE))
            else:
                iterator = iter(maybe_distribution_dataset(strategy, ds.prefetch(
                    tf.data.experimental.AUTOTUNE)))
            this_results = []
            while True:
                try:
                    this_results.append(predict_fn(iterator))
                except (StopIteration, tf.errors.OutOfRangeError):
                    break
            if map_func is None:
                results[name] = this_results
            else:
                results[name] = map_func(this_results)
        if not isinstance(custom_dataset, MultipleDataset):
            results = results[_SINGLE_DS_NAME]
    return results


def reduce_eval_results(criterion: Criterion,
                        custom_dataset: Dataset,
                        eval_results) -> Tuple[dict, dict, dict]:
    """

    Args:
        criterion: A criterion instance.
        custom_dataset: The custom dataset object.
        eval_results: The prediction results from `make_predictions`.

    Returns: A tuple of dicts of evaluation results:
        - result dict of each dataset
        - averaged result
        - mixed result
    """
    if isinstance(custom_dataset, MultipleDataset):
        res = {}
        avg_res = {}
        mixed_data = []
        for k, v in eval_results.items():
            res[k] = criterion.reduce_metrics(v)
            mixed_data.extend(v)
            for kk, vv in res[k].items():
                if kk not in avg_res:
                    avg_res[kk] = 0.
                avg_res[kk] += vv * custom_dataset.sample_weights[k]
        mixed_res = criterion.reduce_metrics(mixed_data)
    else:
        res = avg_res = mixed_res = criterion.reduce_metrics(eval_results)
    return (to_numpy_or_python_type(res), to_numpy_or_python_type(avg_res),
            to_numpy_or_python_type(mixed_res))


class TrainingStatusRecorder(object):
    """ Manage the training status with the best metrics. """

    def __init__(self,
                 model,
                 task,
                 metric,
                 estop_patience=None,
                 best_checkpoint_path=None,
                 auto_average_checkpoints=True,
                 best_avg_checkpoint_path=None,
                 top_checkpoints_to_keep=0):
        """ Initializes manager for arbitrary evaluation strategies.

        Args:
            model: The custom keras model (inherent BaseModel).
            task: The custom task.
            metric: The evaluation metric object.
            estop_patience: An integer, the training process will automatically shut down until the program
                fail to acquire a better metric score anymore if `early_stop_patience` greater than 0.
            best_checkpoint_path: The path for checkpoints with best metric scores if provided,
                otherwise, default \"`model_dir`_best\" will be used.
            best_avg_checkpoint_path: The path to saving the averaged checkpoints.
            auto_average_checkpoints: A boolean, whether to do checkpoint average on all model weights.
                An extra directory for averaged weights will be created. It is only available when
                `eval_best_checkpoint_path` is provided.
            top_checkpoints_to_keep: An integer, the maximum number of checkpoints to be saved
                (`max_to_keep` for checkpoint manager), and the number of latest checkpoints to be averaged
                if `eval_auto_average_checkpoints` is True. If <= 0, no more checkpoints will be saved.
        """
        self._model = model
        self._task = task
        self._metric = metric
        self._estop_patience = estop_patience
        self._best_checkpoint_path = best_checkpoint_path
        self._auto_average_checkpoints = auto_average_checkpoints
        self._best_avg_checkpoint_path = best_avg_checkpoint_path
        self._top_checkpoints_to_keep = top_checkpoints_to_keep
        self._keep_best_ckpt_saver = None
        self._average_ckpt_saver = None
        if self._top_checkpoints_to_keep and self._top_checkpoints_to_keep > 0:
            self._keep_best_ckpt_saver = KeepBestCheckpointSaver(
                model=self._model,
                directory=self._best_checkpoint_path,
                metric=self._metric,
                max_to_keep=self._top_checkpoints_to_keep)
            ModelConfigs.dump(self._task.model_configs(self._model), self._keep_best_ckpt_saver.directory)
            if self._auto_average_checkpoints:
                self._average_ckpt_saver = AverageCheckpointSaver(
                    model=self._model,
                    directory=self._best_avg_checkpoint_path,
                    metric=self._metric,
                    max_to_keep=self._top_checkpoints_to_keep)
                ModelConfigs.dump(self._task.model_configs(self._model), self._average_ckpt_saver.directory)
        self._best_metric_result = None
        self._bad_count = 0

    @property
    def best(self):
        return self._best_metric_result

    def record(self, step, metric_result):
        """ Records the metrics and keep the best. """
        metric_result = to_numpy_or_python_type(metric_result)
        if (self._best_metric_result is None
            or self._metric.greater_or_eq(metric_result, self._best_metric_result)):
            self._bad_count = 0
            self._best_metric_result = metric_result
        else:
            self._bad_count += 1

        # re-save the best checkpoint
        if self._keep_best_ckpt_saver is not None:
            start_time = time.time()
            stat = self._keep_best_ckpt_saver.save(step, metric_result)
            logging.info("Checking the best checkpoints kept and %s. Elapsed %.2fs",
                         "a new checkpoint was saved" if stat else "no checkpoint was saved.",
                         time.time() - start_time)
        if self._average_ckpt_saver is not None:
            start_time = time.time()
            stat = self._average_ckpt_saver.save(step, metric_result)
            if stat:
                logging.info("An averaged checkpoint was saved. Elapsed %.2fs", time.time() - start_time)

        if self._estop_patience is not None:
            logging.info(f"Evaluating {self._metric.flag} at step={step} with bad count={self._bad_count} "
                         f"(early_stop_patience={self._estop_patience}).")
        if self._estop_patience and self._bad_count >= self._estop_patience > 0:
            logging.info("Hit maximum patience! Early Stop!!!")

            # kill self and exit with code=0
            def handler(*args):
                sys.exit(0)

            # register for signal
            signal.signal(signal.SIGUSR1, handler)
            os.kill(os.getpid(), signal.SIGUSR1)


def handle_fp16_and_distributed_optimizer(optimizer, lr_schedule, hvd_backend=None):
    if hvd_backend == "horovod":
        import horovod.tensorflow.keras as hvd
        from horovod.tensorflow import Compression
    elif hvd_backend == "byteps":
        import byteps.tensorflow.keras as hvd
        from byteps.tensorflow import Compression

    if hvd_backend:
        compression = Compression.none
        if compat.CUSTOM_GLOBAL_FLOATX == "float16":
            compression = Compression.fp16

    if lr_schedule is not None and hvd_backend is None:
        # TODO(ZhaoChengqi): pay attention to API changes
        optimizer._set_hyper("learning_rate", lr_schedule)
    # specify the following scenario
    if compat.CUSTOM_GLOBAL_FLOATX == "float16":
        if compat.IS_PREV_TF_2_4_0:
            from tensorflow.keras.mixed_precision.experimental import LossScaleOptimizer
            from tensorflow.python.keras import backend
            from tensorflow.python.training.experimental.loss_scale import get_loss_scale_weights

            revised_loss_scale = RevisedDynamicLossScale()
            if hvd_backend:
                opt = LossScaleOptimizer(optimizer, loss_scale=1)
                opt = hvd.DistributedOptimizer(opt, compression=compression, sparse_as_dense=True)
                opt._loss_scale = revised_loss_scale
                for weight in get_loss_scale_weights(opt._loss_scale):
                    backend.track_variable(weight)
                opt._track_trackable(opt._loss_scale, 'loss_scale', overwrite=True)
            else:
                opt = LossScaleOptimizer(optimizer, loss_scale=revised_loss_scale)
        else:
            if hvd_backend:
                opt = HorovodDistributedLossScaleOptimizer(inner_optimizer=optimizer,
                                                           compression=compression,
                                                           sparse_as_dense=True,
                                                           hvd_backend=hvd_backend)
            else:
                opt = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
                opt._loss_scale = RevisedDynamicLossScale(
                    initial_loss_scale=2 ** 15, growth_steps=2000, multiplier=2)
                opt._track_trackable(opt._loss_scale, "loss_scale", overwrite=True)
        return opt

    return optimizer


def validate_unique_varname(variables):
    duplicated = set()
    varname_set = set()
    for var in variables:
        varname = var.name
        if varname in varname_set:
            duplicated.add(varname)
        else:
            varname_set.add(varname)
    if len(duplicated) > 0:
        raise ValueError("Found duplicated variable names: \n" + str(list(duplicated)))
