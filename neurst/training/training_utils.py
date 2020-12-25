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
from tensorflow.python.distribute import distribution_strategy_context, reduce_util
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training.experimental.loss_scale import DynamicLossScale, _assign_if_finite, _op_in_graph_mode
from tensorflow.python.util import nest

from neurst.criterions import Criterion
from neurst.data.datasets import Dataset
from neurst.data.datasets.multiple_dataset import MultipleDataset
from neurst.training import distribution_utils
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
    if (multiple - val) // factor > 0.5:
        cand = multiple - factor
        if cand <= 0:
            return multiple
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
        from tensorflow.keras.mixed_precision import experimental as mixed_precision
        policy = mixed_precision.Policy("mixed_float16", loss_scale="dynamic")
        mixed_precision.set_policy(policy)
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
                   args=None):
    """ Builds datasets and returns datasets. """
    if isinstance(custom_dataset, MultipleDataset):
        return {
            name: task.create_and_batch_tfds(
                d, mode, num_replicas_in_sync=get_num_replicas_in_sync(strategy), args=args)
            for name, d in custom_dataset.datasets.items()
        }
    return task.create_and_batch_tfds(
        custom_dataset, mode, num_replicas_in_sync=get_num_replicas_in_sync(strategy), args=args)


def maybe_distribution_dataset(strategy, dataset):
    try:
        return strategy.experimental_distribute_dataset(dataset)
    except AttributeError:
        return dataset


def make_predictions(strategy,
                     model: tf.keras.models.Model,
                     tfds,
                     custom_dataset,
                     map_func=None):
    """ Makes predictions.
        The `tf_ds` are created by `build_datasets` fn.

    Args:
        strategy: The distritbution strategy.
        model: The keras model.
        tfds: A dataset or a list of datasets returned by `build_datasets`.
        custom_dataset: A Dataset object.
        map_func: A callable function that tasks custom dataset and eval result as inputs
            and converters each eval result.

    Returns: A dict of evaluation results for each dataset
        or the evaluation result for single dataset.
    """
    if not isinstance(custom_dataset, MultipleDataset):
        tfds = {_SINGLE_DS_NAME: tfds}
    results = {}
    with get_strategy_scope(strategy):
        predict_fn = model.make_predict_function()
        for name, ds in tfds.items():
            assert isinstance(ds, tf.data.Dataset), (
                "Unsupported type of dataset({}): {}".format(ds, type(ds)))
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


# (TensorFlow 2.3!) there is a bug on math_ops.reduce_all under Horovod+fp16+XLA
def _refacor_is_all_finite(grads):
    """Returns a scalar boolean tensor indicating if all gradients are finite."""
    is_finite_per_grad = []
    for g in grads:
        if g is None:
            continue
        # is_not_finite = tf.logical_not(tf.math.is_finite(g))
        # reduced_is_finite = tf.logical_not(tf.reduce_any(is_not_finite))

        is_finite = tf.math.is_finite(g)
        reduced_is_finite = tf.equal(tf.size(g, out_type=tf.int64),
                                     tf.math.reduce_sum(tf.cast(is_finite, tf.int64)))
        is_finite_per_grad.append(reduced_is_finite)
    return tf.math.reduce_all(is_finite_per_grad)


class RevisedDynamicLossScale(DynamicLossScale):
    def update(self, grads):
        """Updates loss scale based on if gradients are finite in current step."""
        grads = nest.flatten(grads)
        if distribution_strategy_context.has_strategy():
            distribution = distribution_strategy_context.get_cross_replica_context()

            def get_is_finite(grads):
                is_finite = _refacor_is_all_finite(grads)
                # We cast to float, because we cannot reduce booleans with
                # DistributionStrategy.
                return tf.cast(is_finite, dtypes.float32)

            is_finite_float = distribution.extended.call_for_each_replica(
                get_is_finite, args=(grads,))
            reduced_is_finite_float = distribution.reduce(reduce_util.ReduceOp.SUM,
                                                          is_finite_float, axis=None)
            is_finite = tf.equal(reduced_is_finite_float,
                                 distribution.num_replicas_in_sync)
        else:
            is_finite = _refacor_is_all_finite(grads)

        def update_if_finite_grads():
            """Update assuming the gradients are finite."""

            def incr_loss_scale():
                new_loss_scale = self._current_loss_scale * self._multiplier
                return control_flow_ops.group(
                    _assign_if_finite(self._current_loss_scale, new_loss_scale),
                    self._num_good_steps.assign(0))

            return control_flow_ops.cond(
                self._num_good_steps + 1 >= self._increment_period,
                incr_loss_scale, lambda: _op_in_graph_mode(
                    self._num_good_steps.assign_add(1)))

        def update_if_not_finite_grads():
            """Update assuming the gradients are nonfinite."""

            new_loss_scale = tf.math.maximum(
                self._current_loss_scale / self._multiplier, 1)
            return control_flow_ops.group(
                self._num_good_steps.assign(0),
                self._current_loss_scale.assign(new_loss_scale))

        update_op = control_flow_ops.cond(is_finite, update_if_finite_grads,
                                          update_if_not_finite_grads)
        should_apply_gradients = is_finite
        return update_op, should_apply_gradients
