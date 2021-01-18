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
import tensorflow as tf

from neurst.training.revised_dynamic_loss_scale import RevisedDynamicLossScale
from neurst.utils.compat import IS_PREV_TF_2_4_0

if IS_PREV_TF_2_4_0:
    from tensorflow.python.keras.mixed_precision.experimental.loss_scale_optimizer import LossScaleOptimizer
else:
    from tensorflow.python.keras.mixed_precision.loss_scale_optimizer import LossScaleOptimizer


class HorovodDistributedLossScaleOptimizer(LossScaleOptimizer):
    _HAS_AGGREGATE_GRAD = True

    def __init__(self, compression=None,
                 sparse_as_dense=False,
                 device_dense="",
                 device_sparse="",
                 hvd_backend="horovod",
                 **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        # using revised loss scale
        self._loss_scale = RevisedDynamicLossScale(
            initial_loss_scale=2 ** 15, growth_steps=2000, multiplier=2)
        self._track_trackable(self._loss_scale, "loss_scale", overwrite=True)
        self._device_dense = device_dense
        self._device_sparse = device_sparse
        self._hvd_backend = hvd_backend
        self._compression = compression
        self._sparse_as_dense = sparse_as_dense
        self._aggregated_gradients = False
        if hvd_backend == "horovod":
            import horovod.tensorflow as hvd

            self._allreduce_grads = hvd._make_allreduce_grads_fn(
                "DistributedLossScaleOptimizer", self._device_dense, self._device_sparse,
                compression, sparse_as_dense, hvd.Average, 1.0, 0)
        else:
            assert hvd_backend == "byteps", f"Unknown `hvd_backend`={hvd_backend}"

    def _aggregate_gradients(self, grads_and_vars):
        grads, vars = list(zip(*grads_and_vars))
        aggregated_grads = (self._allreduce(grads) if self._hvd_backend == "horovod"
                            else self._push_pull(grads))
        return list(zip(aggregated_grads, vars))

    def _allreduce(self, grads):
        self._aggregated_gradients = True
        return self._allreduce_grads(grads)

    def _push_pull(self, grads):
        self._aggregated_gradients = True
        import byteps.tensorflow as bps
        if bps.size() > 1:
            averaged_gradients = []
            with tf.name_scope("DistributedLossScaleOptimizer_Push_Pull") as scope:
                for grad in grads:
                    if grad is not None:
                        if self._sparse_as_dense and isinstance(grad, tf.IndexedSlices):
                            grad = tf.convert_to_tensor(grad)
                        avg_grad = bps.push_pull(grad, scope,
                                                 device_dense=self._device_dense,
                                                 device_sparse=self._device_sparse,
                                                 compression=self._compression)
                        averaged_gradients.append(avg_grad)
                    else:
                        averaged_gradients.append(None)
                return averaged_gradients
        else:
            return grads

    def apply_gradients(self,
                        grads_and_vars,
                        name=None,
                        experimental_aggregate_gradients=True):
        results = super(self.__class__, self).apply_gradients(
            self._aggregate_gradients(grads_and_vars), name,
            experimental_aggregate_gradients)

        if not self._aggregated_gradients:
            raise Exception('`apply_gradients()` was called without a call to '
                            '`get_gradients()` or `_aggregate_gradients`. If you\'re '
                            'using TensorFlow 2.0, please specify '
                            '`experimental_run_tf_function=False` in `compile()`.')
        return results
