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
from absl import logging
from tensorflow.python.distribute import distribution_strategy_context, reduce_util
from tensorflow.python.ops import control_flow_ops

from neurst.utils.compat import IS_PREV_TF_2_4_0

if IS_PREV_TF_2_4_0:
    from tensorflow.python.training.experimental.loss_scale import DynamicLossScale as TfDls
    from tensorflow.python.training.experimental.loss_scale import _assign_if_finite, _op_in_graph_mode
else:
    from tensorflow.python.keras.mixed_precision.loss_scale_optimizer import _assign_if_finite
    from tensorflow.python.keras.mixed_precision.loss_scale_optimizer import _DynamicLossScaleState as TfDls
    from tensorflow.python.keras.mixed_precision.loss_scale_optimizer import _op_in_graph_mode


# there is a bug on tf.reduce_all under XLA
#    ==> tf.reduce_all() returns 0 for a large tensor filled with 1s.
def _refactor_is_all_finite(grads):
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


class RevisedDynamicLossScale(TfDls):
    def __init__(self, *args, **kwargs):
        logging.info("Using RevisedDynamaicLossScale under FP16 to ensure tf.reduce_all behaviour, "
                     "especially under XLA")
        super(RevisedDynamicLossScale, self).__init__(*args, **kwargs)

    def update(self, grads):
        """Updates loss scale based on if gradients are finite in current step."""
        counter = self._num_good_steps if IS_PREV_TF_2_4_0 else self.counter
        growth_steps = self._increment_period if IS_PREV_TF_2_4_0 else self.growth_steps
        current_loss_scale = self._current_loss_scale if IS_PREV_TF_2_4_0 else self.current_loss_scale
        multiplier = self._multiplier if IS_PREV_TF_2_4_0 else self.multiplier
        grads = tf.nest.flatten(grads)
        if distribution_strategy_context.has_strategy():
            if IS_PREV_TF_2_4_0:
                distribution = distribution_strategy_context.get_cross_replica_context()
            else:
                distribution = distribution_strategy_context.get_strategy()

            def get_is_finite(grads):
                is_finite = _refactor_is_all_finite(grads)  # !!!!!!!!!
                # We cast to float, because we cannot reduce booleans with
                # DistributionStrategy.
                return tf.cast(is_finite, tf.float32)

            is_finite_float = distribution.extended.call_for_each_replica(
                get_is_finite, args=(grads,))
            reduced_is_finite_float = distribution.reduce(reduce_util.ReduceOp.SUM,
                                                          is_finite_float, axis=None)
            is_finite = tf.equal(reduced_is_finite_float,
                                 distribution.num_replicas_in_sync)
        else:
            is_finite = _refactor_is_all_finite(grads)

        def update_if_finite_grads():
            """Update assuming the gradients are finite."""

            def incr_loss_scale():
                new_loss_scale = current_loss_scale * multiplier
                return control_flow_ops.group(
                    _assign_if_finite(current_loss_scale, new_loss_scale),
                    counter.assign(0))

            return control_flow_ops.cond(
                counter + 1 >= growth_steps,
                incr_loss_scale,
                lambda: _op_in_graph_mode(counter.assign_add(1)))

        def update_if_not_finite_grads():
            """Update assuming the gradients are nonfinite."""

            new_loss_scale = tf.math.maximum(current_loss_scale / multiplier, 1)
            return control_flow_ops.group(
                counter.assign(0),
                current_loss_scale.assign(new_loss_scale))

        update_op = control_flow_ops.cond(is_finite, update_if_finite_grads,
                                          update_if_not_finite_grads)
        should_apply_gradients = is_finite
        return update_op, should_apply_gradients
