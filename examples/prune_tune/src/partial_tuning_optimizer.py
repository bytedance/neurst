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


def create_partial_tuning_optimizer(optimizer, model, load_mask):
    """ Returns an optimizer that will make variable sparse after applying gradients.

    Args:
        optimizer: The optimizer.
        model: The keras model.
        load_mask:
    """

    class _PartialTuningOptimizer(tf.keras.optimizers.Optimizer):

        def __init__(self, **kwargs):
            super(self.__class__, self).__init__(**kwargs)
            # noprune_weights = []
            self._model_weights = []
            for var in model.trainable_weights:
                self._model_weights.append(var)
            self._partial_tuning_vars = self._create_variable_masks()

        def _create_variable_masks(self):
            original_weights = [
                tf.Variable(
                    (tf.cast(weight, weight.dtype.base_dtype)),
                    dtype=weight.dtype.base_dtype,
                    trainable=False,
                    synchronization=tf.VariableSynchronization.ON_READ,
                    aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA) for weight in self._model_weights
            ]
            if load_mask is None:
                original_masks = [
                    tf.Variable(
                        (tf.cast(tf.math.not_equal(weight, 0.), weight.dtype.base_dtype)),
                        dtype=weight.dtype.base_dtype,
                        trainable=False,
                        synchronization=tf.VariableSynchronization.ON_READ,
                        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA) for weight in self._model_weights
                ]
                tuning_masks = [
                    tf.Variable(
                        (tf.cast(tf.math.equal(weight, 0.), weight.dtype.base_dtype)),
                        dtype=weight.dtype.base_dtype,
                        trainable=False,
                        synchronization=tf.VariableSynchronization.ON_READ,
                        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA) for weight in self._model_weights
                ]
            else:
                original_masks = [
                    tf.Variable(
                        (tf.cast(mask, mask.dtype.base_dtype)),
                        dtype=mask.dtype.base_dtype,
                        trainable=False,
                        synchronization=tf.VariableSynchronization.ON_READ,
                        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA) for mask in load_mask
                ]
                tuning_masks = [
                    tf.Variable(
                        (tf.cast(tf.math.equal(mask, 0.), mask.dtype.base_dtype)),
                        dtype=mask.dtype.base_dtype,
                        trainable=False,
                        synchronization=tf.VariableSynchronization.ON_READ,
                        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA) for mask in load_mask
                ]

            return list(zip(self._model_weights, original_weights, original_masks, tuning_masks))

        def apply_gradients(self,
                            grads_and_vars,
                            name=None,
                            experimental_aggregate_gradients=True):
            results = super(self.__class__, self).apply_gradients(
                grads_and_vars, name, experimental_aggregate_gradients)
            for weight, original_weight, original_mask, tuning_mask in self._partial_tuning_vars:
                masked_weight = original_weight * tf.cast(original_mask, weight.dtype.base_dtype) + weight * tf.cast(
                    tuning_mask, weight.dtype.base_dtype)
                weight.assign(masked_weight)
            return results

    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_PartialTuningOptimizer.__dict__))
    new_optimizer = cls.from_config(optimizer.get_config())
    new_optimizer._HAS_AGGREGATE_GRAD = optimizer._HAS_AGGREGATE_GRAD
    return new_optimizer
