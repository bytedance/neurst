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
import re

import tensorflow as tf
from absl import logging

from neurst.sparsity.pruning_schedule import PruningSchedule


def create_pruning_optimizer(optimizer, model,
                             pruning_schedule: PruningSchedule,
                             pruning_variable_pattern=None,
                             nopruning_variable_pattern=None,
                             keep_prune_property=False):
    """ Returns an optimizer that will make variable sparse after applying gradients.

    Args:
        optimizer: The optimizer.
        model: The keras model.
        pruning_schedule: The sparsity schedule for weight weight_pruning.
        pruning_variable_pattern: The regular expression that indicates the variables will be pruned.
        nopruning_variable_pattern: The regular expression that indicates the variables
            will NOT be pruned (will take effect if `pruning_variable_pattern`=None).
        keep_prune_property: if True, deduce weight_pruning mask according to the variables, else initialize
            the mask to ones.
    """

    class _PruningOptimizer(tf.keras.optimizers.Optimizer):

        def __init__(self, **kwargs):
            super(self.__class__, self).__init__(**kwargs)
            self._keep_prune_property = keep_prune_property
            self._pruning_schedule = pruning_schedule
            self._prune_weigths = []
            noprune_weights = []
            for var in model.trainable_weights:
                varname = var.name.split(":")[0]
                if pruning_variable_pattern is not None:
                    if re.search(pruning_variable_pattern, varname):
                        self._prune_weigths.append(var)
                    else:
                        noprune_weights.append(var)
                elif nopruning_variable_pattern is not None:
                    if re.search(nopruning_variable_pattern, varname):
                        noprune_weights.append(var)
                    else:
                        self._prune_weigths.append(var)
                else:
                    self._prune_weigths.append(var)
            if len(self._prune_weigths) == 0:
                logging.info("Pruning: all trainable weights will be pruned. ")
            else:
                logging.info(f"Pruning: following {len(self._prune_weigths)} weights will be pruned: ")
                for var in self._prune_weigths:
                    logging.info(f"  {var.name.split(':')[0]}")
                logging.info("")
                logging.info(f"Pruning: following {len(self._prune_weigths)} weights will NOT be pruned: ")
                for var in noprune_weights:
                    logging.info(f"  {var.name.split(':')[0]}")
            self._prune_vars = self._create_variable_masks()

        def _create_variable_masks(self):
            masks = [
                tf.Variable(
                    (tf.cast(tf.math.not_equal(weight, 0.), weight.dtype.base_dtype)
                     if self._keep_prune_property else tf.ones_like(weight)),
                    dtype=weight.dtype.base_dtype,
                    trainable=False,
                    synchronization=tf.VariableSynchronization.ON_READ,
                    aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA) for weight in self._prune_weigths
            ]
            thresholds = [
                tf.Variable(
                    tf.convert_to_tensor(0., dtype=tf.float32),
                    trainable=False,
                    synchronization=tf.VariableSynchronization.ON_READ,
                    aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA) for _ in self._prune_weigths
            ]
            return list(zip(self._prune_weigths, masks, thresholds))

        def apply_gradients(self,
                            grads_and_vars,
                            name=None,
                            experimental_aggregate_gradients=True):
            results = super(self.__class__, self).apply_gradients(
                grads_and_vars, name, experimental_aggregate_gradients)
            should_update_mask = self._pruning_schedule.should_prune_in_step(self.iterations)

            def _apply_mask():
                for weight, mask, _ in self._prune_vars:
                    masked_weight = weight * tf.cast(mask, weight.dtype.base_dtype)
                    weight.assign(masked_weight)

            def _update_mask():
                sparsity = self._pruning_schedule.sparsity_in_step(self.iterations)
                for weight, mask, threshold in self._prune_vars:
                    abs_weight = tf.math.abs(weight)
                    k = tf.cast(tf.math.round(tf.cast(tf.size(weight), tf.float32) * (1. - sparsity)), tf.int32)
                    # sort the entire array
                    values, _ = tf.math.top_k(tf.reshape(abs_weight, [-1]), k=k)
                    # grab the (k-1)-th value
                    current_threshold = tf.gather(values, k - 1)
                    new_mask = tf.cast(tf.math.greater_equal(abs_weight, current_threshold),
                                       weight.dtype.base_dtype)
                    mask.assign(new_mask)
                    threshold.assign(current_threshold)
                    weight.assign(weight * tf.cast(new_mask, weight.dtype.base_dtype))

            tf.cond(should_update_mask, true_fn=_update_mask, false_fn=_apply_mask)
            return results

    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_PruningOptimizer.__dict__))
    new_optimizer = cls.from_config(optimizer.get_config())
    new_optimizer._HAS_AGGREGATE_GRAD = optimizer._HAS_AGGREGATE_GRAD
    return new_optimizer
