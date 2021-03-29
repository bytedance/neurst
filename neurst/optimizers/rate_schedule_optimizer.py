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

from neurst.optimizers import register_optimizer
from neurst.utils import compat


@register_optimizer("rate_scheduled")
class RateScheduledOptimizer(tf.keras.optimizers.Optimizer):

    def __init__(self, warm_steps=10000, freeze_steps=20000,
                 controlled_varname_pattern=None, **kwargs):
        """

        Args:
            warm_steps:
            freeze_steps:
            controlled_varname_pattern: The name matched with this pattern would be controlled.
                None indicates all.
            **kwargs: The arguments for the optimizer.
        """

        super(self.__class__, self).__init__(**kwargs)
        self._warm_steps = warm_steps
        self._freeze_steps = freeze_steps
        self._controlled_varname_pattern = controlled_varname_pattern

    def reset_hparams(self, params: dict):
        for k, v in params.items():
            setattr(self, "_" + k, v)
        logging.info("[INFO] RateScheduledOptimizer: "
                     f"warm_steps={self._warm_steps}, freeze_steps={self._freeze_steps}, "
                     f"controlled_varname_pattern={self._controlled_varname_pattern}")

    def apply_gradients(self,
                        grads_and_vars,
                        name=None,
                        experimental_aggregate_gradients=True):
        new_grads_and_vars = []
        for grad, var in grads_and_vars:
            if (self._controlled_varname_pattern is None
                or re.search(self._controlled_varname_pattern, var.name) is not None):
                if grad is not None:
                    warm_steps = tf.convert_to_tensor(self._warm_steps, tf.int64)
                    freeze_steps = tf.convert_to_tensor(self._freeze_steps, tf.int64)
                    t_leq_T0 = tf.cast(self.iterations, compat.CUSTOM_GLOBAL_FLOATX) / float(self._warm_steps)
                    T0_leq_t_lt_T = 1. - (tf.cast(self.iterations - warm_steps, compat.CUSTOM_GLOBAL_FLOATX)
                                          / float(self._freeze_steps - self._warm_steps))
                    rou = (tf.cast(tf.less(self.iterations, warm_steps), t_leq_T0.dtype) * t_leq_T0
                           + tf.cast(tf.logical_and(tf.greater_equal(self.iterations, warm_steps),
                                                    tf.less(self.iterations, freeze_steps)), t_leq_T0.dtype)
                           * T0_leq_t_lt_T)
                    grad = tf.cast(rou, grad.dtype) * grad

            new_grads_and_vars.append((grad, var))

        results = super(self.__class__, self).apply_gradients(
            new_grads_and_vars, name, experimental_aggregate_gradients)

        return results

    def get_config(self):
        config = super(self.__class__, self).get_config()
        config.update({
            "warm_steps": self._warm_steps,
            "freeze_steps": self._freeze_steps,
            "controlled_varname_pattern": self._controlled_varname_pattern
        })
        return config
