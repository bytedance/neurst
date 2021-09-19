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

from neurst.optimizers.schedules import register_lr_schedule
from neurst.utils import compat
from neurst.utils.flags_core import Flag


@register_lr_schedule("inverse_sqrt")
class InverseSquareRootSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule."""

    def __init__(self, args):
        """Initialize configuration of the learning rate schedule.

        Args:
          args: A dict of full parameters.
        """
        super(InverseSquareRootSchedule, self).__init__()
        self._initial_step = compat.get_registered_initial_step()
        logging.info(f"Initialize InverseSquareRootSchedule from global step={self._initial_step}. ")
        self._initial_step = tf.convert_to_tensor(self._initial_step, dtype=tf.float32)
        self._lr = tf.cast(args["peak_lr"], tf.float32)
        self._init_lr = tf.cast(args["init_lr"], tf.float32)
        self._warmup_steps = tf.cast(args["warmup_steps"], tf.float32)
        self._lr_step = (self._lr - self._init_lr) / self._warmup_steps
        self._decay_factor = self._lr * self._warmup_steps ** 0.5

    @staticmethod
    def class_or_method_args():
        return [
            Flag("peak_lr", dtype=Flag.TYPE.FLOAT, default=5e-4,
                 help="The configured lr."),
            Flag("init_lr", dtype=Flag.TYPE.FLOAT, default=0.,
                 help="The initial lr."),
            Flag("warmup_steps", dtype=Flag.TYPE.INTEGER, default=4000,
                 help="The number of steps required for linear warmup."),
        ]

    def __call__(self, global_step):
        """Calculate learning rate with linear warmup and rsqrt decay.

        Args:
          global_step: An integer, the current global step used for learning rate
            calculation.

        Returns:
          A float, the learning rate needs to be used for current global step.
        """
        with tf.name_scope('learning_rate_schedule'):
            global_step = tf.cast(global_step, tf.float32) + self._initial_step + 1.
            is_warmup = tf.cast(tf.less(global_step, self._warmup_steps), tf.float32)
            warmup_lr = self._init_lr + global_step * self._lr_step
            final_lr = self._decay_factor * global_step ** -0.5
            return is_warmup * warmup_lr + (1. - is_warmup) * final_lr

    def get_config(self):
        """Get the configuration of the learning rate schedule."""
        return {
            "peak_lr": float(self._lr.numpy()),
            "init_lr": float(self._lr.numpy()),
            "warmup_steps": int(self._warmup_steps.numpy()),
        }
