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


@register_lr_schedule("piecewise")
class PiecewiseSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule."""

    def __init__(self, args):
        """Initialize configuration of the learning rate schedule.

        Args:
          args: A dict of full parameters.
        """
        super(PiecewiseSchedule, self).__init__()
        self._schedule_steps = args["schedule_steps"]
        self._schedule_lrs = args["schedule_lrs"]
        assert len(self._schedule_steps) + 1 == len(self._schedule_lrs)
        self._initial_step = compat.get_registered_initial_step()
        logging.info("Initialize PiecewiseSchedule from global step={}. "
                     "The learning rate will be:".format(self._initial_step))
        for idx, (step, lr) in enumerate(zip(self._schedule_steps, self._schedule_lrs)):
            if idx == 0:
                logging.info("    linear warmup from 0~{} for {} steps".format(lr, step))
            else:
                logging.info("    {} from step={} to step={}".format(lr, self._schedule_steps[idx - 1], step))
        logging.info("    {} for step>{}".format(self._schedule_lrs[-1], self._schedule_steps[-1]))

    @staticmethod
    def class_or_method_args():
        return [
            Flag("schedule_steps", dtype=Flag.TYPE.STRING, default=None,
                 help="A list of triggered steps."),
            Flag("schedule_lrs", dtype=Flag.TYPE.STRING, default=None,
                 help="A list of learning rates."),
        ]

    def __call__(self, global_step):
        """Calculate learning rate.

        Args:
          global_step: An integer, the current global step used for learning rate
            calculation.

        Returns:
          A float, the learning rate needs to be used for current global step.
        """
        with tf.name_scope('learning_rate_schedule'):
            global_step = tf.cast(global_step, tf.float32) + self._initial_step + 1.
            pred_fn_pairs = [(tf.less(global_step, self._schedule_steps[0]),
                              lambda: self._schedule_lrs[0] / float(self._schedule_steps[0]) * global_step)]
            for step, lr in zip(self._schedule_steps[1:], self._schedule_lrs[1:-1]):
                pred_fn_pairs.append((tf.less(global_step, step), lambda: tf.constant(lr)))

            return tf.case(pred_fn_pairs, default=(lambda: tf.constant(self._schedule_lrs[-1])))

    def get_config(self):
        """Get the configuration of the learning rate schedule."""
        return {
            "schedule_steps": self._schedule_steps,
            "schedule_lrs": self._schedule_lrs
        }
