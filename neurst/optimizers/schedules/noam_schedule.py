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


@register_lr_schedule("noam")
class NoamSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule."""

    def __init__(self, args):
        """Initialize configuration of the learning rate schedule.

        Args:
          args: A dict of full parameters.
        """
        super(NoamSchedule, self).__init__()

        self._dmodel = args["dmodel"]
        self._warmup_steps = tf.cast(args["warmup_steps"], tf.float32)
        self._initial_step = compat.get_registered_initial_step()
        logging.info("Initialize NoamSchedule from global step={}. "
                     "The result learning rate will be scaled by {}"
                     "".format(int(self._initial_step), args["initial_factor"]))
        self._initial_step = tf.convert_to_tensor(self._initial_step, dtype=tf.float32)
        _initial_learning_rate = args["initial_factor"]
        self._initial_learning_rate = tf.convert_to_tensor(_initial_learning_rate, tf.float32)
        _end_learning_rate = args["end_factor"]
        if (_end_learning_rate is not None and args["start_decay_at"] is not None
            and args["decay_steps"] is not None):
            start_decay_at = args["start_decay_at"]
            decay_steps = args["decay_steps"]
            logging.info("\tThe scaling factor will start to decay from {} to {} at step {} "
                         "and finish at step {}.".format(_initial_learning_rate, _end_learning_rate,
                                                         start_decay_at, start_decay_at + decay_steps))
        else:
            _end_learning_rate = _initial_learning_rate
            start_decay_at = 0
            decay_steps = 1
        self._end_learning_rate = tf.convert_to_tensor(_end_learning_rate, tf.float32)
        self._start_decay_at = tf.convert_to_tensor(start_decay_at, tf.float32)
        self._decay_steps = tf.convert_to_tensor(decay_steps, tf.float32)

    @staticmethod
    def class_or_method_args():
        return [
            Flag("dmodel", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The model dimension in the hidden layers."),
            Flag("warmup_steps", dtype=Flag.TYPE.INTEGER, default=4000,
                 help="The number of steps required for linear warmup."),
            Flag("initial_factor", dtype=Flag.TYPE.FLOAT, default=1.,
                 help="The initial learning rate scaling factor."),
            Flag("end_factor", dtype=Flag.TYPE.FLOAT, default=None,
                 help="The final decayed learning rate scaling factor."),
            Flag("start_decay_at", dtype=Flag.TYPE.INTEGER, default=0,
                 help="The `initial_factor` will start to decay at this step."),
            Flag("decay_steps", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The `initial_factor` will decay to `end_learning_rate` in this many steps.")
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
            step_factor = tf.maximum(tf.minimum(
                global_step - self._start_decay_at, self._decay_steps), 0.)
            learning_rate = self._end_learning_rate + (
                self._initial_learning_rate - self._end_learning_rate) * (1. - step_factor / self._decay_steps)
            learning_rate *= (self._dmodel ** -0.5)
            # Apply linear warmup
            learning_rate *= tf.minimum(1.0, global_step / self._warmup_steps)
            # Apply rsqrt decay
            learning_rate /= tf.sqrt(tf.maximum(global_step, self._warmup_steps))
            return learning_rate

    def get_config(self):
        """Get the configuration of the learning rate schedule."""
        return {
            "initial_factor": float(self._initial_learning_rate.numpy()),
            "dmodel": self._dmodel,
            "warmup_steps": int(self._warmup_steps.numpy()),
            "end_factor": float(self._end_learning_rate.numpy()),
            "start_decay_at": int(self._start_decay_at.numpy()),
            "decay_steps": int(self._decay_steps.numpy())
        }
