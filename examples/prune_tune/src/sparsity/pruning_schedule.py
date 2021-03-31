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
"""Pruning Schedule classes to control weight_pruning rate during training.
    The original link: https://github.com/tensorflow/model-optimization/blob/master/
    tensorflow_model_optimization/python/core/sparsity/keras/pruning_schedule.py
"""
from abc import ABCMeta, abstractmethod

import six
import tensorflow as tf

from neurst.utils.flags_core import Flag
from neurst.utils.registry import setup_registry


@six.add_metaclass(ABCMeta)
class PruningSchedule(object):
    """ Specifies when to prune layer and the sparsity(%) at each training step.
    PruningSchedule controls weight_pruning during training by notifying at each step
    whether the layer's weights should be pruned or not, and the sparsity(%) at
    which they should be pruned.
    It can be invoked as a `callable` by providing the training `step` Tensor. It
    returns a tuple of bool and float tensors.
    ```python
      should_prune, sparsity = pruning_schedule(step)
    ```
    You can inherit this class to write your own custom weight_pruning schedule.
    """
    REGISTRY_NAME = "pruning_schedule"

    def __init__(self,
                 target_sparsity=0.,
                 begin_pruning_step=0,
                 end_pruning_step=-1,
                 pruning_frequency=100,
                 **kwargs):
        if begin_pruning_step < 0:
            raise ValueError('begin_pruning_step should be >= 0')
        if end_pruning_step != -1:
            if end_pruning_step < begin_pruning_step:
                raise ValueError("begin_pruning_step should be <= end_pruning_step "
                                 "if end_pruning_step != -1")
        if pruning_frequency <= 0:
            raise ValueError("pruning_frequency should be > 0")
        if not 0.0 <= target_sparsity < 1.0:
            raise ValueError("target_sparsity must be in range [0,1)")
        self._target_sparsity = target_sparsity
        self._begin_pruning_step = begin_pruning_step
        self._end_pruning_step = end_pruning_step
        self._pruning_frequency = pruning_frequency
        self._decay_pruning_steps = end_pruning_step - begin_pruning_step

    @property
    def target_sparsity(self):
        return self._target_sparsity

    def should_prune_in_step(self, step):
        """Checks if weight_pruning should be applied in the current training step.
        Pruning should only occur within the [`begin_step`, `end_step`] range every
        `frequency` number of steps.
        Args:
          step: Current training step.
        Returns:
          True/False, if weight_pruning should be applied in current step.
        """
        is_in_pruning_range = tf.math.logical_and(
            tf.math.greater_equal(step, self._begin_pruning_step),
            # If end_pruning_step is negative, keep weight_pruning forever!
            tf.math.logical_or(
                tf.math.less_equal(step, self._end_pruning_step),
                tf.math.less(self._end_pruning_step, 0)))

        is_pruning_turn = tf.math.equal(
            tf.math.floormod(tf.math.subtract(step, self._begin_pruning_step),
                             self._pruning_frequency), 0)
        return tf.math.logical_and(is_in_pruning_range, is_pruning_turn)

    @staticmethod
    def class_or_method_args():
        return [Flag("target_sparsity", dtype=Flag.TYPE.FLOAT, default=0.,
                     help="The sparsity at final (the percent of zero values)."),
                Flag("begin_pruning_step", dtype=Flag.TYPE.INTEGER, default=0,
                     help="Step at which to begin weight_pruning."),
                Flag("end_pruning_step", dtype=Flag.TYPE.INTEGER, default=-1,
                     help="Step at which to begin weight_pruning."),
                Flag("pruning_frequency", dtype=Flag.TYPE.INTEGER, default=100,
                     help="Only apply weight_pruning every `pruning_frequency` steps.")]

    @abstractmethod
    def sparsity_in_step(self, step):
        """Returns the sparsity(%) to be applied.
        If the returned sparsity(%) is 0, weight_pruning is ignored for the step.
        Args:
          step: Current step in graph execution.
        Returns:
          Sparsity (%) that should be applied to the weights for the step.
        """
        raise NotImplementedError(
            "PruningSchedule implementation must override sparsity_in_step")

    @classmethod
    def new(cls, args: dict):
        if args["target_sparsity"] == 0:
            return None
        return cls(**args)


build_pruning_schedule, register_pruning_schedule = setup_registry(
    PruningSchedule.REGISTRY_NAME, base_class=PruningSchedule,
    create_fn="new", verbose_creation=True)


@register_pruning_schedule
class ConstantSparsity(PruningSchedule):
    """Pruning schedule with constant sparsity(%) throughout training."""

    def __init__(self, **kwargs):
        super(ConstantSparsity, self).__init__(**kwargs)

    def sparsity_in_step(self, step):
        return tf.constant(self.target_sparsity, dtype=tf.float32)


@register_pruning_schedule
class PolynomialDecay(PruningSchedule):
    """Pruning Schedule with a PolynomialDecay function."""

    def __init__(self, initial_sparsity=0., polynomial_power=3, **kwargs):
        """Initializes a Pruning schedule with a PolynomialDecay function.
        Pruning rate grows rapidly in the beginning from initial_sparsity, but then
        plateaus slowly to the target sparsity. The function applied is
        current_sparsity = target_sparsity + (initial_sparsity - target_sparsity)
              * (1 - (step - begin_step)/(end_step - begin_step)) ^ exponent
        which is a polynomial decay function. See
        [paper](https://arxiv.org/abs/1710.01878).
        Args:
          initial_sparsity: Sparsity (%) at which weight_pruning begins.
          polynomial_power: Exponent to be used in the sparsity function.
        """
        super(PolynomialDecay, self).__init__(**kwargs)
        if not 0.0 <= initial_sparsity < 1.0:
            raise ValueError("initial_sparsity must be in range [0,1)")
        self._initial_sparsity = initial_sparsity
        self._power = polynomial_power
        if self._end_pruning_step < 0:
            raise ValueError("end_pruning_step must > 0 when using PolynomialDecay.")

    @staticmethod
    def class_or_method_args():
        this_args = super(PolynomialDecay, PolynomialDecay).class_or_method_args()
        this_args += [
            Flag("initial_sparsity", dtype=Flag.TYPE.FLOAT, default=0.,
                 help="Sparsity (%) at which weight_pruning begins"),
            Flag("polynomial_power", dtype=Flag.TYPE.FLOAT, default=3,
                 help="Exponent to be used in the sparsity function")
        ]
        return this_args

    def sparsity_in_step(self, step):
        p = tf.math.divide(tf.cast(step - self._begin_pruning_step, tf.float32), self._decay_pruning_steps)
        p = tf.math.minimum(1.0, tf.math.maximum(0.0, p))
        sparsity = ((self._initial_sparsity - self._target_sparsity)
                    * ((1 - p) ** self._power)) + self._target_sparsity
        return sparsity
