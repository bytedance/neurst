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
from abc import ABCMeta, abstractmethod

import six

from neurst.training.callbacks import CentralizedCallback
from neurst.utils import compat
from neurst.utils.flags_core import Flag


@six.add_metaclass(ABCMeta)
class Validator(CentralizedCallback):
    REGISTRY_NAME = "validator"

    def __init__(self, args):
        super(Validator, self).__init__()
        self._eval_steps = args["eval_steps"]
        self._eval_start_at = args["eval_start_at"]
        self._eval_on_begin = args["eval_on_begin"]

    @staticmethod
    def class_or_method_args():
        return [
            Flag("eval_steps", dtype=Flag.TYPE.INTEGER, default=1000,
                 help="The steps between two validation steps."),
            Flag("eval_start_at", dtype=Flag.TYPE.INTEGER, default=0,
                 help="The step to start validation process."),
            Flag("eval_on_begin", dtype=Flag.TYPE.BOOLEAN, default=False,
                 help="Whether to trigger evaluation on the beginning.")
        ]

    @abstractmethod
    def build(self, strategy, task, model):
        """ Builds the validator and returns self. """
        return self

    @abstractmethod
    def validate(self, step):
        """ Validation process. """
        raise NotImplementedError

    def custom_on_train_batch_end(self, step, logs=None):
        _ = logs
        if step >= self._eval_start_at and step % self._eval_steps == 0:
            self.validate(step)

    def on_train_begin(self, logs=None):
        super(Validator, self).on_train_begin(logs)
        if self._eval_on_begin:
            self.validate(compat.get_registered_initial_step())
