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
import tensorflow as tf


@six.add_metaclass(ABCMeta)
class BaseModel(tf.keras.Model):
    REGISTRY_NAME = "model"

    def __init__(self, args: dict, name=None):
        self._args = args
        super(BaseModel, self).__init__(name=name)

    @property
    def args(self):
        return self._args

    @staticmethod
    def class_or_method_args():
        return []

    @classmethod
    def new(cls, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def call(self, inputs, is_training=True):
        """ Forward pass of the model.

        Args:
            inputs: A dict of model inputs.
            is_training: A bool, whether in training mode or not.

        Returns:
            The model output.
        """
        raise NotImplementedError
