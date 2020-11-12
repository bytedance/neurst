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
class SequenceSearch(tf.keras.layers.Layer):
    REGISTRY_NAME = "search_method"

    def __init__(self):
        """ Initializes.

        Args:
            model: The model for generation.
        """
        self._model = None
        super(SequenceSearch, self).__init__()

    def set_model(self, model):
        self._model = model

    @staticmethod
    def class_or_method_args():
        return []

    def build(self, input_shape):
        super(SequenceSearch, self).build(input_shape)

    @abstractmethod
    def call(self, parsed_inp, **kwargs) -> dict:
        raise NotImplementedError
