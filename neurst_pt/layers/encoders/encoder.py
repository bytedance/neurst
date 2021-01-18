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
""" Base Encoder class. """
from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod

import six
import torch.nn as nn

from neurst.utils.configurable import extract_constructor_params


@six.add_metaclass(ABCMeta)
class Encoder(nn.Module):
    """ Base class for encoders. """
    REGISTRY_NAME = "encoder"

    def __init__(self, **kwargs):
        """ Initializes the parameters of the encoders. """
        self._params = extract_constructor_params(locals(), verbose=False)
        super(Encoder, self).__init__()

    @abstractmethod
    def forward(self, inputs, inputs_padding, is_training=True):
        """ Encodes the inputs.

        Args:
            inputs: The embedded input, a float tensor with shape
                [batch_size, max_length, embedding_dim].
            inputs_padding: A float tensor with shape [batch_size, max_length],
                indicating the padding positions, where 1.0 for padding and
                0.0 for non-padding.
            is_training: A bool, whether in training mode or not.

        Returns:
            The encoded output with shape [batch_size, max_length, hidden_size]
        """
        raise NotImplementedError
