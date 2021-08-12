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
""" Base Decoder class. """
from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod

import six

from neurst.layers.quantization.quant_layers import QuantLayer
from neurst.utils.configurable import extract_constructor_params


@six.add_metaclass(ABCMeta)
class Decoder(QuantLayer):
    """Base class for decoders. """
    REGISTRY_NAME = "decoder"

    def __init__(self, name=None, **kwargs):
        """ Initializes the parameters of the decoder. """
        self._params = extract_constructor_params(locals(), verbose=False)
        super(Decoder, self).__init__(name=name)

    def build(self, input_shape):
        super(Decoder, self).build(input_shape)

    @abstractmethod
    def create_decoding_internal_cache(self,
                                       encoder_outputs,
                                       encoder_inputs_padding,
                                       is_inference=False,
                                       decode_padded_length=None):
        """ Creates internal cache for decoding.

        Args:
            encoder_outputs: The output tensor from encoder
                with shape [batch_size, max_input_length, hidden_size].
            encoder_inputs_padding: A float tensor with shape [batch_size, max_length],
                indicating the padding positions of `encoder_output`, where 1.0 for
                padding and 0.0 for non-padding.
            is_inference: A boolean scalar, whether in inference mode or not.
            decode_padded_length: The maximum decoding length when inference, for creating
                static-shape cache.

        Returns:
            `cache`, a dictionary containing static(e.g. encoder hidden states
            for attention) and dynamic(e.g. transformer decoding cache) tensors used
            during decoding and will be passed to `call()`. Note that, the dynamic
            tensors must store in cache["decoding_states"] for beam search use.
        """
        raise NotImplementedError

    def get_config(self):
        return self._params

    @abstractmethod
    def call(self, decoder_inputs, cache, decode_lagging=None,
             is_training=True, decode_loop_step=None):
        """ Encodes the inputs.

        Args:
            decoder_inputs: The embedded decoder input, a float tensor with shape
                [batch_size, max_target_length, embedding_dim] or
                [batch_size, embedding_dim] for one decoding step.
            cache: A dictionary, generated from self.create_decoding_internal_cache.
            decode_lagging: The lagging for streaming input. Only available for training.
            is_training: A bool, whether in training mode or not.
            decode_loop_step: An integer, step number of the decoding loop. Used only
                for autoregressive inference with static-shape cache.

        Returns:
            The decoder output with shape [batch_size, max_length, hidden_size]
            when `decoder_inputs` is a 3-d tensor or with shape
            [batch_size, hidden_size] when `decoder_inputs` is a 2-d tensor.
        """
        raise NotImplementedError
