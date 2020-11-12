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
from absl import logging

from neurst.models.model import BaseModel


class EncoderDecoderEnsembleModel(BaseModel):
    """ Defines the basic sequence to sequence model. """

    def __init__(self, encoder_decoder_models):
        """ Initializes the ensemble model.

        Args:
            encoder_decoder_models: A list of encoder decoder models.
        """
        logging.info("EncoderDecoderEnsembleModel is only available for generation.")
        super(EncoderDecoderEnsembleModel, self).__init__(None, None)
        self._encoder_decoder_models = encoder_decoder_models

    @property
    def model_num(self):
        return len(self._encoder_decoder_models)

    @classmethod
    def new(cls, encoder_decoder_models):
        """ Creates keras model and model object.

        Args:
            encoder_decoder_models: A list of encoder decoder models.

        Returns:
            An encoder decoder ensemble model.
        """
        logging.info("Create model: {}".format(cls.__name__))
        return cls(encoder_decoder_models)

    def get_symbols_to_logits_fn(self, inputs, is_training=False, is_inference=True,
                                 decode_padded_length=None):
        """ Prepares for decoding.

        Args:
            inputs: A dict of model inputs.
                - src: int tensor with shape [batch_size, src_input_length].
                - src_padding: float tensor with shape [batch_size, src_input_length].
                - trg_input: int tensor with shape [batch_size, trg_length].
            is_training: A bool, whether in training mode or not.
            is_inference: A bool, whether in generation mode or not.
            decode_padded_length: The maximum decoding length when inference, for creating
                static-shape cache.

        Returns:  A tuple of (decoding_internal_states, decoder_input, symbol_to_logit_fn)
        """
        _ = is_training
        _ = is_inference
        is_training = False
        is_inference = True

        symbols_to_logits_fn_per_model = []
        final_generation_initializer = None
        for model in self._encoder_decoder_models:
            model_symbols_to_logits_fn, model_generation_initializer = model.get_symbols_to_logits_fn(
                inputs, is_training, is_inference, decode_padded_length=decode_padded_length)
            symbols_to_logits_fn_per_model.append(model_symbols_to_logits_fn)
            if final_generation_initializer is None:
                final_generation_initializer = model_generation_initializer
                final_generation_initializer["decoder_internal_cache"] = [
                    model_generation_initializer["decoder_internal_cache"]]
            else:
                final_generation_initializer["decoder_internal_cache"].append(
                    model_generation_initializer["decoder_internal_cache"])

        def symbols_to_logits_fn(symbols, cache_per_model, time=None):
            """ Generate logits for next potential IDs

            Args:
                symbols: Current decoded sequence.
                cache_per_model: A list of dictionaries of values storing the encoder output, encoder-decoder
                    attention bias, and previous decoder attention values.
                time: Loop index, or None for transformer training

            Returns: The logits Tensor.
            """
            logits_per_model = [
                fn(symbols, cache, time=time) for fn, cache in
                zip(symbols_to_logits_fn_per_model, cache_per_model)]
            return logits_per_model

        return symbols_to_logits_fn, final_generation_initializer

    def call(self, inputs, is_training=True):
        raise NotImplementedError("No need to implement call function for EncoderDecoderEnsembleModel.")
