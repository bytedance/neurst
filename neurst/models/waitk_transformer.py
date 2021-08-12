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

from neurst.layers.decoders import build_decoder
from neurst.layers.encoders import build_encoder
from neurst.models import register_model
from neurst.models.model_utils import input_length_to_padding
from neurst.models.transformer import Transformer


@register_model
class WaitkTransformer(Transformer):
    """ Defines the WaitkTransformer model. """

    @property
    def wait_k(self):
        if not hasattr(self, "_wait_k"):
            self._wait_k = None
        return self._wait_k

    @wait_k.setter
    def wait_k(self, val):
        self._wait_k = val

    @classmethod
    def new(cls, args, src_meta, trg_meta, waitk_lagging, name=None):
        """ Builds a sequence to sequence model.

        Args:
            args: A dict containing all model parameters.
            src_meta: A dict containing source-side vocabulary meta data, e.g. eos_id, vocab_size.
            trg_meta: A dict containing target-side vocabulary meta data, e.g. eos_id, vocab_size.
            name: The name of the model.

        Returns:
            An encoder decoder model.
        """
        # build source and target modality
        src_modality, trg_modality = cls.build_modalities(args, src_meta, trg_meta)
        encoder_params, decoder_params = {}, {}
        for f in cls.class_or_method_args():
            if f.name in args:
                if f.name.startswith("encoder."):
                    encoder_params[f.name[8:]] = args[f.name]
                elif f.name.startswith("decoder."):
                    decoder_params[f.name[8:]] = args[f.name]
        # build encoder and decoder
        encoder = build_encoder({
            "encoder.class": "TransformerEncoder",
            "encoder.params": encoder_params})
        decoder = build_decoder({
            "decoder.class": "TransformerDecoder",
            "decoder.params": decoder_params})
        model = cls(args, src_meta, trg_meta, src_modality, trg_modality,
                    encoder, decoder, name=name)
        model.wait_k = waitk_lagging
        _ = model({"src": tf.convert_to_tensor([[1, 2, 3]], tf.int64),
                   "src_padding": tf.convert_to_tensor([[0, 0., 0]], tf.float32),
                   "trg_input": tf.convert_to_tensor([[1, 2, 3]], tf.int64)})
        return model

    @classmethod
    def build_model_args_by_name(cls, name):
        args = None
        if name.startswith("waitk_transformer_"):
            args = Transformer.build_model_args_by_name(name[6:])
        elif name.startswith("waitktransformer_"):
            args = Transformer.build_model_args_by_name(name[5:])
        if args is not None:
            args["model.class"] = cls.__name__
            args["model.params"]["encoder.attention_monotonic"] = True
        return args

    def get_decoder_output(self, symbols, cache, time=None,
                           is_training=False, decode_padded_length=None):
        """ Forward pass of the decoder.

        Args:
            symbols: Current decoded sequence.
            cache: A dictionary of values storing the encoder output, encoder-decoder
                attention bias, and previous decoder attention values.
            time: Loop index, or None for transformer training.
            is_training: Whether is under training or not.
            decode_padded_length: The maximum decoding length when inference, for creating
                static-shape cache.

        Returns: A Tensor.
        """
        inputs = self._trg_modality(symbols, time=time)
        lagging = self.wait_k
        if is_training:
            if tf.nest.is_nested(lagging):
                lagging = tf.convert_to_tensor(lagging, dtype=tf.int32)
                lagging = lagging[tf.random.uniform(shape=[], maxval=tf.shape(lagging)[0], dtype=tf.int32)]
        elif tf.nest.is_nested(lagging):
            lagging = lagging[0]
        if time is not None:  # during inference
            lagging += time
        if decode_padded_length is None:
            decoder_output = self._decoder(inputs, cache, decode_lagging=lagging,
                                           is_training=is_training, decode_loop_step=None)
        else:
            decoder_output = self._decoder(inputs, cache, decode_lagging=lagging,
                                           is_training=is_training, decode_loop_step=time)
        return decoder_output

    def incremental_encode(self, inputs, encoder_cache, decoder_cache, time=None):
        src_ndims = tf.convert_to_tensor(inputs["src"], tf.int32).get_shape().ndims
        assert not (src_ndims == 1 and time is None)
        embedded_inputs = self._src_modality(inputs["src"], time=time)
        src_padding = inputs.get("src_padding", None)
        if src_padding is None:
            # [batch_size, max_len]
            src_padding = input_length_to_padding(inputs["src_length"],
                                                  (tf.shape(embedded_inputs)[1] if src_ndims == 2 else 1))
        encoder_outputs, encoder_cache = self._encoder.incremental_encode(embedded_inputs, encoder_cache, time)
        decoder_cache = self._decoder.update_incremental_cache(decoder_cache, encoder_outputs, src_padding)
        return encoder_cache, decoder_cache

    def incremental_decode(self, symbols, cache, time=None):
        # ensure encoder_outputs != None and encoder_inputs_padding != None
        #   when first calling this function
        decoder_output = super(WaitkTransformer, self).get_decoder_output(symbols, cache, time,
                                                                          is_training=False,
                                                                          decode_padded_length=None)
        logits = self.output_logits_layer(decoder_output)
        return logits, cache
