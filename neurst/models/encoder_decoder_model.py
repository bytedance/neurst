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

from neurst.layers.common_layers import PositionEmbeddingWrapper
from neurst.layers.decoders import Decoder, build_decoder
from neurst.layers.encoders import Encoder, build_encoder
from neurst.layers.modalities.text_modalities import WordEmbeddingSharedWeights
from neurst.models import register_model
from neurst.models.model import BaseModel
from neurst.models.model_utils import input_length_to_padding
from neurst.utils.flags_core import Flag, ModuleFlag


@register_model(["seq2seq", "sequence_to_sequence", "SequenceToSequence"])
class EncoderDecoderModel(BaseModel):
    """ Defines the basic encoder-decoder model.

    All other encoder-decoder structure should inherit this class.
    """

    def __init__(self,
                 args,
                 src_meta,
                 trg_meta,
                 src_modality,
                 trg_modality,
                 encoder,
                 decoder,
                 name=None):
        """ Initializes a sequence to sequence model.

        Args:
            args: A dict, containing the model configuration.
            src_meta: A dict containing source-side vocabulary meta data, e.g. eos_id, unk_id.
            trg_meta: A dict containing target-side vocabulary meta data, e.g. eos_id, unk_id.
            src_modality: The source side modality.
            trg_modality: The target side modality.
            encoder: The encoder.
            decoder: The decoder.
            name: The name of the model.options = tf.data.Options()
        """
        super(EncoderDecoderModel, self).__init__(
            args=args, name=name or "SequenceToSequence")
        self._src_meta = src_meta
        self._trg_meta = trg_meta
        self._src_modality = src_modality
        self._trg_modality = trg_modality
        self._encoder = encoder
        self._decoder = decoder
        self._output_linear_layer = None
        if not self._args["modality.share_embedding_and_softmax_weights"]:
            self._output_linear_layer = tf.keras.layers.Dense(
                trg_meta["vocab_size"], activation=None,
                use_bias=True, name="softmax_linear")

    @staticmethod
    def class_or_method_args():
        return [
            ModuleFlag(Encoder.REGISTRY_NAME, default=None, help="The encoder."),
            ModuleFlag(Decoder.REGISTRY_NAME, default=None, help="The decoder."),
            Flag("modality.share_source_target_embedding", dtype=Flag.TYPE.BOOLEAN, default=False,
                 help="Whether to share source and target embedding table."),
            Flag("modality.share_embedding_and_softmax_weights", dtype=Flag.TYPE.BOOLEAN, default=False,
                 help="Whether to share the target embedding table and softmax weights."),
            Flag("modality.dim", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The default embedding dimension for both source and target side."),
            Flag("modality.source.dim", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The source-side embedding dimension, or `modality.dim` if not provided."),
            Flag("modality.target.dim", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The target-side embedding dimension, or `modality.dim` if not provided."),
            Flag("modality.timing", dtype=Flag.TYPE.STRING, default=None,
                 help="The arbitrary parameters for positional encoding of both source and target side."),
            Flag("modality.source.timing", dtype=Flag.TYPE.STRING, default=None,
                 help="The arbitrary parameters for source-side positional encoding, "
                      "or `modality.timing` if not provided."),
            Flag("modality.target.timing", dtype=Flag.TYPE.STRING, default=None,
                 help="The arbitrary parameters for target-side positional encoding, "
                      "or `modality.timing` if not provided.")
        ]

    @classmethod
    def new(cls, args, src_meta, trg_meta, name=None):
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
        # build encoder and decoder
        encoder = build_encoder(args)
        decoder = build_decoder(args)
        model = cls(args, src_meta, trg_meta, src_modality, trg_modality, encoder, decoder, name=name)
        _ = model({"src": tf.convert_to_tensor([[1, 2, 3]], tf.int64),
                   "src_padding": tf.convert_to_tensor([[0, 0., 0]], tf.float32),
                   "trg_input": tf.convert_to_tensor([[1, 2, 3]], tf.int64)})
        return model

    @classmethod
    def build_modality(cls, vocab_size, emb_dim, name, timing=None,
                       share_embedding_and_softmax_weights=False):
        """ Creates modality layer.

        Args:
            vocab_size: An integer, the vocabulary size.
            emb_dim: An integer, the dimension of the embedding.
            timing: A string or a dict of parameter for positional embedding parameters.
            name: A string, the layer name.
            share_embedding_and_softmax_weights: Whether to share the embedding table and softmax weight.

        Returns:
            A modality layer.
        """
        modality = WordEmbeddingSharedWeights(
            embedding_dim=emb_dim, vocab_size=vocab_size,
            share_softmax_weights=share_embedding_and_softmax_weights,
            name=name)
        # position embedding wrapper
        if timing:
            if isinstance(timing, str):
                timing = {"timing": timing}
            elif not isinstance(timing, dict):
                raise ValueError("Unknown type of timing params: {}".format(str(timing)))
            modality = PositionEmbeddingWrapper(
                embedding_layer=modality, name=name + "_posenc_wrapper", **timing)
        return modality

    @classmethod
    def build_modalities(cls, model_args, src_meta, trg_meta):
        """ Create source and target modality. """
        # if modality.source.dim is not defined, then use modality.dim as default
        src_dim = model_args["modality.source.dim"] or model_args["modality.dim"]
        # if modality.target.dim is not defined, then use modality.dim as default
        trg_dim = model_args["modality.target.dim"] or model_args["modality.dim"]
        # whether to share source and target embedding
        if model_args["modality.share_source_target_embedding"]:
            assert src_meta["vocab_size"] == trg_meta["vocab_size"], (
                "Source vocab_size should be equal to target vocab_size "
                "when modality.share_source_and_target=True")
            logging.info("Ignore `modality.source.*` when sharing source and target modality.")
            input_name = target_name = "shared_symbol_modality"
        else:
            input_name = "input_symbol_modality"
            target_name = "target_symbol_modality"
        # creates target embedding table
        target_modality = cls.build_modality(
            vocab_size=trg_meta["vocab_size"], emb_dim=trg_dim, name=target_name,
            timing=(model_args["modality.target.timing"] or model_args["modality.timing"]),
            share_embedding_and_softmax_weights=model_args["modality.share_embedding_and_softmax_weights"])

        # creates source embedding table
        if model_args["modality.share_source_target_embedding"]:
            input_modality = target_modality
        else:
            input_modality = cls.build_modality(
                vocab_size=src_meta["vocab_size"], emb_dim=src_dim, name=input_name,
                timing=(model_args["modality.source.timing"] or model_args["modality.timing"]))

        return input_modality, target_modality

    def output_logits_layer(self, features):
        """ Projects the decoder output to logits. """
        if self._output_linear_layer is None:
            return self._trg_modality(features, mode="linear")
        else:
            return self._output_linear_layer(features)

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
        if decode_padded_length is None:
            decoder_output = self._decoder(inputs, cache, is_training=is_training,
                                           decode_loop_step=None)
        else:
            decoder_output = self._decoder(inputs, cache, is_training=is_training,
                                           decode_loop_step=time)
        return decoder_output

    def get_symbols_to_logits_fn(self, inputs, is_training, is_inference,
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
        embedded_inputs = self._src_modality(inputs["src"])
        src_padding = inputs.get("src_padding", None)
        if src_padding is None:
            src_padding = input_length_to_padding(inputs["src_length"], tf.shape(embedded_inputs)[1])
        encoder_outputs = self._encoder(embedded_inputs, src_padding, is_training=is_training)
        decoder_internal_cache = self._decoder.create_decoding_internal_cache(
            encoder_outputs=encoder_outputs,
            encoder_inputs_padding=src_padding,
            is_inference=is_inference,
            decode_padded_length=decode_padded_length)

        def symbols_to_logits_fn(symbols, cache, time=None):
            """ Generate logits for next potential IDs

            Args:
                symbols: Current decoded sequence.
                cache: A dictionary of values storing the encoder output, encoder-decoder
                    attention bias, and previous decoder attention values.
                time: Loop index, or None for transformer training

            Returns: The logits Tensor.
            """
            decoder_output = self.get_decoder_output(symbols, cache, time,
                                                     is_training, decode_padded_length)
            logits = self.output_logits_layer(decoder_output)
            return logits

        generation_initializer = {
            "decoder_input": inputs["trg_input"],
            "decoder_internal_cache": decoder_internal_cache,
            "encoder_inputs_maxlen": tf.shape(encoder_outputs)[1],
            "eos_id": self._trg_meta["eos_id"],
            "unk_id": self._trg_meta["unk_id"]
        }
        return symbols_to_logits_fn, generation_initializer

    def call(self, inputs, is_training=True):
        """ Forward pass of the sequence to sequence model.

        Args:
            inputs: A dict of model inputs.
                - src: int tensor with shape [batch_size, src_input_length].
                - src_padding: float tensor with shape [batch_size, src_input_length].
                - trg_input: int tensor with shape [batch_size, trg_length].
            is_training: A bool, whether in training mode or not.

        Returns:
            A logits Tensor.
        """
        symbols_to_logits_fn, generation_initializer = self.get_symbols_to_logits_fn(
            inputs, is_training=is_training, is_inference=False)
        return symbols_to_logits_fn(generation_initializer["decoder_input"],
                                    generation_initializer["decoder_internal_cache"])
