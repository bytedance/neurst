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

from neurst.layers.common_layers import PositionEmbeddingWrapper
from neurst.layers.decoders.transformer_decoder import TransformerDecoder
from neurst.layers.modalities.text_modalities import WordEmbeddingSharedWeights
from neurst.models import register_model
from neurst.models.model import BaseModel
from neurst.utils.flags_core import Flag
from neurst.utils.hparams_sets import register_hparams_set


@register_model
class GPT2(BaseModel):
    """ Defines the GPT2 model. """

    def __init__(self,
                 args,
                 vocab_meta,
                 embedding,
                 decoder,
                 name=None):
        """ Initializes a GPT2 model.

        Args:
            args: A dict, containing the model configuration.
            vocab_meta: A dict containing vocabulary meta data, e.g. unk_id, mask_id, sep_id.
            embedding: The embedding layer (wrapped by positional encoding).
            decoder: The decoder.
            name: The name of the model.options = tf.data.Options()
        """
        super(GPT2, self).__init__(args, name=name or "gpt2")
        self._vocab_meta = vocab_meta
        self._embedding = embedding
        self._decoder = decoder
        self._output_linear_layer = None
        if not self._args["share_embedding_and_softmax_weights"]:
            self._output_linear_layer = tf.keras.layers.Dense(
                vocab_meta["vocab_size"], activation=None,
                use_bias=self._args["softmax_bias"], name="softmax_linear")

    @staticmethod
    def class_or_method_args():
        return [
            Flag("share_embedding_and_softmax_weights", dtype=Flag.TYPE.BOOLEAN, default=False,
                 help="Whether to share the target embedding table and softmax weights."),
            Flag("softmax_bias", dtype=Flag.TYPE.BOOLEAN, default=False,
                 help="Whether to add a bias tensor to the softmax output."),
            Flag("timing", dtype=Flag.TYPE.STRING, default=None,
                 help="The arbitrary parameters for positional encoding."),
            Flag("num_layers", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The number of stacking layers of the decoder."),
            Flag("hidden_size", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The number of hidden units of the decoder."),
            Flag("num_attention_heads", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The number of heads of decoder self-attention."),
            Flag("filter_size", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The number of the filter size of decoder ffn."),
            Flag("ffn_activation", dtype=Flag.TYPE.STRING, default="gelu",
                 help="The activation function of decoder ffn layer."),
            Flag("attention_dropout_rate", dtype=Flag.TYPE.FLOAT, default=0.,
                 help="The dropout rate of decoder self-attention layer."),
            Flag("attention_type", dtype=Flag.TYPE.STRING, default="dot_product",
                 help="The type of the attention function of decoder self-attention layer."),
            Flag("ffn_dropout_rate", dtype=Flag.TYPE.FLOAT, default=0.,
                 help="The dropout rate of decoder ffn layer."),
            Flag("layer_postprocess_dropout_rate", dtype=Flag.TYPE.FLOAT, default=0.,
                 help="The dropout rate for each layer's post process in decoder."),
            Flag("layer_postprocess_epsilon", dtype=Flag.TYPE.FLOAT, default=1e-5,
                 help="The epsilon for layer normalization in decoder."),
        ]

    @classmethod
    def new(cls, args: dict, vocab_meta, name=None):
        """ Builds a sequence to sequence model.

        Args:
            args: A dict containing all model parameters.
            vocab_meta: A dict containing source-side vocabulary meta data, e.g. eos_id, vocab_size.
            name: The name of the model.

        Returns:
            A GPT2 model.
        """
        embedding = WordEmbeddingSharedWeights(
            embedding_dim=args["hidden_size"], vocab_size=vocab_meta["vocab_size"],
            share_softmax_weights=True, use_bias=args["softmax_bias"], name="embeddings")
        timing = args["timing"]
        if timing:
            if isinstance(timing, str):
                timing = {"timing": timing}
            elif not isinstance(timing, dict):
                raise ValueError("Unknown type of timing params: {}".format(str(timing)))
            embedding = PositionEmbeddingWrapper(
                embedding_layer=embedding, name="posenc_wrapper", **timing)

        decoder = TransformerDecoder(
            num_layers=args["num_layers"],
            hidden_size=args["hidden_size"],
            num_attention_heads=args["num_attention_heads"],
            filter_size=args["filter_size"],
            ffn_activation=args["ffn_activation"],
            attention_dropout_rate=args["attention_dropout_rate"],
            attention_type=args["attention_type"],
            ffn_dropout_rate=args["ffn_dropout_rate"],
            layer_postprocess_dropout_rate=args["layer_postprocess_dropout_rate"],
            layer_postprocess_epsilon=args["layer_postprocess_epsilon"],
            no_cross_attn_layer_list=[i for i in range(args["num_layers"])],
            name="decoder")
        model = cls(args, vocab_meta, embedding, decoder, name=name)
        _ = model({"trg": tf.convert_to_tensor([[0, 1, 2, vocab_meta["pad_id"]]], tf.int64),
                   "trg_input": tf.convert_to_tensor([[vocab_meta["bos_id"], 1, 2]], tf.int64),
                   "trg_length": tf.convert_to_tensor([4], tf.int64)})
        return model

    def output_logits_layer(self, features):
        """ Projects the decoder output to logits. """
        if self._output_linear_layer is None:
            return self._embedding(features, mode="linear")
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
        inputs = self._embedding(symbols, time=time)
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
                - tokens: int tensor with shape [batch_size, src_input_length].
            is_training: A bool, whether in training mode or not.
            is_inference: A bool, whether in generation mode or not.
            decode_padded_length: The maximum decoding length when inference, for creating
                static-shape cache.

        Returns:  A tuple of (decoding_internal_states, decoder_input, symbol_to_logit_fn)
        """
        input_tokens = inputs["trg_input"]
        # [batch, length, hidden size]
        decoder_internal_cache = self._decoder.create_decoding_internal_cache(
            input_tokens, None, is_inference=is_inference,
            decode_padded_length=decode_padded_length)

        def symbols_to_logits_fn(symbols, cache, time=None):
            """ Generate logits for next potential IDs

            Args:
                symbols: Current decoded sequence.
                cache: A dictionary of values storing the previous decoder attention values.
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
            "eos_id": self._vocab_meta["eos_id"],
            "unk_id": self._vocab_meta.get("unk_id", None)
        }
        return symbols_to_logits_fn, generation_initializer

    def call(self, inputs, is_training=True):
        """ Forward pass of the language model.

        Args:
            inputs: A dict of model inputs.
                - tokens: int tensor with shape [batch_size, length].
            is_training: A bool, whether in training mode or not.

        Returns:
            A logits Tensor.
        """
        symbols_to_logits_fn, generation_initializer = self.get_symbols_to_logits_fn(
            inputs, is_training=is_training, is_inference=False)
        return symbols_to_logits_fn(generation_initializer["decoder_input"],
                                    generation_initializer["decoder_internal_cache"])


def _gpt2_hparams(num_layers,
                  hidden_size,
                  filter_size,
                  dropout_rate,
                  epsilon,
                  num_heads,
                  max_positions):
    return {
        "model.class": GPT2.__name__,
        "model.params": {
            "share_embedding_and_softmax_weights": True,
            "timing": {
                "timing": "emb",
                "max_positions": max_positions,
            },
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "num_attention_heads": num_heads,
            "filter_size": filter_size,
            "ffn_activation": "gelu",
            "attention_dropout_rate": dropout_rate,
            "attention_type": "dot_product",
            "ffn_dropout_rate": dropout_rate,
            "layer_postprocess_dropout_rate": dropout_rate,
            "layer_postprocess_epsilon": epsilon
        }
    }


@register_hparams_set("gpt2_117m")
def gpt2_117m():
    return _gpt2_hparams(
        num_layers=12,
        hidden_size=768,
        filter_size=3072,
        dropout_rate=0.1,
        epsilon=1e-5,
        num_heads=12,
        max_positions=1024)


@register_hparams_set("gpt2_345m")
def gpt2_345m():
    return _gpt2_hparams(
        num_layers=24,
        hidden_size=1024,
        filter_size=4096,
        dropout_rate=0.1,
        epsilon=1e-5,
        num_heads=16,
        max_positions=1024)


@register_hparams_set("gpt2_toy")
def gpt2_toy():
    return _gpt2_hparams(
        num_layers=4,
        hidden_size=8,
        filter_size=16,
        dropout_rate=0.1,
        epsilon=1e-5,
        num_heads=2,
        max_positions=32)
