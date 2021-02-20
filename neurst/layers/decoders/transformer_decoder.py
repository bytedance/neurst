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
""" Implements transformer decoder in TF2 as described in https://arxiv.org/abs/1706.03762. """
import tensorflow as tf

from neurst.layers import build_transformer_component, layer_utils
from neurst.layers.attentions.multi_head_attention import MultiHeadAttention, MultiHeadSelfAttention
from neurst.layers.common_layers import TransformerFFN
from neurst.layers.decoders import Decoder, register_decoder
from neurst.utils import compat


@register_decoder
class TransformerDecoder(Decoder):
    """ Defines transformer encoder as described
    in https://arxiv.org/abs/1706.03762. """

    def __init__(self,
                 num_layers,
                 hidden_size,
                 num_attention_heads,
                 filter_size,
                 ffn_activation="relu",
                 attention_dropout_rate=0.,
                 attention_type="dot_product",
                 ffn_dropout_rate=0.,
                 layer_postprocess_dropout_rate=0.,
                 layer_postprocess_epsilon=1e-6,
                 with_encoder_decoder_attention=True,
                 post_normalize=False,
                 name=None):
        """ Initializes the parameters of the transformer decoder.

        Args:
            num_layers: The number of stacked layers.
            hidden_size: The number of hidden units.
            num_attention_heads: The number of self attention heads, for both self-attention
                and encoder-decoder attention.
            filter_size: The filter size of ffn layer.
            ffn_activation: The activation function of ffn layer.
            ffn_dropout_rate: The dropout rate for ffn layer.
            attention_dropout_rate: The dropout rate for ffn layer, for both self-attention
                and encoder-decoder attention.
            attention_type: The self attention type, for both self-attention and
                encoder-decoder attention.
            layer_postprocess_dropout_rate: The dropout rate for each
                layer post process.
            layer_postprocess_epsilon: The epsilon for layer norm.
            post_normalize: Whether to apply layernorm after each block.
            name: The name of this decoder.
        """
        super(TransformerDecoder, self).__init__(
            num_layers=num_layers, hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            filter_size=filter_size, ffn_activation=ffn_activation,
            ffn_dropout_rate=ffn_dropout_rate,
            attention_type=attention_type,
            attention_dropout_rate=attention_dropout_rate,
            layer_postprocess_dropout_rate=layer_postprocess_dropout_rate,
            layer_postprocess_epsilon=layer_postprocess_epsilon,
            with_encoder_decoder_attention=with_encoder_decoder_attention,
            post_normalize=post_normalize,
            name=name or self.__class__.__name__)
        self._stacking_layers = []
        self._with_encoder_decoder_attention = with_encoder_decoder_attention

    def build(self, input_shape):
        """ Builds the transformer decoder layer. """
        params = self.get_config()
        for _ in range(params["num_layers"]):
            self._stacking_layers.append([
                build_transformer_component({
                    "base_layer.class": MultiHeadSelfAttention.__name__,
                    "base_layer.params": dict(
                        num_heads=params["num_attention_heads"],
                        num_units=params["hidden_size"],
                        attention_dropout_rate=params["attention_dropout_rate"],
                        attention_type=params["attention_type"],
                        name="self_attention"
                    )},
                    dropout_rate=params["layer_postprocess_dropout_rate"],
                    epsilon=params["layer_postprocess_epsilon"],
                    pre_norm=(not params["post_normalize"])),
                (build_transformer_component({
                    "base_layer.class": MultiHeadAttention.__name__,
                    "base_layer.params": dict(
                        num_heads=params["num_attention_heads"],
                        num_units=params["hidden_size"],
                        attention_dropout_rate=params["attention_dropout_rate"],
                        attention_type=params["attention_type"],
                        name="encdec_attention")},
                    dropout_rate=params["layer_postprocess_dropout_rate"],
                    epsilon=params["layer_postprocess_epsilon"],
                    pre_norm=(not params["post_normalize"]))
                 if self._with_encoder_decoder_attention else None),
                build_transformer_component({
                    "base_layer.class": TransformerFFN.__name__,
                    "base_layer.params": dict(
                        filter_size=params["filter_size"],
                        output_size=params["hidden_size"],
                        dropout_rate=params["ffn_dropout_rate"],
                        activation=params["ffn_activation"],
                        name="ffn")},
                    dropout_rate=params["layer_postprocess_dropout_rate"],
                    epsilon=params["layer_postprocess_epsilon"],
                    pre_norm=(not params["post_normalize"]))])
        if not params["post_normalize"]:
            self._output_norm_layer = tf.keras.layers.LayerNormalization(
                epsilon=params["layer_postprocess_epsilon"],
                dtype="float32", name="output_ln")
            self.add_activation_quantizer(name="output_ln", activation="act")
        super(TransformerDecoder, self).build(input_shape)

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
        # [batch_size, max_length], FLOAT_MIN for padding, 0.0 for non-padding
        if is_inference:
            params = self.get_config()
            decoding_states = {}
            batch_size = tf.shape(encoder_outputs)[0]
            num_heads = params["num_attention_heads"]
            num_units_per_head = params["hidden_size"] // num_heads
            # initialize decoder self attention keys/values
            for lid in range(params["num_layers"]):
                # Ensure shape invariance for tf.while_loop.
                decoding_states["layer_{}".format(lid)] = {
                    "self_attention": {
                        "keys": tf.zeros([batch_size, decode_padded_length or 0, num_heads, num_units_per_head],
                                         dtype=compat.CUSTOM_GLOBAL_FLOATX),
                        "values": tf.zeros([batch_size, decode_padded_length or 0, num_heads, num_units_per_head],
                                           dtype=compat.CUSTOM_GLOBAL_FLOATX)},
                }
        else:
            decoding_states = None
        cache = dict(decoding_states=decoding_states)
        if self._with_encoder_decoder_attention:
            cache["memory"] = encoder_outputs
            cache["memory_bias"] = layer_utils.input_padding_to_bias(
                encoder_inputs_padding)
        return cache

    def call(self, decoder_inputs, cache, is_training=True, decode_loop_step=None):
        """ Encodes the inputs.

        Args:
            decoder_inputs: The embedded decoder input, a float tensor with shape
                [batch_size, max_target_length, embedding_dim] or
                [batch_size, embedding_dim] for one decoding step.
            cache: A dictionary, generated from self.create_decoding_internal_cache.
            is_training: A bool, whether in training mode or not.
            decode_loop_step: An integer, step number of the decoding loop. Used only
                for autoregressive inference with static-shape cache.

        Returns:
            The decoder output with shape [batch_size, max_length, hidden_size]
            when `decoder_inputs` is a 3-d tensor or with shape
            [batch_size, hidden_size] when `decoder_inputs` is a 2-d tensor.
        """
        ori_ndims = decoder_inputs.get_shape().ndims
        if ori_ndims == 2:
            decoder_inputs = tf.expand_dims(decoder_inputs, axis=1)

        # decoder self attention has shape [1, 1, max_target_len, max_target_len]
        decoder_self_attention_bias = layer_utils.lower_triangle_attention_bias(
            tf.shape(decoder_inputs)[1])
        x = decoder_inputs
        if is_training:
            x = tf.nn.dropout(
                decoder_inputs, rate=self.get_config()["layer_postprocess_dropout_rate"])
        for idx, layer in enumerate(self._stacking_layers):
            selfatt_layer = layer[0]
            encdecatt_layer = layer[1]
            ffn_layer = layer[2]
            layer_name = "layer_{}".format(idx)
            layer_cache = None if cache["decoding_states"] is None else cache["decoding_states"][layer_name]
            selfatt_cache = None if layer_cache is None else layer_cache["self_attention"]
            with tf.name_scope(layer_name):
                # self attention layer
                x = selfatt_layer(
                    x,  # x as query
                    bias=decoder_self_attention_bias,
                    cache=selfatt_cache,
                    is_training=is_training,
                    decode_loop_step=decode_loop_step)
                # enc-dec attention layer
                if encdecatt_layer is not None:
                    x = encdecatt_layer(
                        x,  # x as query
                        memory=cache["memory"],  # None indicates self-attention
                        memory_bias=cache["memory_bias"],
                        is_training=is_training)
                # ffn
                x = ffn_layer(x, is_training=is_training)
        outputs = x
        if not self.get_config()["post_normalize"]:
            outputs = self.quant(self._output_norm_layer(x), name="output_ln")
        if ori_ndims == 2:
            outputs = tf.squeeze(outputs, axis=1)
        return outputs
