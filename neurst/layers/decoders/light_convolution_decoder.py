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
""" Implements lightweight conv decoder as described in https://arxiv.org/pdf/1901.10430.pdf. """
import tensorflow as tf

from neurst.layers import build_transformer_component, layer_utils
from neurst.layers.attentions.light_convolution_layer import LightConvolutionLayer
from neurst.layers.attentions.multi_head_attention import MultiHeadAttention
from neurst.layers.common_layers import TransformerFFN
from neurst.layers.decoders import Decoder, register_decoder
from neurst.utils import compat


@register_decoder
class LightConvolutionDecoder(Decoder):
    """ Implements lightweight conv decoder as described in https://arxiv.org/pdf/1901.10430.pdf. """

    def __init__(self,
                 num_layers,
                 conv_kernel_size_list,
                 num_conv_heads,
                 conv_hidden_size,
                 num_attention_heads,
                 filter_size,
                 glu_after_proj=True,
                 conv_type="lightweight",
                 ffn_activation="relu",
                 conv_weight_dropout_rate=0.,
                 attention_dropout_rate=0.,
                 attention_type="dot_product",
                 ffn_dropout_rate=0.,
                 layer_postprocess_dropout_rate=0.,
                 layer_postprocess_epsilon=1e-6,
                 name=None):
        """ Initializes the parameters of the transformer decoder.

        Args:
            num_layers: The number of stacked layers.
            conv_kernel_size_list: An int list of encoder kernel sizes. The length of the list must
                be equal to `num_layers`.
            num_conv_heads: An integer, the number of heads for conv shared weights.
            num_attention_heads: The number of self attention heads, for encoder-decoder attention.
            filter_size: The filter size of ffn layer.
            conv_type: The type of conv layer, one of lightweight or dynamic.
            conv_hidden_size: The hidden size of conv layer.
            glu_after_proj: Whether to apply glu activation after input projection.
            conv_weight_dropout_rate: The dropout rate of the conv weights.
            ffn_activation: The activation function of ffn layer.
            ffn_dropout_rate: The dropout rate for ffn layer.
            attention_dropout_rate: The dropout rate for ffn layer, for encoder-decoder attention.
            attention_type: The self attention type, for encoder-decoder attention.
            layer_postprocess_dropout_rate: The dropout rate for each
                layer post process.
            layer_postprocess_epsilon: The epsilon for layer norm.
            name: The name of this decoder.
        """
        super(LightConvolutionDecoder, self).__init__(
            num_layers=num_layers, conv_kernel_size_list=conv_kernel_size_list,
            num_conv_heads=num_conv_heads, conv_hidden_size=conv_hidden_size,
            num_attention_heads=num_attention_heads, conv_type=conv_type,
            filter_size=filter_size, ffn_activation=ffn_activation,
            ffn_dropout_rate=ffn_dropout_rate,
            glu_after_proj=glu_after_proj,
            conv_weight_dropout_rate=conv_weight_dropout_rate,
            attention_type=attention_type,
            attention_dropout_rate=attention_dropout_rate,
            layer_postprocess_dropout_rate=layer_postprocess_dropout_rate,
            layer_postprocess_epsilon=layer_postprocess_epsilon,
            name=name or self.__class__.__name__)
        self._stacking_layers = []

    def build(self, input_shape):
        """ Builds the transformer decoder layer. """
        params = self.get_config()
        for lid in range(params["num_layers"]):
            self._stacking_layers.append([
                build_transformer_component({
                    "base_layer.class": LightConvolutionLayer.__name__,
                    "base_layer.params": dict(
                        kernel_size=params["conv_kernel_size_list"][lid],
                        num_heads=params["num_conv_heads"],
                        conv_type=params["conv_type"],
                        conv_dim=params["conv_hidden_size"],
                        use_glu=params["glu_after_proj"],
                        weight_dropout_rate=params["conv_weight_dropout_rate"],
                        name="light_conv"
                    )},
                    dropout_rate=params["layer_postprocess_dropout_rate"],
                    epsilon=params["layer_postprocess_epsilon"]),
                build_transformer_component({
                    "base_layer.class": MultiHeadAttention.__name__,
                    "base_layer.params": dict(
                        num_heads=params["num_attention_heads"],
                        num_units=input_shape[-1],
                        attention_dropout_rate=params["attention_dropout_rate"],
                        attention_type=params["attention_type"],
                        name="encdec_attention")},
                    dropout_rate=params["layer_postprocess_dropout_rate"],
                    epsilon=params["layer_postprocess_epsilon"]),
                build_transformer_component({
                    "base_layer.class": TransformerFFN.__name__,
                    "base_layer.params": dict(
                        filter_size=params["filter_size"],
                        output_size=input_shape[-1],
                        dropout_rate=params["ffn_dropout_rate"],
                        activation=params["ffn_activation"],
                        name="ffn")},
                    dropout_rate=params["layer_postprocess_dropout_rate"],
                    epsilon=params["layer_postprocess_epsilon"])])
        self._output_norm_layer = tf.keras.layers.LayerNormalization(
            epsilon=params["layer_postprocess_epsilon"],
            dtype="float32", name="output_ln")
        super(LightConvolutionDecoder, self).build(input_shape)

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
        enc_dec_attention_bias = layer_utils.input_padding_to_bias(
            encoder_inputs_padding)
        if is_inference:
            params = self.get_config()
            decoding_states = {}
            batch_size = tf.shape(encoder_outputs)[0]
            # initialize decoder conv hidden states
            for lid in range(params["num_layers"]):
                # Ensure shape invariance for tf.while_loop.
                if decode_padded_length is None:
                    init_len = params["conv_kernel_size_list"][lid] - 1
                else:
                    init_len = params["conv_kernel_size_list"][lid] - 1 + decode_padded_length
                decoding_states["layer_{}".format(lid)] = {
                    "light_conv": {
                        "conv": tf.zeros([batch_size, init_len,
                                          params["conv_hidden_size"]], dtype=compat.CUSTOM_GLOBAL_FLOATX)},
                }
        else:
            decoding_states = None
        cache = dict(decoding_states=decoding_states,
                     memory=encoder_outputs,
                     memory_bias=enc_dec_attention_bias)
        return cache

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
        ori_ndims = decoder_inputs.get_shape().ndims
        if ori_ndims == 2:
            decoder_inputs = tf.expand_dims(decoder_inputs, axis=1)
        memory_bias = cache["memory_bias"]
        if decode_lagging is not None:
            if ori_ndims == 3:
                memory_bias = tf.minimum(tf.expand_dims(memory_bias, axis=1),
                                         tf.expand_dims(
                                             layer_utils.waitk_attention_bias(
                                                 memory_length=tf.shape(cache["memory"])[1],
                                                 query_length=tf.shape(decoder_inputs)[1],
                                                 waitk_lagging=decode_lagging), axis=0))
            else:  # ori_ndims == 2
                memory_bias = tf.minimum(memory_bias,
                                         tf.expand_dims(layer_utils.waitk_attention_bias(
                                             memory_length=tf.shape(cache["memory"])[1],
                                             waitk_lagging=decode_lagging), axis=0))
        x = decoder_inputs
        if is_training:
            x = tf.nn.dropout(
                decoder_inputs, rate=self.get_config()["layer_postprocess_dropout_rate"])
        for idx, layer in enumerate(self._stacking_layers):
            conv_layer = layer[0]
            encdecatt_layer = layer[1]
            ffn_layer = layer[2]
            layer_name = "layer_{}".format(idx)
            layer_cache = None if cache["decoding_states"] is None else cache["decoding_states"][layer_name]
            conv_cache = None if layer_cache is None else layer_cache["light_conv"]
            with tf.name_scope(layer_name):
                # self attention layer
                x = conv_layer(x,
                               left_padding=True,
                               cache=conv_cache,
                               is_training=is_training,
                               decode_loop_step=decode_loop_step)
                # enc-dec attention layer
                x = encdecatt_layer(
                    x,  # x as query
                    memory=cache["memory"],  # None indicates self-attention
                    memory_bias=memory_bias,
                    is_training=is_training)
                # ffn
                x = ffn_layer(x, is_training=is_training)
        outputs = self._output_norm_layer(x)
        if ori_ndims == 2:
            outputs = tf.squeeze(outputs, axis=1)
        return outputs
