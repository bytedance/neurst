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

from neurst.layers import layer_utils
from neurst.layers.decoders import Decoder, register_decoder
from neurst.layers.layer_utils import tile_tensor
from neurst.layers.transformer_layers import TransformerDecoderLayer


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
                 no_cross_attn_layer_list=None,
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
            no_cross_attn_layer_list: A list of layer indexes (start from 0),
                which do not include cross attention.
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
            no_cross_attn_layer_list=(no_cross_attn_layer_list or []),
            post_normalize=post_normalize,
            name=name or self.__class__.__name__)
        self._stacking_layers = []

    def build(self, input_shape):
        """ Builds the transformer decoder layer. """
        params = self.get_config()
        for idx in range(params["num_layers"]):
            self._stacking_layers.append(
                TransformerDecoderLayer(
                    hidden_size=params["hidden_size"],
                    num_attention_heads=params["num_attention_heads"],
                    filter_size=params["filter_size"],
                    ffn_activation=params["ffn_activation"],
                    attention_dropout_rate=params["attention_dropout_rate"],
                    attention_type=params["attention_type"],
                    ffn_dropout_rate=params["ffn_dropout_rate"],
                    layer_postprocess_dropout_rate=params["layer_postprocess_dropout_rate"],
                    layer_postprocess_epsilon=params["layer_postprocess_epsilon"],
                    post_normalize=params["post_normalize"],
                    with_cross_attention=(idx not in params["no_cross_attn_layer_list"]),
                    name=f"layer_{idx}"
                ))

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
            decoding_states = {}
            batch_size = tf.shape(encoder_outputs)[0]
            # initialize decoder self attention keys/values
            for lid, layer in enumerate(self._stacking_layers):
                # Ensure shape invariance for tf.while_loop.
                decoding_states[f"layer_{lid}"] = layer.create_decoding_internal_cache(
                    decode_padded_length)
            decoding_states = tf.nest.map_structure(
                lambda ts: tile_tensor(ts, batch_size, axis=0), decoding_states)
            for lid, layer in enumerate(self._stacking_layers):
                decoding_states[f"layer_{lid}"].update(layer.memorize_memory(encoder_outputs))
        else:
            decoding_states = None
        cache = dict(decoding_states=decoding_states)
        if encoder_inputs_padding is not None:
            cache["memory"] = encoder_outputs
            cache["memory_bias"] = layer_utils.input_padding_to_bias(encoder_inputs_padding)
        return cache

    def update_incremental_cache(self,
                                 cache,
                                 encoder_outputs,
                                 encoder_inputs_padding):
        if cache is None or len(cache) == 0:
            cache = self.create_decoding_internal_cache(
                encoder_outputs=encoder_outputs,
                encoder_inputs_padding=encoder_inputs_padding,
                is_inference=True,
                decode_padded_length=None)
        elif encoder_inputs_padding is not None:
            cache["memory"] = tf.concat([cache["memory"], encoder_outputs], axis=1)
            cache["memory_bias"] = tf.concat([cache["memory_bias"], encoder_inputs_padding], axis=1)
            for lid, layer in enumerate(self._stacking_layers):
                for k, v in layer.memorize_memory(encoder_outputs).items():
                    cache["decoding_states"][f"layer_{lid}"][k] = tf.nest.pack_sequence_as(
                        structure=v, flat_sequence=tf.nest.map_structure(
                            lambda x, y: tf.concat([x, y], axis=1),
                            tf.nest.flatten(cache["decoding_states"][f"layer_{lid}"][k]),
                            tf.nest.flatten(v)))
        return cache

    def call(self, decoder_inputs, cache, decode_lagging=None,
             is_training=True, decode_loop_step=None):
        """ Encodes the inputs.

        Args:
            decoder_inputs: The embedded decoder input, a float tensor with shape
                [batch_size, max_target_length, embedding_dim] or
                [batch_size, embedding_dim] for one decoding step.
            cache: A dictionary, generated from self.create_decoding_internal_cache.
            decode_lagging: The waitk lagging for streaming input when training.
                During inference, it is the lagging for current step.
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
        memory_bias = cache.get("memory_bias", None)  # [batch, memory_length]
        if memory_bias is not None and decode_lagging is not None:
            if ori_ndims == 3:
                memory_bias = tf.minimum(tf.expand_dims(memory_bias, axis=1),
                                         tf.expand_dims(
                                             layer_utils.waitk_attention_bias(
                                                 memory_length=tf.shape(memory_bias)[1],
                                                 query_length=tf.shape(decoder_inputs)[1],
                                                 waitk_lagging=decode_lagging), axis=0))
            else:  # ori_ndims == 2
                memory_bias = tf.minimum(memory_bias,
                                         tf.expand_dims(layer_utils.waitk_attention_bias(
                                             memory_length=tf.shape(memory_bias)[1],
                                             waitk_lagging=decode_lagging), axis=0))
        # decoder self attention has shape [1, 1, max_target_len, max_target_len]
        decoder_self_attention_bias = layer_utils.lower_triangle_attention_bias(
            tf.shape(decoder_inputs)[1])
        x = decoder_inputs
        if is_training:
            x = tf.nn.dropout(
                decoder_inputs, rate=self.get_config()["layer_postprocess_dropout_rate"])
        for idx, layer in enumerate(self._stacking_layers):
            layer_cache = (None if cache["decoding_states"] is None
                           else cache["decoding_states"][f"layer_{idx}"])
            x = self._stacking_layers[idx](x, decoder_self_attention_bias, layer_cache,
                                           memory=cache.get("memory", None),
                                           memory_bias=memory_bias,
                                           is_training=is_training,
                                           decode_loop_step=decode_loop_step)
        outputs = x
        if not self.get_config()["post_normalize"]:
            outputs = self.quant(self._output_norm_layer(x), name="output_ln")
        if ori_ndims == 2:
            outputs = tf.squeeze(outputs, axis=1)
        return outputs
