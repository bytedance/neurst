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
import torch
import torch.nn.functional as F

from neurst_pt.layers import build_transformer_component, layer_utils
from neurst_pt.layers.attentions.multi_head_attention import MultiHeadAttention, MultiHeadSelfAttention
from neurst_pt.layers.common_layers import LayerNorm, TransformerFFN
from neurst_pt.layers.decoders import Decoder, register_decoder


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
                 post_normalize=False):
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
            post_normalize=post_normalize)
        self._stacking_layers = []
        self._with_encoder_decoder_attention = with_encoder_decoder_attention
        for _ in range(num_layers):
            self._stacking_layers.append([
                build_transformer_component({
                    "base_layer.class": MultiHeadSelfAttention.__name__,
                    "base_layer.params": dict(
                        input_depth=hidden_size,
                        num_heads=num_attention_heads,
                        num_units=hidden_size,
                        attention_dropout_rate=attention_dropout_rate,
                        attention_type=attention_type,
                    )},
                    norm_shape=hidden_size,
                    dropout_rate=layer_postprocess_dropout_rate,
                    epsilon=layer_postprocess_epsilon,
                    pre_norm=(not post_normalize)),
                (build_transformer_component({
                    "base_layer.class": MultiHeadAttention.__name__,
                    "base_layer.params": dict(
                        input_depth=hidden_size,
                        num_heads=num_attention_heads,
                        num_units=hidden_size,
                        attention_dropout_rate=attention_dropout_rate,
                        attention_type=attention_type)},
                    norm_shape=hidden_size,
                    dropout_rate=layer_postprocess_dropout_rate,
                    epsilon=layer_postprocess_epsilon,
                    pre_norm=(not post_normalize))
                 if self._with_encoder_decoder_attention else None),
                build_transformer_component({
                    "base_layer.class": TransformerFFN.__name__,
                    "base_layer.params": dict(
                        input_size=hidden_size,
                        filter_size=filter_size,
                        output_size=hidden_size,
                        dropout_rate=ffn_dropout_rate,
                        activation=ffn_activation,
                    )},
                    norm_shape=hidden_size,
                    dropout_rate=layer_postprocess_dropout_rate,
                    epsilon=layer_postprocess_epsilon,
                    pre_norm=(not post_normalize))
            ])
        if not post_normalize:
            self._output_norm_layer = LayerNorm(hidden_size, layer_postprocess_epsilon)

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
            batch_size = encoder_outputs.size()[0]
            num_heads = self._params["num_attention_heads"]
            num_units_per_head = self._params["hidden_size"] // num_heads
            # initialize decoder self attention keys/values
            for lid in range(self._params["num_layers"]):
                # Ensure shape invariance for tf.while_loop.
                decoding_states["layer_{}".format(lid)] = {
                    "self_attention": {
                        "keys": torch.zeros(batch_size, decode_padded_length or 0, num_heads, num_units_per_head,
                                            dtype=torch.float),
                        "values": torch.zeros(batch_size, decode_padded_length or 0, num_heads, num_units_per_head,
                                              dtype=torch.float)},
                }
        else:
            decoding_states = None
        cache = dict(decoding_states=decoding_states)
        if self._with_encoder_decoder_attention:
            cache["memory"] = encoder_outputs
            cache["memory_bias"] = layer_utils.input_padding_to_bias(
                encoder_inputs_padding)
        return cache

    def forward(self, decoder_inputs, cache, is_training=True, decode_loop_step=None):
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
        ori_ndims = decoder_inputs.ndim
        if ori_ndims == 2:
            decoder_inputs = decoder_inputs.unsqueeze(1)

        # decoder self attention has shape [1, 1, max_target_len, max_target_len]
        decoder_self_attention_bias = layer_utils.lower_triangle_attention_bias(
            decoder_inputs.size()[1])
        x = F.dropout(decoder_inputs, training=is_training,
                      p=self._params["layer_postprocess_dropout_rate"])
        for idx, layer in enumerate(self._stacking_layers):
            selfatt_layer = layer[0]
            encdecatt_layer = layer[1]
            ffn_layer = layer[2]
            layer_name = "layer_{}".format(idx)
            layer_cache = None if cache["decoding_states"] is None else cache["decoding_states"][layer_name]
            selfatt_cache = None if layer_cache is None else layer_cache["self_attention"]
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
        if not self._params["post_normalize"]:
            outputs = self._output_norm_layer(x)
        if ori_ndims == 2:
            outputs = outputs.squeeze(1)
        return outputs
