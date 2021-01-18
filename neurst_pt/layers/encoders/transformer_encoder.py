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
""" Implements transformer encoders as described in https://arxiv.org/abs/1706.03762. """
import torch.nn.functional as F

from neurst_pt.layers import build_transformer_component, layer_utils
from neurst_pt.layers.attentions.multi_head_attention import MultiHeadSelfAttention
from neurst_pt.layers.common_layers import LayerNorm, TransformerFFN
from neurst_pt.layers.encoders import register_encoder
from neurst_pt.layers.encoders.encoder import Encoder


@register_encoder
class TransformerEncoder(Encoder):
    """ Defines transformer encoders as described
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
                 post_normalize=False,
                 return_all_layers=False):
        """ Initializes the transformer encoders.

        Args:
            num_layers: The number of stacked layers.
            hidden_size: The number of hidden units.
            num_attention_heads: The number of self attention heads.
            filter_size: The filter size of ffn layer.
            ffn_activation: The activation function of ffn layer.
            ffn_dropout_rate: The dropout rate for ffn layer.
            attention_dropout_rate: The dropout rate for ffn layer.
            attention_type: The self attention type.
            layer_postprocess_dropout_rate: The dropout rate for each layer post process.
            layer_postprocess_epsilon: The epsilon for layer norm.
            post_normalize: Whether to apply layernorm after each block.
            return_all_layers: Whether to return all encoding layers.
        """
        super(TransformerEncoder, self).__init__(
            num_layers=num_layers, hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            filter_size=filter_size, ffn_activation=ffn_activation,
            ffn_dropout_rate=ffn_dropout_rate,
            attention_type=attention_type,
            attention_dropout_rate=attention_dropout_rate,
            layer_postprocess_dropout_rate=layer_postprocess_dropout_rate,
            layer_postprocess_epsilon=layer_postprocess_epsilon,
            post_normalize=post_normalize)
        assert post_normalize or (not post_normalize and not return_all_layers), (
            "`return_all_layers` is only available when `post_normalize`=True.")
        self._return_all_layers = return_all_layers
        self._stacking_layers = []
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
        # [batch_size, max_length], FLOAT_MIN for padding, 0.0 for non-padding
        all_layers = []
        self_attention_bias = layer_utils.input_padding_to_bias(inputs_padding)
        x = F.dropout(inputs, p=self._params["layer_postprocess_dropout_rate"],
                      training=is_training)
        for idx, layer in enumerate(self._stacking_layers):
            self_attention_layer = layer[0]
            ffn_layer = layer[1]
            # self attention layer
            x = self_attention_layer(
                x,  # x as query
                bias=self_attention_bias,
                is_training=is_training)
            # ffn
            x = ffn_layer(x, is_training=is_training)
            all_layers.append(x)
        if self._params["post_normalize"]:
            if self._return_all_layers:
                return all_layers
            return x
        return self._output_norm_layer(x)
