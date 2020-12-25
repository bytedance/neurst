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
""" Implements light convolution encoder as described in https://arxiv.org/pdf/1901.10430.pdf. """
import tensorflow as tf

from neurst.layers import build_transformer_component
from neurst.layers.attentions.light_convolution_layer import LightConvolutionLayer
from neurst.layers.common_layers import TransformerFFN
from neurst.layers.encoders import register_encoder
from neurst.layers.encoders.encoder import Encoder


@register_encoder
class LightConvolutionEncoder(Encoder):
    """ Defines light convolution encoder as described https://arxiv.org/pdf/1901.10430.pdf. """

    def __init__(self,
                 num_layers,
                 conv_kernel_size_list,
                 num_conv_heads,
                 conv_hidden_size,
                 filter_size,
                 conv_type="lightweight",
                 glu_after_proj=True,
                 conv_weight_dropout_rate=0.,
                 ffn_activation="relu",
                 ffn_dropout_rate=0.,
                 layer_postprocess_dropout_rate=0.,
                 layer_postprocess_epsilon=1e-6,
                 name=None):
        """ Initializes the transformer encoders.

        Args:
            num_layers: The number of stacked layers.
            conv_kernel_size_list: An int list of encoder kernel sizes. The length of the list must
                be equal to `num_layers`.
            num_conv_heads: An integer, the number of heads for conv shared weights.
            conv_hidden_size: The hidden size of conv layer.
            filter_size: The filter size of ffn layer.
            conv_type: The type of conv layer, one of lightweight or dynamic.
            ffn_activation: The activation function of ffn layer.
            ffn_dropout_rate: The dropout rate for ffn layer.
            glu_after_proj: Whether to apply glu activation after input projection.
            conv_weight_dropout_rate: The dropout rate of the conv weights.
            layer_postprocess_dropout_rate: The dropout rate for each layer post process.
            layer_postprocess_epsilon: The epsilon for layer norm.
            name: The name of this encoder.
        """
        super(LightConvolutionEncoder, self).__init__(
            num_layers=num_layers, conv_kernel_size_list=conv_kernel_size_list,
            num_conv_heads=num_conv_heads, conv_hidden_size=conv_hidden_size,
            filter_size=filter_size, ffn_activation=ffn_activation,
            ffn_dropout_rate=ffn_dropout_rate, conv_type=conv_type,
            glu_after_proj=glu_after_proj,
            conv_weight_dropout_rate=conv_weight_dropout_rate,
            layer_postprocess_dropout_rate=layer_postprocess_dropout_rate,
            layer_postprocess_epsilon=layer_postprocess_epsilon,
            name=name or self.__class__.__name__)
        self._stacking_layers = []

    def build(self, input_shape):
        """ Builds the transformer encoder layer. """
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
                    "base_layer.class": TransformerFFN.__name__,
                    "base_layer.params": dict(
                        filter_size=params["filter_size"],
                        output_size=input_shape[-1],
                        dropout_rate=params["ffn_dropout_rate"],
                        activation=params["ffn_activation"],
                        name="ffn")},
                    dropout_rate=params["layer_postprocess_dropout_rate"],
                    epsilon=params["layer_postprocess_epsilon"])
            ])
        self._output_norm_layer = tf.keras.layers.LayerNormalization(
            epsilon=params["layer_postprocess_epsilon"],
            dtype="float32", name="output_ln")
        super(LightConvolutionEncoder, self).build(input_shape)

    def call(self, inputs, inputs_padding, is_training=True):
        """ Encodes the inputs.

        Args:
            inputs: The embedded input, a float tensor with shape
                [batch_size, max_length, embedding_dim].
            inputs_padding: A float tensor with shape [batch_size, max_length],
                indicating the padding positions, where 1.0 for padding andtf
                0.0 for non-padding.
            is_training: A bool, whether in training mode or not.

        Returns:
            The encoded output with shape [batch_size, max_length, hidden_size]
        """
        # [batch_size, max_length], FLOAT_MIN for padding, 0.0 for non-padding
        x = inputs
        if is_training:
            x = tf.nn.dropout(x, rate=self.get_config()[
                "layer_postprocess_dropout_rate"])
        for idx, layer in enumerate(self._stacking_layers):
            conv_layer = layer[0]
            ffn_layer = layer[1]
            with tf.name_scope("layer_{}".format(idx)):
                # self attention layer
                x = conv_layer(x, inputs_padding=inputs_padding, is_training=is_training)
                # ffn
                x = ffn_layer(x, is_training=is_training)

        return self._output_norm_layer(x)
