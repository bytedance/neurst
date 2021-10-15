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
import tensorflow as tf

from neurst.layers import layer_utils, build_transformer_component_with_adapter
from neurst.layers.adapters import build_adapter
from neurst.layers.attentions.multi_head_attention import MultiHeadSelfAttention
from neurst.layers.common_layers import TransformerFFN
from neurst.layers.encoders import register_encoder
from neurst.layers.encoders.encoder import Encoder


@register_encoder
class TransformerEncoderCIAT(Encoder):
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
                 attention_monotonic=False,
                 ffn_dropout_rate=0.,
                 layer_postprocess_dropout_rate=0.,
                 layer_postprocess_epsilon=1e-6,
                 post_normalize=False,
                 return_all_layers=False,
                 emb_adapter_dim=None,
                 layer_adapter_dim=None,
                 is_pretrain=True,
                 name=None):
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
            attention_monotonic: Whether to apply a triangle mask.
            layer_postprocess_dropout_rate: The dropout rate for each layer post process.
            layer_postprocess_epsilon: The epsilon for layer norm.
            post_normalize: Whether to apply layernorm after each block.
            return_all_layers: Whether to return all encoding layers.
            name: The name of this encoder.
        """
        super(TransformerEncoderCIAT, self).__init__(
            num_layers=num_layers, hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            filter_size=filter_size, ffn_activation=ffn_activation,
            ffn_dropout_rate=ffn_dropout_rate,
            attention_type=attention_type,
            attention_dropout_rate=attention_dropout_rate,
            attention_monotonic=attention_monotonic,
            layer_postprocess_dropout_rate=layer_postprocess_dropout_rate,
            layer_postprocess_epsilon=layer_postprocess_epsilon,
            post_normalize=post_normalize,
            name=name or self.__class__.__name__)
        self._stacking_layers = []

        if emb_adapter_dim is None:
            emb_adapter_dim = hidden_size // 2
        if layer_adapter_dim is None:
            layer_adapter_dim = hidden_size // 2
        self._emb_adapter_dim = emb_adapter_dim
        self._layer_adapter_dim = layer_adapter_dim
        self.emb_adapter = None
        self._is_pretrain = is_pretrain

        assert post_normalize or (not post_normalize and not return_all_layers), (
            "`return_all_layers` is only available when `post_normalize`=True.")
        self._return_all_layers = return_all_layers

    def build(self, input_shape):
        """ Builds the transformer encoder layer. """
        params = self.get_config()


        if not params["post_normalize"]:
            self._output_norm_layer = tf.keras.layers.LayerNormalization(
                epsilon=params["layer_postprocess_epsilon"],
                dtype="float32", name="output_ln")
            self.add_activation_quantizer(name="output_ln", activation="act")

        emb_adapter_paras = dict(
            {'hidden_size_inner': self._emb_adapter_dim,
             'hidden_size_outter': params["hidden_size"],
             'dropout_rate': params["layer_postprocess_dropout_rate"]}
        )
        # build emb adapter
        self.emb_adapter = build_adapter(
            {"adapter.class": "AdapterEmb",
             "adapter.params": {**emb_adapter_paras, **{"name": "emb_adapter_encoder"}},
             }
        )
        # build layer adapter
        for _ in range(params["num_layers"]):
            name_list = [{"name": "SA_adapter_encoder"}, {"name": "FFW_adapter_encoder"}]
            adapter_params = dict(
                {'hidden_size_inner': self._layer_adapter_dim,
                 'hidden_size_outter': params["hidden_size"],
                 'dropout_rate': params["layer_postprocess_dropout_rate"]}
            )
            adapter_class="AdapterLayer"
            self._stacking_layers.append([
                build_transformer_component_with_adapter({
                    "base_layer.class": MultiHeadSelfAttention.__name__,
                    "base_layer.params": dict(
                        num_heads=params["num_attention_heads"],
                        num_units=params["hidden_size"],
                        attention_dropout_rate=params["attention_dropout_rate"],
                        attention_type=params["attention_type"],
                        name="self_attention"
                    )},
                    dropout_rate=params["layer_postprocess_dropout_rate"],
                    adapter_args={
                        "adapter.class": adapter_class,
                        "adapter.params": {**adapter_params, **name_list[0]},
                    },
                    is_pretrain=self._is_pretrain,
                    USEADAPTER=True,
                ),
                build_transformer_component_with_adapter({
                    "base_layer.class": TransformerFFN.__name__,
                    "base_layer.params": dict(
                        filter_size=params["filter_size"],
                        output_size=params["hidden_size"],
                        dropout_rate=params["ffn_dropout_rate"],
                        activation=params["ffn_activation"],
                        name="ffn")},
                    dropout_rate=params["layer_postprocess_dropout_rate"],
                    adapter_args={
                        "adapter.class": adapter_class,
                        "adapter.params": {**adapter_params, **name_list[1]},
                    },
                    is_pretrain=self._is_pretrain,
                )
            ])

        super(TransformerEncoderCIAT, self).build(input_shape)

    def call(self, inputs, inputs_padding, is_training=True):
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
        if self.get_config()["attention_monotonic"]:
            self_attention_bias = tf.minimum(tf.expand_dims(tf.expand_dims(self_attention_bias, axis=1), axis=1),
                                             layer_utils.lower_triangle_attention_bias(tf.shape(inputs)[1]))
        x = inputs
        if self.emb_adapter is not None and self._is_pretrain is False:
            x = x + self.emb_adapter(x, is_training=is_training)

        if is_training:
            x = tf.nn.dropout(x, rate=self.get_config()[
                "layer_postprocess_dropout_rate"])
        for idx, layer in enumerate(self._stacking_layers):
            self_attention_layer = layer[0]
            ffn_layer = layer[1]
            with tf.name_scope("layer_{}".format(idx)):
                # self attention layer
                x = self_attention_layer(
                    x,  # x as query
                    bias=self_attention_bias,
                    is_training=is_training)
                # ffn
                x = ffn_layer(x, is_training=is_training)
        if self.get_config()["post_normalize"]:
            if self._return_all_layers:
                return all_layers
            return x
        outputs = self.quant(self._output_norm_layer(x), name="output_ln")
        return outputs

    def incremental_encode(self, inputs, cache, time=None):
        """ Encoding function for streaming input.

        Args:
            inputs: The embedded input at time t, a float tensor with shape [batch, embedding_dim]
                or [batch, length, embedding_dim]
            cache: A dict containing cached tensors.
            time: The start time of the inputs

        Returns: The incremented encoder output with shape [batch, t+1, dim],
            and the updated cache dict.
        """
        params = self.get_config()
        assert params["attention_monotonic"], (
            "function `incremental_encode` only available when attention_monotonic=True")
        if cache is None:
            cache = {}
        if cache is not None and len(cache) == 0:
            batch_size = tf.shape(inputs)[0]
            for lid in range(params["num_layers"]):
                cache[f"layer_{lid}"] = self._stacking_layers[lid].create_internal_cache()
            cache = tf.nest.map_structure(
                lambda ts: layer_utils.tile_tensor(ts, batch_size, axis=0), cache)
        if inputs.get_shape().ndims == 2:
            x = tf.expand_dims(inputs, axis=1)
            x_bias = None
        else:
            x = inputs
            if time is None:
                time = 0
            x_bias = layer_utils.lower_triangle_attention_bias(time + tf.shape(x)[1])[:, :, -tf.shape(x)[1]:]
        for idx, layer in enumerate(self._stacking_layers):
            layer_cache = None if cache is None else cache[f"layer_{idx}"]
            x = layer(x, x_bias, layer_cache, is_training=False)
        outputs = x
        if not params["post_normalize"]:
            outputs = self.quant(self._output_norm_layer(x), name="output_ln")
        return outputs, cache
