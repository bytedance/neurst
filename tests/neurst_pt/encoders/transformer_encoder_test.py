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
import torch

from neurst.layers.encoders.transformer_encoder import TransformerEncoder as TFTransformerEncoder
from neurst.utils.misc import assert_equal_numpy
from neurst_pt.layers.encoders.transformer_encoder import TransformerEncoder


def test_transformer_encoder_prenorm():
    # batch_size = 2
    # max_len = 4
    dmodel = 4
    num_layers = 1
    num_self_attention_heads = 2
    hidden_size = dmodel
    filter_size = 16
    self_attention_dropout_rate = 0.1
    ffn_dropout_rate = 0.1
    layer_postprocess_dropout_rate = 0.1

    tf_encoder = TFTransformerEncoder(
        num_layers=num_layers,
        num_attention_heads=num_self_attention_heads,
        hidden_size=hidden_size,
        filter_size=filter_size,
        attention_dropout_rate=self_attention_dropout_rate,
        ffn_dropout_rate=ffn_dropout_rate,
        layer_postprocess_dropout_rate=layer_postprocess_dropout_rate)
    pt_encoder = TransformerEncoder(
        num_layers=num_layers,
        num_attention_heads=num_self_attention_heads,
        hidden_size=hidden_size,
        filter_size=filter_size,
        attention_dropout_rate=self_attention_dropout_rate,
        ffn_dropout_rate=ffn_dropout_rate,
        layer_postprocess_dropout_rate=layer_postprocess_dropout_rate)

    inputs = [[[-0.37282175, 0.62301564, -2.0221813, -0.00875833],
               [0.31516594, -1.117763, -1.0697726, 0.80373234],
               [-0.717022, 0.3300997, -0.44306225, 1.550383],
               [-1.5516962, 0.6025011, 1.8262954, 0.42469704]],

              [[-0.98617625, 2.2856202, -1.3063533, 0.4174998],
               [1.5724765, 1.2201295, 1.1479746, 0.7810888],
               [0.8343642, -1.073388, 1.2718492, -0.7290778],
               [-1.4126722, 1.8000795, -2.118672, -0.1366007]]]
    input_padding = [[0, 0, 0, 0], [0, 0, 1., 1.]]
    tf_inp = tf.convert_to_tensor(inputs, dtype=tf.float32)
    pt_inp = torch.FloatTensor(inputs)
    tf_inppad = tf.convert_to_tensor(input_padding, dtype=tf.float32)
    pt_inppad = torch.FloatTensor(input_padding)
    _ = tf_encoder(tf_inp, tf_inppad, is_training=False)
    _ = pt_encoder(pt_inp, pt_inppad, is_training=False)
    pt_encoder._output_norm_layer.weight.data = torch.FloatTensor(tf_encoder._output_norm_layer.gamma.numpy())
    pt_encoder._output_norm_layer.bias.data = torch.FloatTensor(tf_encoder._output_norm_layer.beta.numpy())
    pt_encoder._stacking_layers[0][0]._layer._qkv_transform_layer._kernel.data = torch.FloatTensor(
        tf_encoder._stacking_layers[0]._selfatt_layer._layer._qkv_transform_layer._kernel.numpy())
    pt_encoder._stacking_layers[0][0]._layer._qkv_transform_layer._bias.data = torch.FloatTensor(
        tf_encoder._stacking_layers[0]._selfatt_layer._layer._qkv_transform_layer._bias.numpy())
    pt_encoder._stacking_layers[0][0]._layer._output_transform_layer._kernel.data = torch.FloatTensor(
        tf_encoder._stacking_layers[0]._selfatt_layer._layer._output_transform_layer._kernel.numpy())
    pt_encoder._stacking_layers[0][0]._layer._output_transform_layer._bias.data = torch.FloatTensor(
        tf_encoder._stacking_layers[0]._selfatt_layer._layer._output_transform_layer._bias.numpy())
    pt_encoder._stacking_layers[0][1]._layer._dense1.weight.data = torch.FloatTensor(
        tf_encoder._stacking_layers[0]._ffn_layer._layer._conv1.kernel.numpy().transpose([1, 0]))
    pt_encoder._stacking_layers[0][1]._layer._dense1.bias.data = torch.FloatTensor(
        tf_encoder._stacking_layers[0]._ffn_layer._layer._conv1.bias.numpy())
    pt_encoder._stacking_layers[0][1]._layer._dense2.weight.data = torch.FloatTensor(
        tf_encoder._stacking_layers[0]._ffn_layer._layer._conv2.kernel.numpy().transpose([1, 0]))
    pt_encoder._stacking_layers[0][1]._layer._dense2.bias.data = torch.FloatTensor(
        tf_encoder._stacking_layers[0]._ffn_layer._layer._conv2.bias.numpy())
    pt_encoder._stacking_layers[0][0]._norm_layer.weight.data = torch.FloatTensor(
        tf_encoder._stacking_layers[0]._selfatt_layer._norm_layer.gamma.numpy())
    pt_encoder._stacking_layers[0][0]._norm_layer.bias.data = torch.FloatTensor(
        tf_encoder._stacking_layers[0]._selfatt_layer._norm_layer.beta.numpy())
    pt_encoder._stacking_layers[0][1]._norm_layer.weight.data = torch.FloatTensor(
        tf_encoder._stacking_layers[0]._ffn_layer._norm_layer.gamma.numpy())
    pt_encoder._stacking_layers[0][1]._norm_layer.bias.data = torch.FloatTensor(
        tf_encoder._stacking_layers[0]._ffn_layer._norm_layer.beta.numpy())
    assert_equal_numpy(tf_encoder(tf_inp, tf_inppad, is_training=False).numpy(),
                       pt_encoder(pt_inp, pt_inppad, is_training=False).detach().numpy(), 1e-5)


def test_transformer_encoder_postnorm():
    # batch_size = 2
    # max_len = 4
    dmodel = 4
    num_layers = 1
    num_self_attention_heads = 2
    hidden_size = dmodel
    filter_size = 16
    self_attention_dropout_rate = 0.1
    ffn_dropout_rate = 0.1
    layer_postprocess_dropout_rate = 0.1

    tf_encoder = TFTransformerEncoder(
        num_layers=num_layers,
        num_attention_heads=num_self_attention_heads,
        hidden_size=hidden_size,
        filter_size=filter_size,
        attention_dropout_rate=self_attention_dropout_rate,
        ffn_dropout_rate=ffn_dropout_rate,
        layer_postprocess_dropout_rate=layer_postprocess_dropout_rate,
        post_normalize=True)
    pt_encoder = TransformerEncoder(
        num_layers=num_layers,
        num_attention_heads=num_self_attention_heads,
        hidden_size=hidden_size,
        filter_size=filter_size,
        attention_dropout_rate=self_attention_dropout_rate,
        ffn_dropout_rate=ffn_dropout_rate,
        layer_postprocess_dropout_rate=layer_postprocess_dropout_rate,
        post_normalize=True)

    inputs = [[[-0.37282175, 0.62301564, -2.0221813, -0.00875833],
               [0.31516594, -1.117763, -1.0697726, 0.80373234],
               [-0.717022, 0.3300997, -0.44306225, 1.550383],
               [-1.5516962, 0.6025011, 1.8262954, 0.42469704]],

              [[-0.98617625, 2.2856202, -1.3063533, 0.4174998],
               [1.5724765, 1.2201295, 1.1479746, 0.7810888],
               [0.8343642, -1.073388, 1.2718492, -0.7290778],
               [-1.4126722, 1.8000795, -2.118672, -0.1366007]]]
    input_padding = [[0, 0, 0, 0], [0, 0, 1., 1.]]
    tf_inp = tf.convert_to_tensor(inputs, dtype=tf.float32)
    pt_inp = torch.FloatTensor(inputs)
    tf_inppad = tf.convert_to_tensor(input_padding, dtype=tf.float32)
    pt_inppad = torch.FloatTensor(input_padding)
    _ = tf_encoder(tf_inp, tf_inppad, is_training=False)
    _ = pt_encoder(pt_inp, pt_inppad, is_training=False)
    pt_encoder._stacking_layers[0][0]._layer._qkv_transform_layer._kernel.data = torch.FloatTensor(
        tf_encoder._stacking_layers[0]._selfatt_layer._layer._qkv_transform_layer._kernel.numpy())
    pt_encoder._stacking_layers[0][0]._layer._qkv_transform_layer._bias.data = torch.FloatTensor(
        tf_encoder._stacking_layers[0]._selfatt_layer._layer._qkv_transform_layer._bias.numpy())
    pt_encoder._stacking_layers[0][0]._layer._output_transform_layer._kernel.data = torch.FloatTensor(
        tf_encoder._stacking_layers[0]._selfatt_layer._layer._output_transform_layer._kernel.numpy())
    pt_encoder._stacking_layers[0][0]._layer._output_transform_layer._bias.data = torch.FloatTensor(
        tf_encoder._stacking_layers[0]._selfatt_layer._layer._output_transform_layer._bias.numpy())
    pt_encoder._stacking_layers[0][1]._layer._dense1.weight.data = torch.FloatTensor(
        tf_encoder._stacking_layers[0]._ffn_layer._layer._conv1.kernel.numpy().transpose([1, 0]))
    pt_encoder._stacking_layers[0][1]._layer._dense1.bias.data = torch.FloatTensor(
        tf_encoder._stacking_layers[0]._ffn_layer._layer._conv1.bias.numpy())
    pt_encoder._stacking_layers[0][1]._layer._dense2.weight.data = torch.FloatTensor(
        tf_encoder._stacking_layers[0]._ffn_layer._layer._conv2.kernel.numpy().transpose([1, 0]))
    pt_encoder._stacking_layers[0][1]._layer._dense2.bias.data = torch.FloatTensor(
        tf_encoder._stacking_layers[0]._ffn_layer._layer._conv2.bias.numpy())
    pt_encoder._stacking_layers[0][0]._norm_layer.weight.data = torch.FloatTensor(
        tf_encoder._stacking_layers[0]._selfatt_layer._norm_layer.gamma.numpy())
    pt_encoder._stacking_layers[0][0]._norm_layer.bias.data = torch.FloatTensor(
        tf_encoder._stacking_layers[0]._selfatt_layer._norm_layer.beta.numpy())
    pt_encoder._stacking_layers[0][1]._norm_layer.weight.data = torch.FloatTensor(
        tf_encoder._stacking_layers[0]._ffn_layer._norm_layer.gamma.numpy())
    pt_encoder._stacking_layers[0][1]._norm_layer.bias.data = torch.FloatTensor(
        tf_encoder._stacking_layers[0]._ffn_layer._norm_layer.beta.numpy())
    assert_equal_numpy(tf_encoder(tf_inp, tf_inppad, is_training=False).numpy(),
                       pt_encoder(pt_inp, pt_inppad, is_training=False).detach().numpy(), 1e-5)


if __name__ == "__main__":
    test_transformer_encoder_prenorm()
    test_transformer_encoder_postnorm()
