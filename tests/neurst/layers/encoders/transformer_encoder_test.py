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
import numpy
import tensorflow as tf

from neurst.layers.encoders.transformer_encoder import TransformerEncoder
from neurst.utils.misc import assert_equal_numpy


def test_transformer_encoder():
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

    encoder = TransformerEncoder(
        num_layers=num_layers,
        num_attention_heads=num_self_attention_heads,
        hidden_size=hidden_size,
        filter_size=filter_size,
        attention_dropout_rate=self_attention_dropout_rate,
        ffn_dropout_rate=ffn_dropout_rate,
        layer_postprocess_dropout_rate=layer_postprocess_dropout_rate)
    inputs = tf.convert_to_tensor(
        [[[-0.37282175, 0.62301564, -2.0221813, -0.00875833],
          [0.31516594, -1.117763, -1.0697726, 0.80373234],
          [-0.717022, 0.3300997, -0.44306225, 1.550383],
          [-1.5516962, 0.6025011, 1.8262954, 0.42469704]],

         [[-0.98617625, 2.2856202, -1.3063533, 0.4174998],
          [1.5724765, 1.2201295, 1.1479746, 0.7810888],
          [0.8343642, -1.073388, 1.2718492, -0.7290778],
          [-1.4126722, 1.8000795, -2.118672, -0.1366007]]], dtype=tf.float32)
    input_padding = tf.convert_to_tensor(
        [[0, 0, 0, 0], [0, 0, 1., 1.]], dtype=tf.float32)
    _ = encoder(inputs, input_padding, is_training=False)
    for w in encoder.trainable_weights:
        if "layer_0/self_attention_prepost_wrapper/self_attention/output_transform/kernel" in w.name:
            tf.compat.v1.assign(w, tf.convert_to_tensor(
                [[-0.04742211, -0.42928827, -0.54485893, -0.7514334],
                 [0.3391741, 0.61141425, -0.23809844, 0.27043575],
                 [-0.7315594, 0.8002729, -0.2958873, 0.698168],
                 [-0.59683925, -0.38270262, -0.59893274, -0.4040773]],
                dtype=tf.float32))
        elif "layer_0/self_attention_prepost_wrapper/self_attention/qkv_transform/kernel" in w.name:
            tf.compat.v1.assign(w, tf.convert_to_tensor(
                [[0.5907243, -0.5555184, 0.5612393, -0.2724994, 0.23405826,
                  0.38096863, -0.02200276, -0.26264596, 0.36556423, 0.10351193,
                  -0.1946517, 0.60423344],
                 [0.16057128, -0.4464733, 0.32041794, -0.30858415, 0.26626736,
                  0.579398, -0.19076341, 0.1072132, -0.43820834, 0.05253071,
                  0.08801651, -0.4995584],
                 [-0.48593724, 0.1275987, 0.15794194, -0.4632662, 0.54038125,
                  -0.45666856, -0.16076824, 0.43855423, 0.32468224, -0.1931965,
                  -0.42853987, 0.2411524],
                 [-0.32923162, -0.06395793, 0.33392805, -0.46701026, -0.06507087,
                  -0.61020637, 0.545703, -0.23786944, -0.2854141, -0.1698403,
                  -0.1244911, 0.40745395]], dtype=tf.float32))
        elif "layer_0/ffn_prepost_wrapper/ffn/dense1/kernel" in w.name:
            tf.compat.v1.assign(w, tf.convert_to_tensor(
                [[-0.14616564, 0.30248666, 0.5319947, 0.5002098, 0.2705282,
                  -0.21612385, -0.3336154, 0.03436899, 0.26958936, 0.26834202,
                  0.0843057, -0.50728637, 0.19995207, -0.3930181, -0.4985036,
                  0.33232063],
                 [-0.04522616, -0.20491397, -0.19712418, 0.18106508, 0.33636385,
                  0.4030161, -0.30252987, 0.11853886, 0.2238034, 0.3744824,
                  -0.28127617, -0.03388816, 0.32239246, -0.25639355, 0.02382994,
                  0.34818083],
                 [0.4456296, -0.48834273, -0.26576972, 0.28717202, 0.02354515,
                  -0.2434513, -0.26277977, -0.05434859, 0.09830189, 0.08207488,
                  -0.28704825, -0.19418713, 0.47731507, 0.14538354, -0.3832153,
                  -0.5143249],
                 [0.33276683, -0.248025, -0.13612089, -0.15473047, 0.33012676,
                  -0.39191568, -0.32679468, 0.52579904, -0.17942387, -0.39317977,
                  0.13891649, -0.17397407, -0.19002154, 0.05117792, 0.34706026,
                  0.11179692]], dtype=tf.float32))
        elif "layer_0/ffn_prepost_wrapper/ffn/dense2/kernel" in w.name:
            tf.compat.v1.assign(w, tf.convert_to_tensor(
                [[0.18234771, 0.23902518, 0.4304248, -0.05616844],
                 [-0.01435661, 0.11098373, 0.5370636, -0.5271752],
                 [-0.3239155, 0.5083337, 0.43396413, -0.47642848],
                 [0.31562793, -0.04991594, 0.530545, -0.51263183],
                 [0.10357869, 0.2883237, 0.16929054, 0.18414849],
                 [-0.30361128, -0.2045235, 0.05544132, 0.22116774],
                 [0.05548936, -0.11504656, 0.13726586, -0.13652831],
                 [0.5011635, 0.45315623, -0.35243145, 0.17173672],
                 [-0.52015716, 0.42873853, -0.09965438, -0.45107275],
                 [0.00233686, 0.2797522, 0.2702785, 0.33721972],
                 [0.10216439, -0.14768293, -0.5122431, -0.3882924],
                 [-0.44032216, -0.09983957, -0.41019306, -0.26434696],
                 [0.50977015, -0.18238857, 0.54663074, 0.05787665],
                 [0.3197481, -0.45845133, -0.14075449, -0.33339915],
                 [0.10717738, 0.28995162, 0.47179937, 0.01342988],
                 [0.37111026, -0.31352338, 0.37098122, 0.3895113]],
                dtype=tf.float32))

    assert_equal_numpy(encoder(inputs, input_padding, is_training=False).numpy(),
                       numpy.array([[[-0.2709918, 0.95230484, -1.5212451, 0.83993214],
                                     [0.7688386, -0.69726187, -1.2441225, 1.1725458],
                                     [-1.1408244, 0.57164305, -0.76654106, 1.3357224],
                                     [-1.5286305, 0.23827001, 1.267273, 0.02308742]],
                                    [[-1.0156152, 1.4036102, -0.8733843, 0.48538923],
                                     [-0.60578734, 0.23574206, 1.5095922, -1.1395471],
                                     [0.53838307, -0.7913252, 1.3617758, -1.1088338],
                                     [-0.8927619, 1.3975127, -1.001557, 0.49680638]]]))


def test_incremental_encode():
    max_time = 5
    inputs = tf.random.normal([1, max_time, 8])
    inputs_padding = tf.convert_to_tensor([[0., 0., 0., 0., 0., ]], dtype=tf.float32)
    encoder = TransformerEncoder(
        num_layers=2,
        hidden_size=8,
        num_attention_heads=2,
        filter_size=20,
        attention_monotonic=True,
    )
    encoder_outputs = encoder(inputs, inputs_padding, is_training=False)

    incremental_encoder_outputs, _ = encoder.incremental_encode(inputs, {}, time=0)
    assert_equal_numpy(encoder_outputs.numpy(), incremental_encoder_outputs.numpy(), 1e-5)

    incremental_encoder_outputs0, cache = encoder.incremental_encode(inputs[:, :2], {}, time=0)
    incremental_encoder_outputs1, cache = encoder.incremental_encode(inputs[:, 2], cache, time=2)
    incremental_encoder_outputs2, cache = encoder.incremental_encode(inputs[:, 3:], cache, time=3)

    assert_equal_numpy(encoder_outputs.numpy(),
                       tf.concat([incremental_encoder_outputs0,
                                  incremental_encoder_outputs1,
                                  incremental_encoder_outputs2], axis=1), 1e-5)


if __name__ == "__main__":
    test_transformer_encoder()
    test_incremental_encode()
