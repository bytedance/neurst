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

from neurst.layers.decoders.transformer_decoder import TransformerDecoder


def test_transformer_decoder():
    dmodel = 4
    batch_size = 2
    num_layers = 1
    num_self_attention_heads = 2
    hidden_size = dmodel
    filter_size = 16
    self_attention_dropout_rate = 0.1
    ffn_dropout_rate = 0.1
    layer_postprocess_dropout_rate = 0.1
    # max_len = 4
    # max_decoder_len = 3

    decoder = TransformerDecoder(
        num_layers=num_layers,
        num_attention_heads=num_self_attention_heads,
        hidden_size=hidden_size,
        filter_size=filter_size,
        attention_dropout_rate=self_attention_dropout_rate,
        ffn_dropout_rate=ffn_dropout_rate,
        layer_postprocess_dropout_rate=layer_postprocess_dropout_rate)
    encoder_outputs = tf.convert_to_tensor(
        [[[-0.37282175, 0.62301564, -2.0221813, -0.00875833],
          [0.31516594, -1.117763, -1.0697726, 0.80373234],
          [-0.717022, 0.3300997, -0.44306225, 1.550383],
          [-1.5516962, 0.6025011, 1.8262954, 0.42469704]],

         [[-0.98617625, 2.2856202, -1.3063533, 0.4174998],
          [1.5724765, 1.2201295, 1.1479746, 0.7810888],
          [0.8343642, -1.073388, 1.2718492, -0.7290778],
          [-1.4126722, 1.8000795, -2.118672, -0.1366007]]], dtype=tf.float32)
    encoder_inputs_padding = tf.convert_to_tensor(
        [[0, 0, 0, 0], [0, 0, 1., 1.]], dtype=tf.float32)
    decoder_inputs = tf.convert_to_tensor(
        [[[8.6675537e-01, 2.2135425e-01, 1.4054185e+00, -4.2268831e-01],
          [1.9606155e+00, -1.8318410e+00, -1.8158482e+00, -3.7030798e-01],
          [-1.1357157e-03, 5.5629879e-01, 6.6107117e-02, -1.7330967e+00]],

         [[-1.1870812e+00, -5.4499257e-01, -8.6622888e-01, -7.4098641e-01],
          [2.2233427e-01, 5.3582352e-01, 3.0567116e-01, 1.0201423e-01],
          [-1.8053315e+00, 7.2125041e-01, 1.0072237e+00, -2.0333264e+00]]], dtype=tf.float32)
    # test for training
    cache = decoder.create_decoding_internal_cache(
        encoder_outputs, encoder_inputs_padding, is_inference=False)
    _ = decoder(decoder_inputs, cache, is_training=False)
    for w in decoder.trainable_weights:
        if "layer_0/self_attention_prepost_wrapper/self_attention/output_transform/kernel" in w.name:
            tf.compat.v1.assign(w, tf.convert_to_tensor(
                [[0.39332086, -0.3676856, -0.50203305, 0.6782059],
                 [-0.41239128, -0.15406412, 0.3964849, -0.79016757],
                 [0.6749844, -0.09548753, 0.16253561, -0.0560202],
                 [-0.4699119, 0.82842, 0.35657936, -0.45770356]],
                dtype=tf.float32))
        elif "layer_0/self_attention_prepost_wrapper/self_attention/qkv_transform/kernel" in w.name:
            tf.compat.v1.assign(w, tf.convert_to_tensor(
                [[0.03949255, 0.32946128, 0.38817757, 0.47047406, 0.07609951,
                  0.03131855, 0.15958023, 0.3292094, 0.42809182, 0.27969742,
                  0.39156157, -0.604576],
                 [0.4869359, -0.590637, 0.3092571, 0.10321742, 0.45608515,
                  0.27015948, 0.2959339, 0.32079375, 0.480197, -0.35878542,
                  0.04467481, 0.467416],
                 [-0.40064478, -0.05089319, -0.0999378, -0.6048573, 0.4379304,
                  0.3692366, 0.39103013, 0.24920046, -0.37060317, -0.03119427,
                  0.25101495, -0.21076846],
                 [0.42842942, 0.48276085, -0.2498649, -0.0978691, -0.01024461,
                  -0.04072392, -0.43499938, -0.09718102, 0.18174142, 0.07100755,
                  -0.6075252, -0.3018506]],
                dtype=tf.float32))
        elif "layer_0/encdec_attention_prepost_wrapper/encdec_attention/output_transform/kernel" in w.name:
            tf.compat.v1.assign(w, tf.convert_to_tensor(
                [[-0.31871676, 0.46451026, -0.32600254, -0.42110354],
                 [0.45953768, -0.52176374, -0.47615638, -0.7818449],
                 [0.7724063, -0.25975162, -0.49630436, 0.4681155],
                 [0.7189149, 0.25591546, 0.2100411, -0.3439259]],
                dtype=tf.float32))
        elif "layer_0/encdec_attention_prepost_wrapper/encdec_attention/q_transform/kernel" in w.name:
            tf.compat.v1.assign(w, tf.convert_to_tensor(
                [[0.27346164, -0.12056953, 0.4617111, 0.3126462],
                 [-0.65311253, 0.24505383, 0.56249744, -0.5582411],
                 [-0.47464705, -0.60553044, 0.3019113, 0.33609575],
                 [-0.24644238, -0.16026068, -0.0945828, -0.05111927]],
                dtype=tf.float32))
        elif "layer_0/encdec_attention_prepost_wrapper/encdec_attention/kv_transform/kernel" in w.name:
            tf.compat.v1.assign(w, tf.convert_to_tensor(
                [[-0.4204824, -0.23150605, 0.12045383, -0.6538836, 0.29070246,
                  -0.38376695, 0.65055054, -0.51375425],
                 [0.67025226, 0.0928542, -0.56662744, 0.12781924, -0.6193744,
                  -0.61801594, 0.07964879, 0.16530299],
                 [-0.06940353, -0.08732289, 0.24984497, 0.18489975, 0.5354368,
                  -0.07608587, -0.5801205, -0.17658263],
                 [0.54784423, -0.39817223, -0.11673075, 0.14106786, -0.1637184,
                  0.00750518, -0.44365695, -0.38458544]],
                dtype=tf.float32))
        elif "layer_0/ffn_prepost_wrapper/ffn/dense1/kernel" in w.name:
            tf.compat.v1.assign(w, tf.convert_to_tensor(
                [[-2.9522404e-01, -1.1858380e-01, 1.3743329e-01, -3.3782017e-01,
                  -3.8876867e-01, 4.8396683e-01, 1.5062505e-01, -3.7749952e-01,
                  -2.9512924e-01, -1.6212821e-02, -1.8608570e-04, -4.1960135e-01,
                  5.3800035e-01, 2.7734953e-01, 5.5179596e-03, -3.4055352e-02],
                 [2.1051055e-01, 3.6151302e-01, 3.1045640e-01, -1.1510965e-01,
                  4.6738219e-01, 1.2504590e-01, -1.9454169e-01, 4.1786206e-01,
                  -3.7045652e-01, 3.3854598e-01, -5.0978750e-01, 5.2220762e-01,
                  1.6077441e-01, -3.9631999e-01, 2.1259248e-01, 2.3286474e-01],
                 [-1.0005751e-01, -5.0858349e-01, 3.6911082e-01, -5.1783592e-02,
                  7.1038425e-02, -1.1148521e-01, -5.3392905e-01, 3.6009926e-01,
                  7.9382658e-02, 1.0371411e-01, -5.0254786e-01, 1.7596281e-01,
                  -9.2926025e-03, -6.4194202e-04, -1.4125884e-02, 4.7321141e-01],
                 [2.8647327e-01, 2.6127762e-01, 4.5843053e-01, 4.9775457e-01,
                  3.8056010e-01, -4.0995055e-01, 3.6980593e-01, 3.3520699e-02,
                  -1.8056035e-03, 1.6578972e-02, 1.6026449e-01, -2.4952739e-01,
                  -3.1434530e-01, -1.3158950e-01, 7.9998970e-03, 1.1293548e-01]],
                dtype=tf.float32))
        elif "layer_0/ffn_prepost_wrapper/ffn/dense2/kernel" in w.name:
            tf.compat.v1.assign(w, tf.convert_to_tensor(
                [[0.2794218, 0.29263318, 0.42604703, -0.24461824],
                 [0.32469118, -0.2654639, 0.17872995, 0.06222689],
                 [-0.07604656, -0.29360557, -0.462821, 0.3731665],
                 [0.27989155, 0.53663385, -0.12042063, 0.34913152],
                 [-0.50028926, 0.08958912, 0.50753117, -0.03860039],
                 [0.12980306, -0.47548878, 0.5443562, -0.41777247],
                 [0.16824102, -0.5271052, -0.18454444, 0.2987221],
                 [0.22610295, -0.3761598, 0.4983195, 0.31664205],
                 [-0.36606842, -0.3778124, 0.01393354, 0.23516071],
                 [0.26510388, -0.47218412, 0.42749757, 0.22174352],
                 [0.4139307, 0.09682184, -0.1447433, -0.07231569],
                 [0.01711905, -0.18132755, 0.03224993, 0.2071482],
                 [0.12195373, -0.52764714, 0.48840046, -0.21843264],
                 [0.12467605, -0.45452338, 0.05892056, -0.2852741],
                 [-0.5464495, -0.4856094, -0.29271287, 0.10828984],
                 [0.37080926, 0.01543814, 0.10875225, -0.2678996]],
                dtype=tf.float32))

    assert numpy.sum((decoder(decoder_inputs, cache, is_training=False).numpy()
                      - numpy.array([[[0.4727962, -0.6863654, 1.387909, -1.1743398],
                                      [1.4770155, -1.2802002, 0.18456227, -0.38137752],
                                      [0.6776164, -0.4934968, 1.1886327, -1.3727522]],
                                     [[-1.6973993, 0.26954588, 0.59817475, 0.82967865],
                                      [-1.6315649, -0.0030859, 0.7861572, 0.8484935],
                                      [-1.4942819, 0.42606276, 1.246516, -0.17829692]]])) ** 2) < 1e-9

    # for inference
    cache = decoder.create_decoding_internal_cache(
        encoder_outputs, encoder_inputs_padding, is_inference=True)
    decoder_inputs = tf.convert_to_tensor(
        [[1.9606155e+00, -1.8318410e+00, -1.8158482e+00, -3.7030798e-01],
         [-1.1357157e-03, 5.5629879e-01, 6.6107117e-02, -1.7330967e+00]], dtype=tf.float32)
    assert numpy.sum(
        (decoder(decoder_inputs, cache, is_training=False).numpy()
         - numpy.array([[1.4581295, -1.3640043, -0.1138487, 0.01972346],
                        [-0.06228875, -1.0514979, 1.6223053, -0.5085185]])) ** 2) < 1e-9
    assert numpy.sum(
        (cache["decoding_states"]["layer_0"]["self_attention"]["keys"].numpy()
         - numpy.array(numpy.reshape([[[-0.63596207, -0.49432975, -0.36614707, 0.03477353]],
                                      [[0.6539597, 0.4846998, 1.2206339, 0.67560077]]],
                                     [batch_size, 1, num_self_attention_heads,
                                      hidden_size // num_self_attention_heads]))) ** 2) < 1e-9
    assert numpy.sum(
        (cache["decoding_states"]["layer_0"]["self_attention"]["values"].numpy()
         - numpy.array(numpy.reshape([[[0.6045396, 0.78576076, 0.3205938, -1.2158906]],
                                      [[0.14660448, -0.38737938, 1.2869109, 0.6795136]]],
                                     [batch_size, 1, num_self_attention_heads,
                                      hidden_size // num_self_attention_heads]))) ** 2) < 1e-9


if __name__ == "__main__":
    test_transformer_decoder()
