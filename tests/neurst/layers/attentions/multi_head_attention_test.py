import numpy
import tensorflow as tf

from neurst.layers.attentions.multi_head_attention import MultiHeadAttention, MultiHeadSelfAttention


def test_multihead_attention():
    # num_input = 1
    # batch_size = 1
    # length = 2
    num_heads = 2
    num_units = 4
    dropout_rate = 0.
    output_depth = 3
    attention_layer = MultiHeadAttention(
        num_heads=num_heads,
        num_units=num_units,
        output_depth=output_depth,
        attention_dropout_rate=dropout_rate)
    query = tf.convert_to_tensor(
        [[[-1.3010577], [0.79014736]]], dtype=tf.float32)
    memory = tf.convert_to_tensor(
        [[[-1.1650294], [-0.88871276]]], dtype=tf.float32)
    # build layer
    attention_layer(query, memory=memory)
    predefined_weights = []
    for w in attention_layer.get_weights():
        # if "output_transform/kernel" in w.name:
        if w.shape == (4, 3):
            predefined_weights.append(
                numpy.array([[-0.18127477, -0.19210565, 0.6076416],
                             [0.43351638, 0.33151555, -0.49277717],
                             [-0.7911906, -0.48433813, 0.19314659],
                             [0.20843744, 0.76494586, -0.24448353]]))
        # elif "output_transform/bias" in w.name:
        elif w.shape == (3,):
            predefined_weights.append(
                numpy.array([0., 0., 0.]))
        # elif "q_transform/kernel" in w.name:
        elif w.shape == (1, 4):
            predefined_weights.append(
                numpy.array([[-0.8584202, 0.7467462, -0.9890406, 0.87380886]]))
        # elif "q_transform/bias" in w.name:
        elif w.shape == (4,):
            predefined_weights.append(
                numpy.array([0., 0., 0., 0.]))
        # elif "kv_transform/kernel" in w.name:
        elif w.shape == (1, 8):
            predefined_weights.append(
                numpy.array([[-0.741413, 0.05889565, -0.5399234, -0.5455819, 0.05704415,
                              0.75564325, -0.01531863, 0.7374855]]))
        # elif "kv_transform/bias" in w.name:
        elif w.shape == (8,):
            predefined_weights.append(
                numpy.array([0., 0., 0., 0., 0., 0., 0., 0.]))
    attention_layer.set_weights(predefined_weights)
    output = attention_layer(query, memory, is_training=False)
    assert numpy.sum((output.numpy() - numpy.array(
        [[[-0.5000115, -0.8363301, 0.5391713],
          [-0.49366233, -0.83081436, 0.5324018]]])) ** 2) < 1e-9


def test_multiheadself_attention():
    # num_input = 2
    # batch_size = 1
    # length = 2
    num_heads = 2
    num_units = 4
    dropout_rate = 0.
    output_depth = 3
    attention_layer = MultiHeadSelfAttention(
        num_heads=num_heads,
        num_units=num_units,
        output_depth=output_depth,
        attention_dropout_rate=dropout_rate)
    query = tf.convert_to_tensor(
        [[[-1.3010577, -2.3010577], [0.79014736, -0.79014736]]], dtype=tf.float32)
    bias = tf.convert_to_tensor([[-0.2276893, 0.11865579]], dtype=tf.float32)
    # build layer
    attention_layer(query)
    predefined_weights = []
    for w in attention_layer.get_weights():
        # if "output_transform/kernel" in w.name:
        if w.shape == (4, 3):
            predefined_weights.append(
                numpy.array([[-0.43040165, 0.841504, 0.8552141],
                             [-0.0111503, -0.14350098, -0.0815115],
                             [-0.54251707, 0.90105367, 0.7500602],
                             [-0.69162977, -0.10975248, 0.80686176]]))
        # elif "output_transform/bias" in w.name:
        elif w.shape == (3,):
            predefined_weights.append(
                numpy.array([0., 0., 0.]))
        # elif "qkv_transform/kernel" in w.name:
        elif w.shape == (2, 12):
            predefined_weights.append(
                numpy.array([[0.33632386, -0.5640344, -0.35587463, -0.5443644, -0.32000348,
                              -0.16645259, 0.38587856, 0.32863826, -0.5743795, 0.56257737,
                              0.17102379, -0.3622746],
                             [-0.1831755, -0.60358423, -0.55541337, 0.47493768, -0.37104,
                              -0.14766628, -0.64589596, 0.13060516, 0.4516778, -0.39557537,
                              0.08319885, 0.5473006]]))
        # elif "qkv_transform/bias" in w.name:
        elif w.shape == (12,):
            predefined_weights.append(
                numpy.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
    attention_layer.set_weights(predefined_weights)
    output = attention_layer(query, bias=bias, is_training=False)
    assert numpy.sum((output.numpy() - numpy.array(
        [[[0.8281387, -0.55132246, -1.1976832],
          [0.8589912, -0.61944896, -1.2623004]]])) ** 2) < 1e-9


def test_multiheadself_attention_under_dec():
    # num_input = 2
    # length = 1
    num_heads = 2
    num_units = 4
    dropout_rate = 0.
    output_depth = 3
    batch_size = 1
    already_time = 2
    attention_layer = MultiHeadSelfAttention(
        num_heads=num_heads,
        num_units=num_units,
        output_depth=output_depth,
        attention_dropout_rate=dropout_rate)
    query = tf.convert_to_tensor(
        [[[-1.3010577, -2.3010577]]], dtype=tf.float32)
    # build layer
    attention_layer(query)
    predefined_weights = []
    for w in attention_layer.get_weights():
        # if "output_transform/kernel" in w.name:
        if w.shape == (4, 3):
            predefined_weights.append(
                numpy.array([[-0.43040165, 0.841504, 0.8552141],
                             [-0.0111503, -0.14350098, -0.0815115],
                             [-0.54251707, 0.90105367, 0.7500602],
                             [-0.69162977, -0.10975248, 0.80686176]]))
        # elif "output_transform/bias" in w.name:
        elif w.shape == (3,):
            predefined_weights.append(
                numpy.array([0., 0., 0.]))
        # elif "qkv_transform/kernel" in w.name:
        elif w.shape == (2, 12):
            predefined_weights.append(
                numpy.array([[0.33632386, -0.5640344, -0.35587463, -0.5443644, -0.32000348,
                              -0.16645259, 0.38587856, 0.32863826, -0.5743795, 0.56257737,
                              0.17102379, -0.3622746],
                             [-0.1831755, -0.60358423, -0.55541337, 0.47493768, -0.37104,
                              -0.14766628, -0.64589596, 0.13060516, 0.4516778, -0.39557537,
                              0.08319885, 0.5473006]]))
        # elif "qkv_transform/bias" in w.name:
        elif w.shape == (12,):
            predefined_weights.append(
                numpy.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
    attention_layer.set_weights(predefined_weights)
    cache = {
        "keys":
            tf.reshape(tf.convert_to_tensor(numpy.array([[[-0.46546218, -1.0054358, 0.42906007, -1.6854379],
                                                          [1.078194, 1.1625745, -0.25033495, -1.980812]]]),
                                            dtype=tf.float32),
                       [batch_size, already_time, num_heads, num_units // num_heads]),
        "values":
            tf.reshape(tf.convert_to_tensor(numpy.array([[[-1.2360295, 0.69050753, -1.8204833, 0.23788007],
                                                          [2.3751693, -1.8772833, -0.2574517, 1.3010416]]]),
                                            dtype=tf.float32),
                       [batch_size, already_time, num_heads, num_units // num_heads])
    }
    attention_layer(query, cache=cache, is_training=False)
    output = attention_layer(query, cache=cache, is_training=False)
    assert numpy.sum((output.numpy() - numpy.array(
        [[[0.14431843, 0.4875261, 0.23705271]]])) ** 2) < 1e-9
    assert numpy.sum(
        (tf.reshape(cache["keys"], [batch_size, already_time + 2, num_units]).numpy()
         - numpy.array([[[-0.46546218, -1.0054358, 0.42906007, -1.6854379],
                         [1.078194, 1.1625745, -0.25033495, -1.980812],
                         [1.2701274, 0.5563531, 0.9841937, -0.72810733],
                         [1.2701274, 0.5563531, 0.9841937, -0.72810733]]]) ** 2)) < 1e-9
    assert numpy.sum(
        (tf.reshape(cache["keys"], [batch_size, already_time + 2, num_units]).numpy()
         - numpy.array([[[-1.2360295, 0.69050753, -1.8204833, 0.23788007],
                         [2.3751693, -1.8772833, -0.2574517, 1.3010416],
                         [-0.29203582, 0.17829615, -0.41395718, -0.78803015],
                         [-0.29203582, 0.17829615, -0.41395718, -0.78803015]]]) ** 2)) < 1e-9


if __name__ == "__main__":
    test_multihead_attention()
    test_multiheadself_attention()
    test_multiheadself_attention_under_dec()
