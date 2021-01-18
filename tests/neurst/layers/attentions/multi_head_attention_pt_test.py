import numpy
import tensorflow as tf
import torch

from neurst.layers.attentions.multi_head_attention import MultiHeadAttention, MultiHeadSelfAttention
from neurst.utils.misc import assert_equal_numpy
from neurst_pt.layers.attentions.multi_head_attention import MultiHeadAttention as PTMultiHeadAttention
from neurst_pt.layers.attentions.multi_head_attention import MultiHeadSelfAttention as PTMultiHeadSelfAttention


def test_multihead_attention():
    length_q = 2
    length_m = 3
    num_heads = 2
    num_units = 4
    dropout_rate = 0.
    output_depth = 3
    tf_att_layer = MultiHeadAttention(num_heads=num_heads, num_units=num_units,
                                      output_depth=output_depth, attention_dropout_rate=dropout_rate)
    pt_att_layer = PTMultiHeadAttention(input_depth=num_units, num_heads=num_heads, num_units=num_units,
                                        output_depth=output_depth, attention_dropout_rate=dropout_rate)

    query = numpy.random.rand(1, length_q, num_units)
    memory = numpy.random.rand(1, length_m, num_units)
    tf_query = tf.convert_to_tensor(query, dtype=tf.float32)
    tf_memory = tf.convert_to_tensor(memory, dtype=tf.float32)
    pt_query = torch.FloatTensor(query)
    pt_memory = torch.FloatTensor(memory)
    # build layer
    _ = tf_att_layer(tf_query, memory=tf_memory)
    _ = pt_att_layer(pt_query, memory=pt_memory)
    pt_att_layer._q_transform_layer._kernel.data = torch.FloatTensor(
        tf_att_layer._q_transform_layer._kernel.numpy())
    pt_att_layer._q_transform_layer._bias.data = torch.FloatTensor(
        tf_att_layer._q_transform_layer._bias.numpy())
    pt_att_layer._kv_transform_layer._kernel.data = torch.FloatTensor(
        tf_att_layer._kv_transform_layer._kernel.numpy())
    pt_att_layer._kv_transform_layer._bias.data = torch.FloatTensor(
        tf_att_layer._kv_transform_layer._bias.numpy())
    pt_att_layer._output_transform_layer._kernel.data = torch.FloatTensor(
        tf_att_layer._output_transform_layer._kernel.numpy())
    pt_att_layer._output_transform_layer._bias.data = torch.FloatTensor(
        tf_att_layer._output_transform_layer._bias.numpy())
    assert_equal_numpy(tf_att_layer(tf_query, tf_memory, is_training=False).numpy(),
                       pt_att_layer(pt_query, pt_memory, is_training=False).detach().numpy())


def test_multiheadself_attention():
    length_q = 4
    num_heads = 2
    num_units = 4
    dropout_rate = 0.
    output_depth = 3
    tf_att_layer = MultiHeadSelfAttention(num_heads=num_heads, num_units=num_units,
                                          output_depth=output_depth, attention_dropout_rate=dropout_rate)
    pt_att_layer = PTMultiHeadSelfAttention(input_depth=num_units, num_heads=num_heads, num_units=num_units,
                                            output_depth=output_depth, attention_dropout_rate=dropout_rate)

    query = numpy.random.rand(1, length_q, num_units)
    bias = numpy.random.rand(1, length_q)
    tf_query = tf.convert_to_tensor(query, dtype=tf.float32)
    tf_bias = tf.convert_to_tensor(bias, dtype=tf.float32)
    pt_query = torch.FloatTensor(query)
    pt_bias = torch.FloatTensor(bias)
    # build layer
    _ = tf_att_layer(tf_query)
    _ = pt_att_layer(pt_query)
    pt_att_layer._qkv_transform_layer._kernel.data = torch.FloatTensor(
        tf_att_layer._qkv_transform_layer._kernel.numpy())
    pt_att_layer._qkv_transform_layer._bias.data = torch.FloatTensor(
        tf_att_layer._qkv_transform_layer._bias.numpy())
    pt_att_layer._output_transform_layer._kernel.data = torch.FloatTensor(
        tf_att_layer._output_transform_layer._kernel.numpy())
    pt_att_layer._output_transform_layer._bias.data = torch.FloatTensor(
        tf_att_layer._output_transform_layer._bias.numpy())
    assert_equal_numpy(tf_att_layer(tf_query, bias=tf_bias, is_training=False).numpy(),
                       pt_att_layer(pt_query, bias=pt_bias, is_training=False).detach().numpy())


def test_multiheadself_attention_under_dec():
    num_heads = 2
    num_units = 4
    dropout_rate = 0.
    output_depth = 3
    tf_att_layer = MultiHeadSelfAttention(num_heads=num_heads, num_units=num_units,
                                          output_depth=output_depth, attention_dropout_rate=dropout_rate)
    pt_att_layer = PTMultiHeadSelfAttention(input_depth=num_units, num_heads=num_heads, num_units=num_units,
                                            output_depth=output_depth, attention_dropout_rate=dropout_rate)

    query = numpy.random.rand(1, 1, num_units)
    tf_query = tf.convert_to_tensor(query, dtype=tf.float32)
    pt_query = torch.FloatTensor(query)
    # build layer
    _ = tf_att_layer(tf_query)
    _ = pt_att_layer(pt_query)
    pt_att_layer._qkv_transform_layer._kernel.data = torch.FloatTensor(
        tf_att_layer._qkv_transform_layer._kernel.numpy())
    pt_att_layer._qkv_transform_layer._bias.data = torch.FloatTensor(
        tf_att_layer._qkv_transform_layer._bias.numpy())
    pt_att_layer._output_transform_layer._kernel.data = torch.FloatTensor(
        tf_att_layer._output_transform_layer._kernel.numpy())
    pt_att_layer._output_transform_layer._bias.data = torch.FloatTensor(
        tf_att_layer._output_transform_layer._bias.numpy())

    cache = {
        "keys": numpy.array([[[-0.46546218, -1.0054358, 0.42906007, -1.6854379],
                              [1.078194, 1.1625745, -0.25033495, -1.980812]]]),
        "values": numpy.array([[[-1.2360295, 0.69050753, -1.8204833, 0.23788007],
                                [2.3751693, -1.8772833, -0.2574517, 1.3010416]]]), }
    tf_cache = {"keys": tf.reshape(tf.convert_to_tensor(cache["keys"], dtype=tf.float32),
                                   [1, 2, num_heads, num_units // num_heads]),
                "values": tf.reshape(tf.convert_to_tensor(cache["values"], dtype=tf.float32),
                                     [1, 2, num_heads, num_units // num_heads])}
    pt_cache = {"keys": torch.reshape(torch.FloatTensor(cache["keys"]), [1, 2, num_heads, num_units // num_heads]),
                "values": torch.reshape(torch.FloatTensor(cache["values"]), [1, 2, num_heads, num_units // num_heads])}
    assert_equal_numpy(tf_att_layer(tf_query, cache=tf_cache, is_training=False).numpy(),
                       pt_att_layer(pt_query, cache=pt_cache, is_training=False).detach().numpy())


if __name__ == "__main__":
    test_multihead_attention()
    test_multiheadself_attention()
    test_multiheadself_attention_under_dec()
