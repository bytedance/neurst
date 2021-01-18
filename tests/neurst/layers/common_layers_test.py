import numpy
import tensorflow as tf

from neurst.layers.common_layers import (MultiHeadDenseLayer, PositionEmbeddingWrapper, PrePostProcessingWrapper,
                                         TransformerFFN)
from neurst.layers.modalities.text_modalities import WordEmbeddingSharedWeights


def test_ffn():
    ffn_layer = TransformerFFN(4, 3, 0.1, name="ffn")
    output = ffn_layer(tf.convert_to_tensor([[[1, 2.]]], dtype=tf.float32))
    assert output.get_shape() == tf.TensorShape([1, 1, 3])
    for w in ffn_layer.trainable_weights:
        if w.name == "ffn/dense1/kernel:0":
            assert w.get_shape() == tf.TensorShape([2, 4])
        elif w.name == "ffn/dense1/bias:0":
            assert w.get_shape() == tf.TensorShape([4, ])
        elif w.name == "ffn/dense2/kernel:0":
            assert w.get_shape() == tf.TensorShape([4, 3])
        elif w.name == "ffn/dense2/bias:0":
            assert w.get_shape() == tf.TensorShape([3, ])
        else:
            raise ValueError


def test_prepost():
    layer = TransformerFFN(4, 3, 0.1, name="ffn")
    prepost_layer = PrePostProcessingWrapper(
        layer, dropout_rate=0.1, name="lpp")
    output = prepost_layer(
        tf.convert_to_tensor([[1, 2, 3.]], dtype=tf.float32),
        is_training=True)
    assert output.get_shape() == tf.TensorShape([1, 3])


def test_multihead_dense():
    num_heads = 3
    output_size = 6
    non_out_layer = MultiHeadDenseLayer(
        output_size, num_heads, use_bias=True, is_output_transform=False,
        name="nonoutput_transform")
    inputs = tf.convert_to_tensor(numpy.random.randn(2, 3, 6), dtype=tf.float32)
    layer_out = non_out_layer(inputs)
    kernel, bias = None, None
    for w in non_out_layer.trainable_weights:
        if "kernel" in w.name:
            kernel = w
        else:
            bias = w
    manual_out = tf.einsum("abc,cd->abd", inputs, kernel) + bias
    manual_out = tf.reshape(manual_out, tf.concat(
        [tf.shape(manual_out)[:-1], [num_heads, output_size // num_heads]], axis=0))
    assert numpy.sum((manual_out.numpy() - layer_out.numpy()) ** 2) < 1e-9

    num_inputs_per_head = 5
    out_layer = MultiHeadDenseLayer(
        output_size, num_heads, use_bias=True, is_output_transform=True,
        name="output_transform")
    inputs = tf.convert_to_tensor(numpy.random.randn(
        1, 2, num_heads, num_inputs_per_head), dtype=tf.float32)
    layer_out = out_layer(inputs)
    kernel, bias = None, None
    for w in out_layer.trainable_weights:
        if "kernel" in w.name:
            kernel = w
        else:
            bias = w
    manual_out = tf.matmul(
        tf.reshape(inputs, tf.concat([tf.shape(inputs)[:-2], [-1]], 0)),
        kernel) + bias
    assert numpy.sum((manual_out.numpy() - layer_out.numpy()) ** 2) < 1e-9

    output_size1 = 9
    non_out_multi_layer = MultiHeadDenseLayer(
        [output_size, output_size1], num_heads, use_bias=True,
        is_output_transform=False, name="nonoutput_transform")
    inputs = tf.convert_to_tensor(numpy.random.randn(2, 3, 6), dtype=tf.float32)
    layer_out0, layer_out1 = non_out_multi_layer(inputs)
    kernel, bias = None, None
    for w in non_out_multi_layer.trainable_weights:
        if "kernel" in w.name:
            kernel = w
        else:
            bias = w
    manual_out = tf.einsum("abc,cd->abd", inputs, kernel) + bias
    manual_out0, manual_out1 = tf.split(
        manual_out, [output_size, output_size1], axis=-1)
    manual_out0 = tf.reshape(manual_out0, tf.concat(
        [tf.shape(manual_out0)[:-1], [num_heads, output_size // num_heads]], axis=0))
    manual_out1 = tf.reshape(manual_out1, tf.concat(
        [tf.shape(manual_out1)[:-1], [num_heads, output_size1 // num_heads]], axis=0))
    assert numpy.sum((manual_out0.numpy() - layer_out0.numpy()) ** 2) < 1e-9
    assert numpy.sum((manual_out1.numpy() - layer_out1.numpy()) ** 2) < 1e-9


def test_position_embedding():
    embedding_layer = WordEmbeddingSharedWeights(
        embedding_dim=5, vocab_size=10,
        share_softmax_weights=False)
    embedding_layer = PositionEmbeddingWrapper(
        timing="sinusoids",
        embedding_layer=embedding_layer,)
    inputs1d = tf.convert_to_tensor([4, 7, 8], tf.int32)
    inputs2d = tf.convert_to_tensor([[3, 1, 1, 1], [8, 1, 6, 4], [6, 6, 0, 5]], tf.int32)
    _ = embedding_layer(inputs2d)
    assert len(embedding_layer.get_weights()) == 1
    assert "emb/weights" in embedding_layer.trainable_weights[0].name
    embedding_layer.set_weights([numpy.array(
        [[-0.22683287, 0.20732224, -0.10953838, 0.15318757, -0.07203472],
         [0.48726183, 0.53683335, 0.38046378, -0.42776877, 0.51263684],
         [-0.20618078, 0.43871957, 0.26764846, 0.57276505, -0.13321346],
         [0.34847826, 0.1998071, 0.48136407, -0.03138721, -0.5397158],
         [-0.31466845, 0.24504018, 0.38156456, -0.03245735, 0.28105468],
         [-0.4769836, -0.2763745, -0.35024986, 0.5304734, -0.2523746],
         [0.13987714, -0.36480358, 0.5633767, 0.04371119, -0.5429846],
         [0.07482189, 0.4224295, 0.5645891, -0.12718052, 0.3637674],
         [0.4379062, 0.11231863, -0.6134181, -0.53932106, -0.5402442],
         [-0.18054467, -0.21964127, -0.14727849, 0.61441237, -0.13402274]])])

    emb_for_2d = embedding_layer(inputs2d)
    emb_for_1d = embedding_layer(inputs1d, time=3)
    assert numpy.sum((emb_for_2d.numpy() - numpy.array(
        [[[0.77922106, 0.4467823, 2.0763628, 0.92981607,
           -1.2068413],
          [1.9310216, 1.2004958, 1.3910451, 0.04347992,
           1.1462909],
          [1.998848, 1.2005959, 0.43459606, 0.04347992,
           1.1462909],
          [1.2306706, 1.2006959, -0.13924962, 0.04347986,
           1.1462909]],

         [[0.9791881, 0.2511521, -0.37164462, -0.2059586,
           -1.2080228],
          [1.9310216, 1.2004958, 1.3910451, 0.04347992,
           1.1462909],
          [1.2220722, -0.81552565, 0.8436019, 1.0977412,
           -1.2141505],
          [-0.56250006, 0.5482265, -0.13678819, 0.9274231,
           0.62845737]],

         [[0.3127748, -0.8157256, 2.2597487, 1.0977412,
           -1.2141505],
          [1.1542457, -0.8156256, 1.800051, 1.0977412,
           -1.2141505],
          [0.4020837, 0.46378663, -0.6610821, 1.3425379,
           -0.16107452],
          [-0.92544776, -0.6176922, -1.773175, 2.1861746,
           -0.56432676]]])) ** 2) < 1e-9
    assert numpy.sum((emb_for_1d.numpy() - numpy.array(
        [[-0.56250006, 0.5482265, -0.13678819, 0.9274231, 0.62845737],
         [0.30842686, 0.9448811, 0.27246714, 0.71561563, 0.8134086],
         [1.120308, 0.2514521, -2.361637, -0.20595866, -1.2080228]])) ** 2) < 1e-9

    emb_shared_layer = WordEmbeddingSharedWeights(
        embedding_dim=5, vocab_size=10,
        share_softmax_weights=True)
    emb_shared_layer = PositionEmbeddingWrapper(
        timing="emb",
        embedding_layer=emb_shared_layer)
    emb_for_2d = emb_shared_layer(inputs2d)
    logits_for_2d = emb_shared_layer(emb_for_2d, mode="linear")
    assert len(emb_shared_layer.get_weights()) == 3
    for w in emb_shared_layer.trainable_weights:
        if "shared/weights" in w.name:
            weights = w
        elif "shared/bias" in w.name:
            bias = w
    assert numpy.sum((numpy.dot(emb_for_2d.numpy(), numpy.transpose(weights.numpy()))
                      + bias.numpy() - logits_for_2d.numpy()) ** 2) < 1e-9


if __name__ == "__main__":
    test_ffn()
    test_prepost()
    test_multihead_dense()
    test_position_embedding()
