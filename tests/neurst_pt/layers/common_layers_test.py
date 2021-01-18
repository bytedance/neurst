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
import torch

from neurst.layers.common_layers import (MultiHeadDenseLayer, PositionEmbeddingWrapper, PrePostProcessingWrapper,
                                         TransformerFFN)
from neurst.layers.modalities.text_modalities import WordEmbeddingSharedWeights
from neurst.utils.misc import assert_equal_numpy
from neurst_pt.layers.common_layers import MultiHeadDenseLayer as PTMultiHeadDenseLayer
from neurst_pt.layers.common_layers import PositionEmbeddingWrapper as PTPositionEmbeddingWrapper
from neurst_pt.layers.common_layers import PrePostProcessingWrapper as PTPrePostProcessingWrapper
from neurst_pt.layers.common_layers import TransformerFFN as PTTransformerFFN
from neurst_pt.layers.modalities.text_modalities import WordEmbeddingSharedWeights as PTWordEmbeddingSharedWeights


def test_ffn():
    numpy_inp = numpy.random.rand(3, 5)
    tf_inp = tf.convert_to_tensor(numpy_inp, tf.float32)
    pt_inp = torch.FloatTensor(numpy_inp)
    tf_ffn = TransformerFFN(7, 11, 0.1)
    tf_out = tf_ffn(tf_inp, is_training=False)
    pt_ffn = PTTransformerFFN(5, 7, 11, 0.1)
    _ = pt_ffn(pt_inp, is_training=False)
    pt_ffn._dense1.weight.data = torch.FloatTensor(tf_ffn._conv1.kernel.numpy().transpose([1, 0]))
    pt_ffn._dense1.bias.data = torch.FloatTensor(tf_ffn._conv1.bias.numpy())
    pt_ffn._dense2.weight.data = torch.FloatTensor(tf_ffn._conv2.kernel.numpy().transpose([1, 0]))
    pt_ffn._dense2.bias.data = torch.FloatTensor(tf_ffn._conv2.bias.numpy())
    pt_out = pt_ffn(pt_inp, is_training=False)
    assert_equal_numpy(pt_out.detach().numpy(), tf_out.numpy())


def test_prepost():
    def layer(x, *args, **kwargs):
        _ = args
        _ = kwargs
        return x

    tf_prepost_layer = PrePostProcessingWrapper(
        layer, dropout_rate=0.1, name="lpp")
    pt_prepost_layer = PTPrePostProcessingWrapper(
        layer, norm_shape=3, dropout_rate=0.1)
    numpy_inp = numpy.array([[1, 2, 3.]])
    tf_inp = tf.convert_to_tensor(numpy_inp, tf.float32)
    pt_inp = torch.FloatTensor(numpy_inp)
    tf_out = tf_prepost_layer(tf_inp, is_training=False)
    _ = pt_prepost_layer(pt_inp, is_training=False)
    pt_prepost_layer._norm_layer.weight.data = torch.FloatTensor(tf_prepost_layer._norm_layer.gamma.numpy())
    pt_prepost_layer._norm_layer.bias.data = torch.FloatTensor(tf_prepost_layer._norm_layer.beta.numpy())
    assert_equal_numpy(tf_out.numpy(), pt_prepost_layer(pt_inp, is_training=False).detach().numpy())


def test_multihead_dense():
    num_heads = 3
    output_size = (6, 12)
    input_size = 6
    numpy_inp = numpy.random.randn(2, 3, input_size)
    pt_inp = torch.FloatTensor(numpy_inp)
    tf_inp = tf.convert_to_tensor(numpy_inp, dtype=tf.float32)
    tf_non_out_layer = MultiHeadDenseLayer(
        output_size, num_heads, use_bias=True, is_output_transform=False,
        name="nonoutput_transform")
    pt_non_out_layer = PTMultiHeadDenseLayer(
        input_size, output_size, num_heads, use_bias=True, is_output_transform=False)
    _ = pt_non_out_layer(pt_inp)
    tf_out = tf_non_out_layer(tf_inp)
    pt_non_out_layer._kernel.data = torch.FloatTensor(tf_non_out_layer._kernel.numpy())
    pt_non_out_layer._bias.data = torch.FloatTensor(tf_non_out_layer._bias.numpy())

    for x, y in zip(tf_out, pt_non_out_layer(pt_inp)):
        assert_equal_numpy(x.numpy(), y.detach().numpy())

    num_inputs_per_head = 5
    output_size = 6
    numpy_inp = numpy.random.randn(1, 2, num_heads, num_inputs_per_head)
    tf_inp = tf.convert_to_tensor(numpy_inp)
    pt_inp = torch.FloatTensor(numpy_inp)
    tf_out_layer = MultiHeadDenseLayer(
        output_size, num_heads, use_bias=True, is_output_transform=True,
        name="output_transform")
    pt_out_layer = PTMultiHeadDenseLayer(
        num_heads * num_inputs_per_head, output_size, num_heads,
        use_bias=True, is_output_transform=True)
    tf_out = tf_out_layer(tf_inp)
    _ = pt_out_layer(pt_inp)
    pt_out_layer._kernel.data = torch.FloatTensor(tf_out_layer._kernel.numpy())
    pt_out_layer._bias.data = torch.FloatTensor(tf_out_layer._bias.numpy())
    assert_equal_numpy(tf_out.numpy(), pt_out_layer(pt_inp).detach().numpy())


def test_position_embedding():
    tf_postbl = PositionEmbeddingWrapper.add_sinusoids_timing_signal(tf.zeros([1, 10, 10]), None)
    pt_postbl = PTPositionEmbeddingWrapper.add_sinusoids_timing_signal(torch.zeros(1, 10, 10), None)
    assert_equal_numpy(tf_postbl.numpy(), pt_postbl.detach().numpy())
    emb_dim = 5
    vocab_size = 10
    tf_emb = WordEmbeddingSharedWeights(emb_dim, vocab_size, False)
    pt_emb = PTWordEmbeddingSharedWeights(emb_dim, vocab_size, False)
    inp_2d = numpy.random.randint(0, 9, [2, 5])
    inp_1d = numpy.random.randint(0, 9, [3, ])
    logits_2d = numpy.random.rand(2, 5)
    logits_3d = numpy.random.rand(2, 4, 5)
    tf_inp_2d = tf.convert_to_tensor(inp_2d, tf.int32)
    tf_inp_1d = tf.convert_to_tensor(inp_1d, tf.int32)
    tf_logits_2d = tf.convert_to_tensor(logits_2d, tf.float32)
    tf_logits_3d = tf.convert_to_tensor(logits_3d, tf.float32)
    pt_inp_2d = torch.IntTensor(inp_2d)
    pt_inp_1d = torch.IntTensor(inp_1d)
    pt_logits_2d = torch.FloatTensor(logits_2d)
    pt_logits_3d = torch.FloatTensor(logits_3d)
    _ = tf_emb(tf_logits_2d, mode="linear")
    _ = pt_emb(pt_logits_2d, mode="linear")
    pt_emb._shared_weights.data = torch.Tensor(tf_emb._shared_weights.numpy())
    tf_posemb = PositionEmbeddingWrapper("sinusoids", tf_emb)
    pt_posemb = PTPositionEmbeddingWrapper("sinusoids", pt_emb)
    assert_equal_numpy(tf_posemb(tf_logits_2d, mode="linear").numpy(),
                       pt_posemb(pt_logits_2d, mode="linear").detach().numpy())
    assert_equal_numpy(tf_posemb(tf_logits_3d, mode="linear").numpy(),
                       pt_posemb(pt_logits_3d, mode="linear").detach().numpy())
    assert_equal_numpy(tf_posemb(tf_inp_2d).numpy(), pt_posemb(pt_inp_2d).detach().numpy())
    assert_equal_numpy(tf_posemb(tf_inp_1d, time=5).numpy(), pt_posemb(pt_inp_1d, time=5).detach().numpy())


if __name__ == "__main__":
    test_ffn()
    test_multihead_dense()
    test_prepost()
    test_position_embedding()
