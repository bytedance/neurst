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

from neurst.layers.modalities.text_modalities import WordEmbeddingSharedWeights
from neurst.utils.misc import assert_equal_numpy
from neurst_pt.layers.modalities.text_modalities import WordEmbeddingSharedWeights as PTWordEmbeddingSharedWeights


def test_emb():
    emb_dim = 5
    vocab_size = 10
    tf_emb = WordEmbeddingSharedWeights(emb_dim, vocab_size, True)
    pt_emb = PTWordEmbeddingSharedWeights(emb_dim, vocab_size, True)
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
    pt_emb._bias.data = torch.Tensor(tf_emb._bias.numpy())
    assert_equal_numpy(tf_emb(tf_logits_2d, mode="linear").numpy(),
                       pt_emb(pt_logits_2d, mode="linear").detach().numpy())
    assert_equal_numpy(tf_emb(tf_logits_3d, mode="linear").numpy(),
                       pt_emb(pt_logits_3d, mode="linear").detach().numpy())
    assert_equal_numpy(tf_emb(tf_inp_2d).numpy(), pt_emb(pt_inp_2d).detach().numpy())
    assert_equal_numpy(tf_emb(tf_inp_1d).numpy(), pt_emb(pt_inp_1d).detach().numpy())


if __name__ == "__main__":
    test_emb()
