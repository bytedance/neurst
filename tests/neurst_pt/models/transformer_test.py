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
import copy

import tensorflow as tf
import torch

from neurst.models import build_model
from neurst.models.transformer import Transformer as TFTransformer
from neurst.utils.hparams_sets import get_hyper_parameters
from neurst.utils.misc import assert_equal_numpy
from neurst_pt.models import build_model as build_pt_model
from neurst_pt.models.transformer import Transformer


def test_seq2seq():
    params = copy.deepcopy(get_hyper_parameters("transformer_toy")["model.params"])
    params["modality.source.dim"] = None
    params["modality.target.dim"] = None
    params["modality.source.timing"] = None
    params["modality.target.timing"] = None
    params["encoder.num_layers"] = 1
    params["decoder.num_layers"] = 1

    src_vocab_meta = dict(vocab_size=8, eos_id=7, bos_id=6, unk_id=5)
    trg_vocab_meta = dict(vocab_size=5, eos_id=4, bos_id=3, unk_id=2)

    pt_inps = {
        "src": torch.LongTensor([[0, 1, 1, 7], [1, 7, 7, 7]]),
        "src_padding": torch.FloatTensor([[0, 0, 0, 0.], [0, 0, 1, 1.]]),
        "trg_input": torch.LongTensor([[3, 0, 1], [3, 2, 4]]),
        "trg": torch.LongTensor([[0, 1, 4], [2, 4, 4]]),
        "trg_padding": torch.FloatTensor([[0, 0, 0.], [0, 0, 1.]]),
    }
    tf_inps = {
        "src": tf.convert_to_tensor(
            [[0, 1, 1, 7], [1, 7, 7, 7]], tf.int64),
        "src_padding": tf.convert_to_tensor([[0, 0, 0, 0.], [0, 0, 1, 1.]], tf.float32),
        "trg_input": tf.convert_to_tensor([[3, 0, 1], [3, 2, 4]], tf.int32),
        "trg": tf.convert_to_tensor([[0, 1, 4], [2, 4, 4]], tf.int32),
        "trg_padding": tf.convert_to_tensor([[0, 0, 0.], [0, 0, 1.]], tf.float32),
    }

    pt_model: Transformer = build_pt_model({"model.class": "transformer", "params": params},
                                           src_meta=src_vocab_meta, trg_meta=trg_vocab_meta)
    tf_model: TFTransformer = build_model({"model.class": "transformer", "params": params},
                                          src_meta=src_vocab_meta, trg_meta=trg_vocab_meta)
    pt_model._src_modality.embedding_layer._shared_weights.data = torch.FloatTensor(
        tf_model._src_modality.embedding_layer._shared_weights.numpy())
    pt_model._trg_modality.embedding_layer._shared_weights.data = torch.FloatTensor(
        tf_model._trg_modality.embedding_layer._shared_weights.numpy())
    pt_model._trg_modality.embedding_layer._bias.data = torch.FloatTensor(
        tf_model._trg_modality.embedding_layer._bias.numpy())
    pt_model._encoder._output_norm_layer.weight.data = torch.FloatTensor(
        tf_model._encoder._output_norm_layer.gamma.numpy())
    pt_model._encoder._output_norm_layer.bias.data = torch.FloatTensor(
        tf_model._encoder._output_norm_layer.beta.numpy())
    pt_model._encoder._stacking_layers[0][0]._layer._qkv_transform_layer._kernel.data = torch.FloatTensor(
        tf_model._encoder._stacking_layers[0]._selfatt_layer._layer._qkv_transform_layer._kernel.numpy())
    pt_model._encoder._stacking_layers[0][0]._layer._qkv_transform_layer._bias.data = torch.FloatTensor(
        tf_model._encoder._stacking_layers[0]._selfatt_layer._layer._qkv_transform_layer._bias.numpy())
    pt_model._encoder._stacking_layers[0][0]._layer._output_transform_layer._kernel.data = torch.FloatTensor(
        tf_model._encoder._stacking_layers[0]._selfatt_layer._layer._output_transform_layer._kernel.numpy())
    pt_model._encoder._stacking_layers[0][0]._layer._output_transform_layer._bias.data = torch.FloatTensor(
        tf_model._encoder._stacking_layers[0]._selfatt_layer._layer._output_transform_layer._bias.numpy())
    pt_model._encoder._stacking_layers[0][1]._layer._dense1.weight.data = torch.FloatTensor(
        tf_model._encoder._stacking_layers[0]._ffn_layer._layer._conv1.kernel.numpy().transpose([1, 0]))
    pt_model._encoder._stacking_layers[0][1]._layer._dense1.bias.data = torch.FloatTensor(
        tf_model._encoder._stacking_layers[0]._ffn_layer._layer._conv1.bias.numpy())
    pt_model._encoder._stacking_layers[0][1]._layer._dense2.weight.data = torch.FloatTensor(
        tf_model._encoder._stacking_layers[0]._ffn_layer._layer._conv2.kernel.numpy().transpose([1, 0]))
    pt_model._encoder._stacking_layers[0][1]._layer._dense2.bias.data = torch.FloatTensor(
        tf_model._encoder._stacking_layers[0]._ffn_layer._layer._conv2.bias.numpy())
    pt_model._encoder._stacking_layers[0][0]._norm_layer.weight.data = torch.FloatTensor(
        tf_model._encoder._stacking_layers[0]._selfatt_layer._norm_layer.gamma.numpy())
    pt_model._encoder._stacking_layers[0][0]._norm_layer.bias.data = torch.FloatTensor(
        tf_model._encoder._stacking_layers[0]._selfatt_layer._norm_layer.beta.numpy())
    pt_model._encoder._stacking_layers[0][1]._norm_layer.weight.data = torch.FloatTensor(
        tf_model._encoder._stacking_layers[0]._ffn_layer._norm_layer.gamma.numpy())
    pt_model._encoder._stacking_layers[0][1]._norm_layer.bias.data = torch.FloatTensor(
        tf_model._encoder._stacking_layers[0]._ffn_layer._norm_layer.beta.numpy())
    pt_model._decoder._output_norm_layer.weight.data = torch.FloatTensor(
        tf_model._decoder._output_norm_layer.gamma.numpy())
    pt_model._decoder._output_norm_layer.bias.data = torch.FloatTensor(
        tf_model._decoder._output_norm_layer.beta.numpy())
    pt_model._decoder._stacking_layers[0][0]._layer._qkv_transform_layer._kernel.data = torch.FloatTensor(
        tf_model._decoder._stacking_layers[0]._selfatt_layer._layer._qkv_transform_layer._kernel.numpy())
    pt_model._decoder._stacking_layers[0][0]._layer._qkv_transform_layer._bias.data = torch.FloatTensor(
        tf_model._decoder._stacking_layers[0]._selfatt_layer._layer._qkv_transform_layer._bias.numpy())
    pt_model._decoder._stacking_layers[0][0]._layer._output_transform_layer._kernel.data = torch.FloatTensor(
        tf_model._decoder._stacking_layers[0]._selfatt_layer._layer._output_transform_layer._kernel.numpy())
    pt_model._decoder._stacking_layers[0][0]._layer._output_transform_layer._bias.data = torch.FloatTensor(
        tf_model._decoder._stacking_layers[0]._selfatt_layer._layer._output_transform_layer._bias.numpy())
    pt_model._decoder._stacking_layers[0][0]._norm_layer.weight.data = torch.FloatTensor(
        tf_model._decoder._stacking_layers[0]._selfatt_layer._norm_layer.gamma.numpy())
    pt_model._decoder._stacking_layers[0][0]._norm_layer.bias.data = torch.FloatTensor(
        tf_model._decoder._stacking_layers[0]._selfatt_layer._norm_layer.beta.numpy())
    pt_model._decoder._stacking_layers[0][1]._layer._q_transform_layer._kernel.data = torch.FloatTensor(
        tf_model._decoder._stacking_layers[0]._crossatt_layer._layer._q_transform_layer._kernel.numpy())
    pt_model._decoder._stacking_layers[0][1]._layer._q_transform_layer._bias.data = torch.FloatTensor(
        tf_model._decoder._stacking_layers[0]._crossatt_layer._layer._q_transform_layer._bias.numpy())
    pt_model._decoder._stacking_layers[0][1]._layer._kv_transform_layer._kernel.data = torch.FloatTensor(
        tf_model._decoder._stacking_layers[0]._crossatt_layer._layer._kv_transform_layer._kernel.numpy())
    pt_model._decoder._stacking_layers[0][1]._layer._kv_transform_layer._bias.data = torch.FloatTensor(
        tf_model._decoder._stacking_layers[0]._crossatt_layer._layer._kv_transform_layer._bias.numpy())
    pt_model._decoder._stacking_layers[0][1]._layer._output_transform_layer._kernel.data = torch.FloatTensor(
        tf_model._decoder._stacking_layers[0]._crossatt_layer._layer._output_transform_layer._kernel.numpy())
    pt_model._decoder._stacking_layers[0][1]._layer._output_transform_layer._bias.data = torch.FloatTensor(
        tf_model._decoder._stacking_layers[0]._crossatt_layer._layer._output_transform_layer._bias.numpy())
    pt_model._decoder._stacking_layers[0][1]._norm_layer.weight.data = torch.FloatTensor(
        tf_model._decoder._stacking_layers[0]._crossatt_layer._norm_layer.gamma.numpy())
    pt_model._decoder._stacking_layers[0][1]._norm_layer.bias.data = torch.FloatTensor(
        tf_model._decoder._stacking_layers[0]._crossatt_layer._norm_layer.beta.numpy())
    pt_model._decoder._stacking_layers[0][2]._layer._dense1.weight.data = torch.FloatTensor(
        tf_model._decoder._stacking_layers[0]._ffn_layer._layer._conv1.kernel.numpy().transpose([1, 0]))
    pt_model._decoder._stacking_layers[0][2]._layer._dense1.bias.data = torch.FloatTensor(
        tf_model._decoder._stacking_layers[0]._ffn_layer._layer._conv1.bias.numpy())
    pt_model._decoder._stacking_layers[0][2]._layer._dense2.weight.data = torch.FloatTensor(
        tf_model._decoder._stacking_layers[0]._ffn_layer._layer._conv2.kernel.numpy().transpose([1, 0]))
    pt_model._decoder._stacking_layers[0][2]._layer._dense2.bias.data = torch.FloatTensor(
        tf_model._decoder._stacking_layers[0]._ffn_layer._layer._conv2.bias.numpy())
    pt_model._decoder._stacking_layers[0][2]._norm_layer.weight.data = torch.FloatTensor(
        tf_model._decoder._stacking_layers[0]._ffn_layer._norm_layer.gamma.numpy())
    pt_model._decoder._stacking_layers[0][2]._norm_layer.bias.data = torch.FloatTensor(
        tf_model._decoder._stacking_layers[0]._ffn_layer._norm_layer.beta.numpy())
    assert_equal_numpy(tf_model(tf_inps, is_training=False).numpy(),
                       pt_model(pt_inps, is_training=False).detach().numpy(), 5e-6)


if __name__ == "__main__":
    test_seq2seq()
