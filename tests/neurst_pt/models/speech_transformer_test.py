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

import numpy
import tensorflow as tf
import torch

from neurst.models import build_model as build_tf_model
from neurst.models.speech_transformer import SpeechTransformer as TFSpeechTransformer
from neurst.utils.hparams_sets import get_hyper_parameters
from neurst.utils.misc import assert_equal_numpy
from neurst_pt.models import build_model
from neurst_pt.models.speech_transformer import SpeechTransformer


def test_st():
    params = copy.deepcopy(get_hyper_parameters("speech_transformer_toy")["model.params"])
    params["modality.source.dim"] = None
    params["modality.target.dim"] = None
    params["modality.source.timing"] = None
    params["modality.target.timing"] = None
    params["encoder.num_layers"] = 1
    params["decoder.num_layers"] = 1

    src_vocab_meta = dict(audio_feature_dim=80, audio_feature_channels=1)
    trg_vocab_meta = dict(vocab_size=5, eos_id=4, bos_id=3, unk_id=2)

    fake_audio = numpy.random.rand(1, 11, 80, 1)
    pt_inps = {
        "src": torch.FloatTensor(fake_audio),
        "src_length": torch.LongTensor([11]),
        "trg_input": torch.LongTensor([[3, 0, 1]]),
    }
    tf_inps = {
        "src": tf.convert_to_tensor(fake_audio, tf.float32),
        "src_length": tf.convert_to_tensor([11], tf.int32),
        "trg_input": tf.convert_to_tensor([[3, 0, 1]], tf.int32),
    }

    pt_model: SpeechTransformer = build_model({"model.class": "speech_transformer", "params": params},
                                              src_meta=src_vocab_meta, trg_meta=trg_vocab_meta)
    tf_model: TFSpeechTransformer = build_tf_model({"model.class": "speech_transformer", "params": params},
                                                   src_meta=src_vocab_meta, trg_meta=trg_vocab_meta)

    pt_model._src_modality.embedding_layer._conv_layer1.weight.data = torch.FloatTensor(
        tf_model._src_modality.embedding_layer._conv_layers[0].kernel.numpy().transpose((3, 2, 0, 1)))
    pt_model._src_modality.embedding_layer._conv_layer1.bias.data = torch.FloatTensor(
        tf_model._src_modality.embedding_layer._conv_layers[0].bias.numpy())
    pt_model._src_modality.embedding_layer._conv_layer2.weight.data = torch.FloatTensor(
        tf_model._src_modality.embedding_layer._conv_layers[1].kernel.numpy().transpose((3, 2, 0, 1)))
    pt_model._src_modality.embedding_layer._conv_layer2.bias.data = torch.FloatTensor(
        tf_model._src_modality.embedding_layer._conv_layers[1].bias.numpy())
    pt_model._src_modality.embedding_layer._norm_layer1.weight.data = torch.FloatTensor(
        tf_model._src_modality.embedding_layer._norm_layers[0].gamma.numpy())
    pt_model._src_modality.embedding_layer._norm_layer1.bias.data = torch.FloatTensor(
        tf_model._src_modality.embedding_layer._norm_layers[0].beta.numpy())
    pt_model._src_modality.embedding_layer._norm_layer2.weight.data = torch.FloatTensor(
        tf_model._src_modality.embedding_layer._norm_layers[1].gamma.numpy())
    pt_model._src_modality.embedding_layer._norm_layer2.bias.data = torch.FloatTensor(
        tf_model._src_modality.embedding_layer._norm_layers[1].beta.numpy())
    pt_model._src_modality.embedding_layer._dense_layer.weight.data = torch.FloatTensor(
        tf_model._src_modality.embedding_layer._dense_layer.kernel.numpy().transpose())
    pt_model._src_modality.embedding_layer._dense_layer.bias.data = torch.FloatTensor(
        tf_model._src_modality.embedding_layer._dense_layer.bias.numpy())
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
    test_st()
