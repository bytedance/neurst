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

from neurst.layers.modalities.audio_modalities import AudioConv2dSubsamplingLayer as TFAudioConvSubsamplingLayer
from neurst.utils.misc import assert_equal_numpy
from neurst_pt.layers.modalities.audio_modalities import AudioConvSubsamplingLayer


def test_subsampler():
    inp = numpy.random.rand(1, 19, 80, 1)
    pt_inp = torch.FloatTensor(inp)
    tf_inp = tf.convert_to_tensor(inp, tf.float32)
    # with layer norm
    tf_layer = TFAudioConvSubsamplingLayer(40)
    pt_layer = AudioConvSubsamplingLayer(40, input_dimension=80)
    _ = tf_layer(tf_inp)
    _ = pt_layer(pt_inp)
    pt_layer._conv_layer1.weight.data = torch.FloatTensor(
        tf_layer._conv_layers[0].kernel.numpy().transpose((3, 2, 0, 1)))
    pt_layer._conv_layer1.bias.data = torch.FloatTensor(tf_layer._conv_layers[0].bias.numpy())
    pt_layer._conv_layer2.weight.data = torch.FloatTensor(
        tf_layer._conv_layers[1].kernel.numpy().transpose((3, 2, 0, 1)))
    pt_layer._conv_layer2.bias.data = torch.FloatTensor(tf_layer._conv_layers[1].bias.numpy())
    pt_layer._norm_layer1.weight.data = torch.FloatTensor(tf_layer._norm_layers[0].gamma.numpy())
    pt_layer._norm_layer1.bias.data = torch.FloatTensor(tf_layer._norm_layers[0].beta.numpy())
    pt_layer._norm_layer2.weight.data = torch.FloatTensor(tf_layer._norm_layers[1].gamma.numpy())
    pt_layer._norm_layer2.bias.data = torch.FloatTensor(tf_layer._norm_layers[1].beta.numpy())
    pt_layer._dense_layer.weight.data = torch.FloatTensor(tf_layer._dense_layer.kernel.numpy().transpose())
    pt_layer._dense_layer.bias.data = torch.FloatTensor(tf_layer._dense_layer.bias.numpy())
    assert_equal_numpy(pt_layer(pt_inp).detach().numpy(), tf_layer(tf_inp).numpy(), 5e-5)

    # without layer norm
    tf_layer = TFAudioConvSubsamplingLayer(40, layer_norm=False)
    pt_layer = AudioConvSubsamplingLayer(40, input_dimension=80, layer_norm=False)
    _ = tf_layer(tf_inp)
    _ = pt_layer(pt_inp)
    pt_layer._conv_layer1.weight.data = torch.FloatTensor(
        tf_layer._conv_layers[0].kernel.numpy().transpose((3, 2, 0, 1)))
    pt_layer._conv_layer1.bias.data = torch.FloatTensor(tf_layer._conv_layers[0].bias.numpy())
    pt_layer._conv_layer2.weight.data = torch.FloatTensor(
        tf_layer._conv_layers[1].kernel.numpy().transpose((3, 2, 0, 1)))
    pt_layer._conv_layer2.bias.data = torch.FloatTensor(tf_layer._conv_layers[1].bias.numpy())
    pt_layer._dense_layer.weight.data = torch.FloatTensor(tf_layer._dense_layer.kernel.numpy().transpose())
    pt_layer._dense_layer.bias.data = torch.FloatTensor(tf_layer._dense_layer.bias.numpy())
    assert_equal_numpy(pt_layer(pt_inp).detach().numpy(), tf_layer(tf_inp).numpy(), 1e-6)


if __name__ == "__main__":
    test_subsampler()
