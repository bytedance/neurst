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
import torch
import torch.nn as nn
import torch.nn.functional as F

from neurst.utils.configurable import extract_constructor_params
from neurst_pt.layers.common_layers import LayerNorm


class AudioConvSubsamplingLayer(nn.Module):
    """ Subsampling for audio features. """

    def __init__(self,
                 embedding_dim,
                 input_dimension=80,
                 input_channels=1,
                 channels=256,
                 kernel_size=3,
                 strides=2,
                 layer_norm=True,
                 name=None):
        """ Initializes the layer for subsample the audio feature.

        Args:
            embedding_dim: An int scalar, the embedding dimension.
            input_channels: An int scalar, the number of input channels of the audio feature.
            input_dimension: An int scalar, the dimension of the audio feature.
            channels: The channel size of the convolution layer.
            kernel_size: The kernel size of the convolution layer.
            strides: The stride size of the convolution layer.
            layer_norm: Whether to apply layer normalization.
            verbose: A boolean, whether to logging the parameters.
            name: The name of the layer.
        """
        self._params = extract_constructor_params(locals(), verbose=True)
        super(AudioConvSubsamplingLayer, self).__init__()
        self._input_channels = input_channels
        self._embedding_dim = embedding_dim
        self._channels = channels
        self._kernel_size = kernel_size
        self._layer_norm = layer_norm
        self._strides = strides
        num_pad = self._kernel_size // 2
        self._conv_layer1 = nn.Conv2d(
            self._input_channels, self._channels,
            kernel_size=(self._kernel_size, self._kernel_size),
            stride=(self._strides, self._strides),
            padding=(num_pad, num_pad))
        self._conv_layer2 = nn.Conv2d(
            self._channels, self._channels,
            kernel_size=(self._kernel_size, self._kernel_size),
            stride=(self._strides, self._strides),
            padding=(num_pad, num_pad))
        if self._layer_norm:
            self._norm_layer1 = LayerNorm(self._channels, eps=1e-6)
            self._norm_layer2 = LayerNorm(self._channels, eps=1e-6)
        self._dense_layer = nn.Linear(
            ((input_dimension + self._strides - 1)
             // self._strides + self._strides - 1) // self._strides * self._channels,
            self._embedding_dim, bias=True)

    @property
    def embedding_dim(self):
        return self._embedding_dim

    def forward(self, inputs, **kwargs):
        """ Gets token embeddings or computes logits.

        Args:
            inputs: An float tensor with shape [batch_size, length, feature_dim, channels].

        Returns:
            A float tensor with shape [batch, new_length, new_feature_dim].
        """
        _ = kwargs
        assert inputs.ndim == 4
        inputs = inputs.permute(0, 3, 1, 2)
        conv1 = self._conv_layer1(inputs)
        if self._layer_norm:
            conv1 = self._norm_layer1(conv1.transpose(1, 3)).transpose(1, 3)
        conv1 = F.relu(conv1)
        conv2 = self._conv_layer2(conv1).permute(0, 2, 3, 1)
        if self._layer_norm:
            conv2 = self._norm_layer2(conv2)
        conv2 = F.relu(conv2)
        conv2_reshape = torch.reshape(conv2, conv2.size()[:2] + (-1,))
        output = self._dense_layer(conv2_reshape)
        return output
