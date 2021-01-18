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
import math

import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

from neurst_pt.utils.activations import get_activation

has_fused_layernorm = True

try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    class FusedLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)

except ImportError:
    has_fused_layernorm = False


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    if torch.jit.is_scripting():
        export = True
    if not export and torch.cuda.is_available() and has_fused_layernorm:
        return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class PrePostProcessingWrapper(nn.Module):
    """ Custom prepost processing for transformer.

    The sequence is specified as a string which may contain the
    following characters:
      a: add previous_x
      n: apply normalization
      d: apply drop'out

    This class only defines the "n" - layer - "da" mode.
    """

    def __init__(self, layer, norm_shape, dropout_rate=0.1, epsilon=1e-12, pre_norm=True):
        """ Initializes.

        Args:
            layer: The layer.
            norm_shape: The normalized shape.
            dropout_rate: The dropout rate.
            epsilon: The epsilon of layer norm.
            pre_norm: Applies norm->layer->dropout->residual if True, else
                layer->dropout->residual->norm
        """
        super(PrePostProcessingWrapper, self).__init__()
        self._dropout_rate = dropout_rate
        self._epsilon = epsilon
        self._layer = layer
        self._norm_layer = LayerNorm(norm_shape, epsilon)
        self._pre_norm = pre_norm

    def forward(self, inputs, *args, **kwargs):
        is_training = kwargs["is_training"]
        if self._pre_norm:
            # n
            y = self._norm_layer(inputs)
            # layer: self att / ffn
            y = self._layer(y, *args, **kwargs)
            # d
            y = F.dropout(y, p=self._dropout_rate, training=is_training)
            # a
            return inputs + y
        else:
            y = self._layer(inputs, *args, **kwargs)
            # d
            y = F.dropout(y, p=self._dropout_rate, training=is_training)
            # an
            return self._norm_layer(inputs + y)


class TransformerFFN(nn.Module):
    """ Applies the position-wise feed-forward as described
    in https://arxiv.org/abs/1706.03762 """

    def __init__(self,
                 input_size,
                 filter_size,
                 output_size,
                 dropout_rate,
                 activation="relu"):
        """ Initializes Transformer FFN.

        Args:
            input_size: The dimension of the input tensor.
            filter_size: The hidden size of the relu layer.
            output_size: The output size.
            dropout_rate: The dropout rate.
            activation: The activation of internal layer.
        """
        super(TransformerFFN, self).__init__()
        self._dropout_rate = dropout_rate
        self._dense1 = nn.Linear(input_size, filter_size, bias=True)
        self._dense2 = nn.Linear(filter_size, output_size, bias=True)
        self._activation_fn = get_activation(activation)

    def forward(self, inputs, is_training=False):
        """ Returns the output of TransformerFFN.

        Args:
            inputs: A tensor with shape [batch_size, length, num_units].
            is_training: A boolean scalar, whether in training mode or not.

        Returns:
            Output of the feedforward network.
            tensor with shape [batch_size, length, output_size]
        """
        output = self._activation_fn(self._dense1(inputs))
        output = F.dropout(output, p=self._dropout_rate, training=is_training)
        output = self._dense2(output)
        return output


class MultiHeadDenseLayer(nn.Module):
    """ Auto splitting or combining heads for the linear transformation. """

    def __init__(self,
                 input_size,
                 output_units,
                 num_heads,
                 activation=None,
                 use_bias=True,
                 is_output_transform=False):
        """ Initializes MultiHeadDenseLayer.

        Args:
            input_size: The input dimension.
            output_units: A int scalar or int list, indicating the transformed output units.
                It must be a int scalar when `is_output_transform` is True.
            num_heads: The head num.
            activation: A string or a callable function for activation.
            use_bias: A boolean, whether to add bias tensor.
            is_output_transform: A boolean, whether to use this layer for the output
                transformation in multi head attention.
        """
        super(MultiHeadDenseLayer, self).__init__()

        self._output_units = output_units
        self._num_heads = num_heads
        self._use_bias = use_bias
        self._is_output_transform = is_output_transform
        self._activation = activation
        self._activation_fn = get_activation(activation)
        # compatible
        self._flatten_output_units = tf.nest.flatten(self._output_units)
        if is_output_transform:
            assert not tf.nest.is_nested(self._output_units)
            self._kernel = torch.nn.Parameter(torch.nn.init.xavier_normal_(
                torch.empty(input_size, self._output_units)))
        else:
            self._kernel = torch.nn.Parameter(torch.nn.init.xavier_normal_(
                torch.empty(input_size, sum(self._flatten_output_units))), requires_grad=True)
        if self._use_bias:
            self._bias = torch.nn.Parameter(torch.zeros(sum(self._flatten_output_units)), requires_grad=True)

    def compat_kernel_shape(self, input_shape):
        """ Compatible kernel for variable storage. """
        if self._is_output_transform:
            return [input_shape[-1] * input_shape[-2], self._output_units]
        return [input_shape[-1], sum(self._flatten_output_units)]

    @property
    def kernel_shape(self):
        """ The kernel shape. """
        if self._is_output_transform:
            return [self._num_heads, -1, self._output_units]
        return [-1, sum(self._flatten_output_units)]

    def forward(self, inputs):
        """ Implements ``call()`` for MultiHeadDenseLayer.

        Args:
            inputs: A float tensor of shape [batch_size, length, hidden_size]
                when output_projection is False, otherwise a float tensor of shape
                [batch_size, length, num_heads, num_units_per_head].

        Returns:
            The projected tensor with shape [batch_size, length, num_heads,
                num_units_per_head] per `self._output_units` when output_projection
                is False, otherwise [batch_size, length, output_units].
        """
        kernel = torch.reshape(self._kernel, self.kernel_shape)
        if self._is_output_transform:
            # a: batch
            # b: length
            # c: num heads
            # d: input units per head
            # e: num_output
            output = torch.einsum("abcd,cde->abe", inputs, kernel)
        else:
            # a: batch
            # b: length
            # c: input size
            # d: total output size
            output = torch.einsum("abc,cd->abd", inputs, kernel)
        if self._use_bias:
            output += self._bias

        if not self._is_output_transform:
            output = torch.split(
                output, self._flatten_output_units, dim=-1)
            output = tf.nest.map_structure(
                lambda x, num_units: torch.reshape(
                    x, list(x.size())[:-1] + [self._num_heads, num_units // self._num_heads]),
                output, self._flatten_output_units, check_types=False)
        output = tf.nest.flatten(output)
        if self._activation_fn is not None:
            output = tf.nest.map_structure(self._activation_fn, output, check_types=False)
        return tf.nest.pack_sequence_as(self._output_units, output)


class PositionEmbeddingWrapper(nn.Module):

    def __init__(self,
                 timing,
                 embedding_layer,
                 max_positions=512,
                 sinusoids_as_variable=False):
        """ Initializes the position embedding layer.

        Args:
            timing: The position embedding type. Now only 'sinusoids'
                and 'emb' are supported.
            embedding_layer: The embedding layer.
            max_positions: The maximum positions.
            sinusoids_as_variable: Whether the sinusoids position embedding
                is pre-calculated and fixed.
        """
        super(PositionEmbeddingWrapper, self).__init__()
        self._timing = timing
        self._embedding_layer = embedding_layer
        self._embedding_dim = embedding_layer.embedding_dim
        self._max_positions = max_positions
        self._sinusoids_as_variable = sinusoids_as_variable
        assert self._timing in [None, "sinusoids", "emb"], (
            "Unknown position embedding type: \"{}\"".format(timing))
        if self._timing == "emb":
            self._position_emb_table = nn.Parameter(nn.init.normal_(
                torch.empty(self._max_positions, self._embedding_dim),
                mean=0., std=self._embedding_dim ** -0.5), requires_grad=True)
        elif self._timing == "sinusoids" and self._sinusoids_as_variable:
            self._position_emb_table = nn.Parameter(
                self.add_sinusoids_timing_signal(
                    torch.zeros(1, self._max_positions, self._embedding_dim), None),
                requires_grad=False)

    @property
    def embedding_layer(self):
        return self._embedding_layer

    @staticmethod
    def add_sinusoids_timing_signal(x, time, min_timescale=1.0, max_timescale=1.0e4):
        """Adds a bunch of sinusoids of different frequencies to a Tensor.

        Each channel of the input Tensor is incremented by a sinusoid of a different
        frequency and phase.

        This allows attention to learn to use absolute and relative positions.
        Timing signals should be added to some precursors of both the query and the
        memory inputs to attention.

        The use of relative position is possible because sin(x+y) and cos(x+y) can be
        experessed in terms of y, sin(x) and cos(x).

        In particular, we use a geometric sequence of timescales starting with
        min_timescale and ending with max_timescale.  The number of different
        timescales is equal to channels / 2. For each timescale, we
        generate the two sinusoidal signals sin(timestep/timescale) and
        cos(timestep/timescale).  All of these sinusoids are concatenated in
        the channels dimension.

        This function is originally copied from Google's Tensor2Tensor code
        and modified to hold the capability for add timing signal at the
        specific time.

        Args:
          x: a Tensor with shape [batch, length, channels]
          min_timescale: a float
          max_timescale: a float

        Returns: A Tensor the same shape as x.
        """
        dtype = x.dtype
        channels = list(x.size())[-1]
        if x.ndim == 3:  # [batch_size, timesteps, dim]
            length = x.size()[1]
            position = torch.arange(0, length).type(dtype)
        elif x.ndim == 2:  # [batch_size, dim]
            length = 1
            position = torch.arange(time, time + 1).type(dtype)
        else:
            raise ValueError("need a Tensor with rank 2 or 3")
        num_timescales = channels // 2
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale))
            / (num_timescales - 1.))
        inv_timescales = min_timescale * torch.exp(
            torch.arange(0, num_timescales).type(dtype) * -log_timescale_increment)
        scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        signal = F.pad(signal, (0, int(math.fmod(channels, 2))))
        if x.ndim == 3:
            signal = torch.reshape(signal, [1, length, channels])
        else:
            signal = torch.reshape(signal, [1, channels])
        return x + signal

    def forward(self, inputs, time=None, **kwargs):
        emb: torch.Tensor = self._embedding_layer(inputs, **kwargs)
        mode = kwargs.get("mode", "embedding")
        if self._timing is None or mode != "embedding":
            return emb
        assert emb.size()[-1] == self._embedding_dim, (
            "The position embedding dimension should match the "
            "embedding dimension: {} vs. {}".format(
                self._embedding_dim, emb.size()[-1]))
        x_ndims = emb.ndim
        if x_ndims == 2 and time is None:
            raise ValueError("\"time\" should be provided when input x has 2-dims")
        if self._timing == "sinusoids":
            emb *= self._embedding_dim ** 0.5
        # TO load from positional embedding from other repos, e.g. fairseq
        #     return self.add_sinusoids_timing_signal(
        #         x=emb, time=time)
        # if self._timing == "emb":
        if self._timing == "sinusoids" and not self._sinusoids_as_variable:
            return self.add_sinusoids_timing_signal(x=emb, time=time)
        if x_ndims == 2:
            position_emb = self._position_emb_table[time.to(torch.long)]
        elif x_ndims == 3:
            position_emb = self._position_emb_table.narrow(0, 0, emb.size()[1])
        else:
            raise ValueError("need a Tensor with rank 2 or 3")
        return emb + position_emb.unsqueeze(0)
