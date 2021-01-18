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
from neurst_pt.layers.common_layers import MultiHeadDenseLayer


class MultiHeadAttention(nn.Module):
    """ Class of multi-head scaled-dot-product attention with input/output
        transformations. """

    def __init__(self,
                 input_depth,
                 num_heads,
                 num_units,
                 attention_key_depth=None,
                 attention_value_depth=None,
                 output_depth=None,
                 attention_dropout_rate=0.1,
                 attention_type="dot_product"):
        """ Initializes the multi head attention layer.

        Args:
            input_depth: The dimension of the input tensor.
            num_heads: A int scalar, the number of heads.
            num_units: A int scalar, the default units if other `depth` is
                not provided.
            attention_key_depth: A int scalar, the dimension for projected
                attention keys. If not provided, then use `num_units` as default.
            attention_value_depth: A int scalar, the dimension for projected
                attention values. If not provided, then use `num_units` as default.
            output_depth: A int scalar, the dimension for projected
                outputs. If not provided, then use `num_units` as default.
            attention_dropout_rate: A float scalar, the dropout rate for attention weight.
            attention_type: A string indicating the attention type.
        """
        self._params = extract_constructor_params(locals(), verbose=False)
        super(MultiHeadAttention, self).__init__()
        self._input_depth = input_depth
        self._num_heads = num_heads
        self._num_units = num_units
        self._attention_key_depth = attention_key_depth or num_units
        self._attention_value_depth = attention_value_depth or num_units
        self._output_depth = output_depth or num_units
        self._attention_dropout_rate = attention_dropout_rate
        self._attention_type = attention_type
        if self._attention_key_depth % self._num_heads != 0:
            raise ValueError("query depth ({}) must be divisible by the number of "
                             "attention heads ({}).".format(self._attention_key_depth, self._num_heads))
        if self._attention_value_depth % self._num_heads != 0:
            raise ValueError("value depth ({}) must be divisible by the number of "
                             "attention heads ({}).".format(self._attention_value_depth, self._num_heads))
        # pre-create output transform layer
        self._output_transform_layer = MultiHeadDenseLayer(
            input_size=input_depth, output_units=self._output_depth,
            num_heads=self._num_heads, is_output_transform=True,
            use_bias=True)
        self._build_qkv_transform_layer()

    def _build_qkv_transform_layer(self):
        """ Builds the layer.
            Layers for linearly projecting the queries, keys, and values."""
        self._q_transform_layer = MultiHeadDenseLayer(
            input_size=self._input_depth, output_units=self._attention_key_depth,
            num_heads=self._num_heads, is_output_transform=False,
            use_bias=True)
        self._kv_transform_layer = MultiHeadDenseLayer(
            input_size=self._input_depth, is_output_transform=False,
            output_units=[self._attention_key_depth, self._attention_value_depth],
            num_heads=self._num_heads, use_bias=True)

    def compute_qkv(self, query, memory, cache, decode_loop_step=None):
        """ Computes linear transformations of query, keys and values.

        Args:
            query: A tensor with shape [batch_size, length_q, query_depth].
            memory: A tensor with shape [batch_size, length_m, memory_depth].
            cache: A dict, used during prediction.
            decode_loop_step: An integer, step number of the decoding loop. Used only
                for autoregressive inference with static-shape cache.

        Returns: A tuple `(query_transformed, key_transformed, memory_transformed)`.
        """
        _ = cache
        _ = decode_loop_step
        # [batch_size, length_q/k/v, num_heads, num_units_per_head]
        q = self._q_transform_layer(query)
        k, v = self._kv_transform_layer(memory)
        return q, k, v

    def att_fn(self, q, k, bias):
        """ Computes attention weights according to attention_type.

        Args:
            q: Attention query tensor with shape
              [batch_size, length_q, num_heads, att_key_depth / num_heads]
            k:  Attention query tensor with shape
              [batch_size, length_k, num_heads, att_key_depth / num_heads]
            bias: The bias tensor with shape [batch_size, length_q, length_k]
                or [batch_size, 1, length_q, length_k]

        Returns: The attention weight with shape
            [batch_size, num_heads, length_q, length_k]
        """
        if self._attention_type == "dot_product":
            # B: batch_size
            # T: length_k
            # F: length_q
            # N: num_heads
            # H: depth per head
            # logits: [batch_size, num_heads, length_q, length_k]
            logits = torch.einsum("btnh,bfnh->bnft", k, q)

            if bias is not None:
                if bias.ndim == 2:
                    bias = bias.unsqueeze(1).unsqueeze(1)
                elif bias.ndim != 4:
                    raise ValueError("bias tensor with {}-dim is not valid".format(bias.ndim))
                logits += bias
            # Note that softmax internally performs math operations using float32
            # for numeric stability. When training with float16, we keep the input
            # and output in float16 for better performance.
            weights = F.softmax(logits, -1)
        else:
            raise NotImplementedError(
                "att_fn for \"{}\" not implemented.".format(self._attention_type))
        return weights

    def forward(self,
                query,
                memory,
                memory_bias=None,
                cache=None,
                is_training=True,
                decode_loop_step=None):
        """ Apply attention mechanism to query and memory.

        Args:
            query: A tensor with shape [batch_size, length_q, query_depth]
                or [batch_size, query_depth].
            memory: A tensor with shape [batch_size, length_m, memory_depth].
            memory_bias: A tensor with shape [batch_size, length_m],
                the attention bias that will be added to the result of the dot product.
            cache: (Used during prediction) A dictionary with tensors containing
                results of previous attentions. The dictionary must have the items:
                    {"keys": tensor with shape [batch_size, i, heads, dim_per_head],
                    "values": tensor with shape [batch_size, i, heads, dim_per_head]}
                where i is the current decoded length.
            is_training: A bool, whether in training mode or not.
            decode_loop_step: An integer, step number of the decoding loop. Used only
                for autoregressive inference with static-shape cache.

        Returns:
            Attention layer output with shape [batch_size, length_q, output_depth]
        """
        query_is_2d = False
        if query.ndim == 2:
            # for using MultiHeadAttention in RNN-based decoders
            query_is_2d = True
            query = query.unsqueeze(1)

        # linear transformation of q, k, v
        q, k, v = self.compute_qkv(query, memory, cache, decode_loop_step)

        # Scale query to prevent the dot product between query and key from growing
        q *= (self._attention_key_depth // self._num_heads) ** (-0.5)

        # compute attention weight, [batch_size, num_heads, length_q, length_k]
        weights = self.att_fn(q, k, memory_bias)
        weights = F.dropout(weights, p=self._attention_dropout_rate, training=is_training)
        # sum over attention values
        # N: num heads
        # F: length_q
        # T: length_k
        # H: num units per head
        # attention output: [batch_size, length_q, num_heads, num_units_per_head]
        attention_output = torch.einsum("bnft,btnh->bfnh", weights, v)

        # Run the outputs through another linear projection layer. Recombining heads
        # is automatically done --> [batch_size, length_q, num_units]
        attention_output = self._output_transform_layer(attention_output)
        if query_is_2d:
            # attention output: [batch_size, depth_value]
            attention_output = attention_output.squeeze(1)
        return attention_output


class MultiHeadSelfAttention(MultiHeadAttention):
    """ Class of multi-head scaled-dot-product self-attention with input/output
      transformations. """

    def _build_qkv_transform_layer(self):
        self._qkv_transform_layer = MultiHeadDenseLayer(
            input_size=self._input_depth,
            output_units=[self._attention_key_depth,
                          self._attention_key_depth,
                          self._attention_value_depth],
            num_heads=self._num_heads,
            is_output_transform=False,
            use_bias=True)

    def forward(self, query, bias=None, cache=None, is_training=True, decode_loop_step=None):
        """ Builds the self-attention context. """
        return super(MultiHeadSelfAttention, self).forward(
            query=query,
            memory=query,
            memory_bias=bias,
            cache=cache,
            is_training=is_training,
            decode_loop_step=decode_loop_step)

    def compute_qkv(self, query, memory, cache, decode_loop_step=None):
        """ Computes linear transformations of query, keys and values, especially
            for self-attention in transformer.

        Args:
            query: Attention query tensor with shape [batch_size, length_q, channels_query],
                or [batch_size, 1, channels_query] for decoder self-attention
            memory: Unused.
            cache: Used during prediction.
            decode_loop_step: An integer, step number of the decoding loop. Used only
                for autoregressive inference with static-shape cache.

        Returns:
            A tuple `(query_transformed, key_transformed, memory_transformed)`.
        """
        _ = memory
        q, k, v = self._qkv_transform_layer(query)
        if cache is not None:
            # for self-attention in transformer decoder when mode=INFER
            if decode_loop_step is None:
                k = torch.cat([cache["keys"], k], dim=1)
                v = torch.cat([cache["values"], v], dim=1)
                cache["keys"] = k
                cache["values"] = v
            else:  # for dynamic shape

                def _insert_curr(_cache, _new_val):
                    size = _cache.size()[1]
                    indices = torch.reshape(F.one_hot(decode_loop_step, size).to(_new_val.dtype),
                                            [1, size, 1, 1])
                    new_val = _cache + _new_val * indices
                    return new_val

                cache["keys"] = _insert_curr(cache["keys"], k)
                cache["values"] = _insert_curr(cache["values"], v)
                k = cache["keys"][:, :decode_loop_step + 1]
                v = cache["values"][:, :decode_loop_step + 1]
        return q, k, v
