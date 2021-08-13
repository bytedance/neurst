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
import tensorflow as tf

from neurst.layers.common_layers import MultiHeadDenseLayer
from neurst.layers.quantization.quant_layers import QuantLayer
from neurst.utils.configurable import extract_constructor_params


class MultiHeadAttention(QuantLayer):
    """ Class of multi-head scaled-dot-product attention with input/output
        transformations. """

    def __init__(self,
                 num_heads,
                 num_units,
                 attention_key_depth=None,
                 attention_value_depth=None,
                 output_depth=None,
                 attention_dropout_rate=0.1,
                 attention_type="dot_product",
                 name=None):
        """ Initializes the multi head attention layer.

        Args:
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
            name: The name of the layer.
        """
        self._params = extract_constructor_params(locals(), verbose=False)
        super(MultiHeadAttention, self).__init__(name=name)
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
            output_units=self._output_depth, num_heads=self._num_heads,
            kernel_initializer="glorot_uniform", is_output_transform=True,
            use_bias=True, name="output_transform")

    def get_config(self):
        return self._params

    def build(self, input_shape):
        """ Builds the layer.
            Layers for linearly projecting the queries, keys, and values."""
        self._q_transform_layer = MultiHeadDenseLayer(
            output_units=self._attention_key_depth, num_heads=self._num_heads,
            kernel_initializer="glorot_uniform", is_output_transform=False,
            use_bias=True, name="q_transform")
        self._kv_transform_layer = MultiHeadDenseLayer(
            output_units=[self._attention_key_depth, self._attention_value_depth],
            num_heads=self._num_heads,
            kernel_initializer="glorot_uniform",
            is_output_transform=False,
            use_bias=True,
            name="kv_transform")
        self.add_activation_quantizer(name="output", activation="act")
        self.add_activation_quantizer(name="softmax", activation="softmax")
        self.built = True

    def compute_kv(self, memory):
        """ Computes linear transformations of keys and values.

        Args:
            memory: A tensor with shape [batch_size, length_m, memory_depth].

        Returns: A tuple `(key_transformed, memory_transformed)`.

        """
        return self._kv_transform_layer(memory)

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
        _ = decode_loop_step
        # [batch_size, length_q/k/v, num_heads, num_units_per_head]
        q = self._q_transform_layer(query)
        if cache is not None:
            k, v = cache["keys"], cache["values"]
        else:
            k, v = self.compute_kv(memory)
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
            logits = tf.einsum("BTNH,BFNH->BNFT", k, q)

            if bias is not None:
                if bias.get_shape().ndims == 2:
                    bias = tf.expand_dims(
                        tf.expand_dims(bias, axis=1), axis=1)
                elif bias.get_shape().ndims == 3:
                    bias = tf.expand_dims(bias, axis=1)
                elif bias.get_shape().ndims != 4:
                    raise ValueError("bias tensor with {}-dim is not valid".format(
                        bias.get_shape().ndims))
                logits += bias
            # Note that softmax internally performs math operations using float32
            # for numeric stability. When training with float16, we keep the input
            # and output in float16 for better performance.
            weights = self.quant(tf.nn.softmax(logits), name="softmax")
        else:
            raise NotImplementedError(
                "att_fn for \"{}\" not implemented.".format(self._attention_type))
        return weights

    def call(self,
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
        if query.get_shape().ndims == 2:
            # for using MultiHeadAttention in RNN-based decoders
            query_is_2d = True
            query = tf.expand_dims(query, axis=1)

        # linear transformation of q, k, v
        q, k, v = self.compute_qkv(query, memory, cache, decode_loop_step)

        # Scale query to prevent the dot product between query and key from growing
        q *= (self._attention_key_depth // self._num_heads) ** (-0.5)

        # compute attention weight, [batch_size, num_heads, length_q, length_k]
        weights = self.att_fn(q, k, memory_bias)
        if is_training:
            weights = tf.nn.dropout(weights, rate=self._attention_dropout_rate)
        # sum over attention values
        # N: num heads
        # F: length_q
        # T: length_k
        # H: num units per head
        # attention output: [batch_size, length_q, num_heads, num_units_per_head]
        attention_output = self.quant(tf.einsum("BNFT,BTNH->BFNH", weights, v), name="output")

        # Run the outputs through another linear projection layer. Recombining heads
        # is automatically done --> [batch_size, length_q, num_units]
        attention_output = self._output_transform_layer(attention_output)
        if query_is_2d:
            # attention output: [batch_size, depth_value]
            attention_output = tf.squeeze(attention_output, axis=1)
        return attention_output


class MultiHeadSelfAttention(MultiHeadAttention):
    """ Class of multi-head scaled-dot-product self-attention with input/output
      transformations. """

    def build(self, input_shape):
        self._qkv_transform_layer = MultiHeadDenseLayer(
            output_units=[self._attention_key_depth,
                          self._attention_key_depth,
                          self._attention_value_depth],
            num_heads=self._num_heads,
            kernel_initializer="glorot_uniform",
            is_output_transform=False,
            use_bias=True,
            name="qkv_transform")
        self.add_activation_quantizer(name="output", activation="act")
        self.add_activation_quantizer(name="softmax", activation="softmax")
        self.built = True

    def call(self, query, bias=None, cache=None, is_training=True, decode_loop_step=None):
        """ Builds the self-attention context. """
        return super(MultiHeadSelfAttention, self).call(
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
        if cache is not None:  # for self-attention in transformer decoder when mode=INFER
            if decode_loop_step is None:  # for dynamic shape
                k = tf.concat([cache["keys"], k], axis=1)
                v = tf.concat([cache["values"], v], axis=1)
                cache["keys"] = k
                cache["values"] = v
            else:  # for static shape

                def _insert_curr(_cache, _new_val):
                    size = _cache.get_shape().as_list()[1]
                    indices = tf.reshape(tf.one_hot(decode_loop_step, size, dtype=_new_val.dtype),
                                         [1, size, 1, 1])
                    new_val = _cache + _new_val * indices
                    return new_val

                cache["keys"] = _insert_curr(cache["keys"], k)
                cache["values"] = _insert_curr(cache["values"], v)
                k = cache["keys"][:, :decode_loop_step + 1]
                v = cache["values"][:, :decode_loop_step + 1]
        return q, k, v
