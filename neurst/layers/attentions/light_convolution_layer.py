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

from neurst.utils.activations import glu


class LightConvolutionLayer(tf.keras.layers.Layer):

    def __init__(self,
                 kernel_size,
                 num_heads,
                 conv_type="lightweight",
                 conv_dim=None,
                 use_glu=True,
                 weight_dropout_rate=0.,
                 name=None):
        super(LightConvolutionLayer, self).__init__(name=name)
        self._conv_type = conv_type
        self._conv_dim = conv_dim
        self._kernel_size = kernel_size
        self._num_heads = num_heads
        self._use_glu = use_glu
        self._weight_dropout_rate = weight_dropout_rate

    def get_config(self):
        return dict(
            conv_type=self._conv_type,
            kernel_size=self._kernel_size,
            num_heads=self._num_heads,
            conv_dim=self._conv_dim,
            use_glu=self._use_glu,
            weight_dropout_rate=self._weight_dropout_rate,
            name=self.name)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if self._conv_dim is None:
            self._conv_dim = input_dim
        assert self._conv_dim % self._num_heads == 0, (
            "The internal dimension({}) of conv must be divided "
            "evenly by num_heads({}).".format(self._conv_dim, self._num_heads))
        if self._use_glu:
            self._dense1 = tf.keras.layers.Dense(
                self._conv_dim * 2, activation=glu, use_bias=True, name="dense1")
        else:
            self._dense1 = tf.keras.layers.Dense(
                self._conv_dim, activation=None, use_bias=True, name="dense1")
        self._dense2 = tf.keras.layers.Dense(
            input_dim, activation=None, use_bias=True, name="dense2")
        if self._conv_type == "lightweight":
            self._conv_shared_weight = self.add_weight(
                "conv_shared_weight",
                shape=[self._num_heads, self._kernel_size],
                trainable=True)
        elif self._conv_type == "dynamic":
            self._conv_weight_linear = tf.keras.layers.Dense(
                self._num_heads * self._kernel_size, name="conv_weight_linear",
                use_bias=False, activation=None)
        else:
            raise NotImplementedError("Not supported conv_type: {}".format(self._conv_type))

    def dynamic_conv(self, x, left_padding=False, cache=None,
                     is_training=False, decode_loop_step=None):
        x_dim = x.get_shape().as_list()[-1]
        batch = tf.shape(x)[0]
        timestep = tf.shape(x)[1]
        # weight softmax [batch, time, heads, kernel]
        w = tf.reshape(self._conv_weight_linear(x),
                       [batch, timestep, self._num_heads, self._kernel_size])
        w = tf.nn.softmax(w)
        if left_padding:
            if cache is not None:
                if decode_loop_step is None:
                    x = tf.concat([cache["conv"], x], axis=1)
                    cache["conv"] = x
                    # [batch, timesteps(kernel_size), dim]
                    x = x[:, -self._kernel_size:, :]
                else:
                    _cache = cache["conv"]
                    size = _cache.get_shape().as_list()[1]
                    indices = tf.reshape(
                        tf.one_hot(decode_loop_step + self._kernel_size - 1, size, dtype=x.dtype),
                        [1, size, 1])
                    cache["conv"] = _cache + x * indices
                    # [batch, timesteps(kernel_size), dim]
                    x = cache["conv"][:, decode_loop_step:(self._kernel_size + decode_loop_step), :]
                # [batch, 1, dim, kernel_size]
                x_unfold = tf.expand_dims(tf.transpose(x, [0, 2, 1]), 1)
            else:
                x = tf.pad(x, [[0, 0], [self._kernel_size - 1, 0], [0, 0]], mode="CONSTANT",
                           constant_values=0)
                # x: [batch, time, dim] => [batch, 1, time, dim]
                x_unfold = tf.image.extract_patches(
                    images=tf.cast(tf.expand_dims(x, 1), dtype=tf.float32),
                    sizes=[1, 1, self._kernel_size, 1],
                    strides=[1, 1, 1, 1],
                    rates=[1, 1, 1, 1],
                    padding="VALID")
                # [batch, time, dim, kernel]
                x_unfold = tf.transpose(tf.reshape(tf.cast(x_unfold, dtype=x.dtype),
                                                   [batch, timestep, self._kernel_size, -1]), [0, 1, 3, 2])
        else:
            # x: [batch, time, dim] => [batch, 1, time, dim]
            x_unfold = tf.image.extract_patches(
                images=tf.cast(tf.expand_dims(x, 1), dtype=tf.float32),
                sizes=[1, 1, self._kernel_size, 1],
                strides=[1, 1, 1, 1],
                rates=[1, 1, 1, 1],
                padding="SAME")
            # [batch, time, dim, kernel]
            x_unfold = tf.transpose(tf.reshape(tf.cast(x_unfold, dtype=x.dtype),
                                               [batch, timestep, self._kernel_size, -1]), [0, 1, 3, 2])

        if is_training:
            w = tf.nn.dropout(w, rate=self._weight_dropout_rate)
        # [batch, time, heads, dim//heads, kernel]
        x_unfold = tf.reshape(x_unfold, [batch, timestep, self._num_heads, x_dim // self._num_heads,
                                         self._kernel_size])
        conv_out = tf.reshape(tf.einsum("abcde,abce->abcd", x_unfold, w), [batch, timestep, x_dim])
        return conv_out

    def lightweight_conv(self, x, left_padding=False, cache=None,
                         is_training=False, decode_loop_step=None):
        """ Use tf.raw_ops. """
        x_dim = x.get_shape().as_list()[-1]
        if left_padding:
            if cache is None:  # for decoder under training
                x = tf.pad(x, [[0, 0], [self._kernel_size - 1, 0], [0, 0]], mode="CONSTANT",
                           constant_values=0)
            else:  # for decoder under inference
                if decode_loop_step is None:
                    x = tf.concat([cache["conv"], x], axis=1)
                    cache["conv"] = x
                    # [batch, timesteps(kernel_size), dim]
                    x = cache["conv"][:, -self._kernel_size:, :]
                else:
                    _cache = cache["conv"]
                    size = _cache.get_shape().as_list()[1]
                    indices = tf.reshape(
                        tf.one_hot(decode_loop_step + self._kernel_size - 1, size, dtype=x.dtype),
                        [1, size, 1])
                    cache["conv"] = _cache + x * indices
                    # [batch, timesteps(kernel_size), dim]
                    x = cache["conv"][:, decode_loop_step:(self._kernel_size + decode_loop_step), :]
            padding_ = "VALID"
        else:
            padding_ = "SAME"

        # [batch, time, 1, dim]
        x = tf.expand_dims(x, axis=2)
        # weight softmax: [heads, kernel]
        w = tf.nn.softmax(self._conv_shared_weight)
        # [kernel, dim]
        filter = tf.reshape(tf.repeat(tf.transpose(w), repeats=x_dim // self._num_heads),
                            [self._kernel_size, x_dim])
        # [kernel, 1, dim, 1]
        filter = tf.expand_dims(tf.expand_dims(filter, 1), 3)
        if is_training:
            filter = tf.nn.dropout(filter, rate=self._weight_dropout_rate)
        # [batch, time, 1, dim]
        conv_out = tf.raw_ops.DepthwiseConv2dNative(
            input=x, filter=filter, strides=[1, 1, 1, 1], padding=padding_)
        conv_out = tf.squeeze(conv_out, axis=2)
        return conv_out

    def call(self, inputs, inputs_padding=None, left_padding=False,
             cache=None, is_training=False, decode_loop_step=None):
        """

        Args:
            inputs: A float tensor of shape [batch_size, length, num_units].
            inputs_padding: A float tensor with the same as `inputs`, where 1.0 indicates the padding position.
            left_padding: Whether to pad (kernel_size-1) tokens on the left of length-dim.
            cache: The internal cache for inference.
            is_training: A bool, whether in training mode or not.
            decode_loop_step: An integer, step number of the decoding loop. Used only
                for autoregressive inference with static-shape cache.

        Returns: A float tensor of shape [batch_size, length, num_units]
        """
        x = self._dense1(inputs)
        if inputs_padding is not None:
            x *= tf.expand_dims((1. - inputs_padding), 2)
        if self._conv_type == "lightweight":
            conv_output = self.lightweight_conv(x,
                                                left_padding=left_padding,
                                                cache=cache, is_training=is_training,
                                                decode_loop_step=decode_loop_step)
        elif self._conv_type == "dynamic":
            conv_output = self.dynamic_conv(x,
                                            left_padding=left_padding,
                                            cache=cache, is_training=is_training,
                                            decode_loop_step=decode_loop_step)
        else:
            raise NotImplementedError("Not supported conv_type: {}".format(self._conv_type))
        return self._dense2(conv_output)
