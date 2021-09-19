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

from neurst.utils.activations import gelu
from neurst.utils.configurable import extract_constructor_params


class AudioConv2dSubsamplingLayer(tf.keras.layers.Layer):
    """ Subsampling for audio features. """

    def __init__(self,
                 embedding_dim,
                 channels=256,
                 kernel_size=3,
                 strides=2,
                 layer_norm=True,
                 num_layers=2,
                 name=None):
        """ Initializes the layer for subsample the audio feature.

        Args:
            embedding_dim: An int scalar, the embedding dimension.
            channels: The channel size of the convolution layer.
            kernel_size: The kernel size of the convolution layer.
            strides: The stride size of the convolution layer.
            layer_norm: Whether to apply layer normalization.
            num_layers: The number of conv layers.
            name: The name of the layer.
        """
        self._params = extract_constructor_params(locals(), verbose=True)
        super(AudioConv2dSubsamplingLayer, self).__init__(name=name)
        self._embedding_dim = embedding_dim
        self._channels = channels
        self._kernel_size = kernel_size
        self._layer_norm = layer_norm
        self._strides = strides
        self._num_layers = num_layers
        self._conv_layers = []
        self._norm_layers = []

    @property
    def embedding_dim(self):
        return self._embedding_dim

    def get_config(self):
        return self._params

    def build(self, input_shape):
        """ Builds the layer."""
        for i in range(1, self._num_layers + 1):
            self._conv_layers.append(tf.keras.layers.Conv2D(
                filters=self._channels,
                kernel_size=(self._kernel_size, self._kernel_size),
                strides=(self._strides, self._strides),
                padding="VALID",
                activation=None,
                name=f"conv{i}"))
            if self._layer_norm:
                self._norm_layers.append(tf.keras.layers.LayerNormalization(
                    epsilon=1e-6, dtype="float32", name=f"ln{i}"))
            else:
                self._norm_layers.append(None)
        self._dense_layer = tf.keras.layers.Dense(
            self._embedding_dim,
            activation=None,
            use_bias=True,
            name="output_dense")
        super(AudioConv2dSubsamplingLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """ Gets token embeddings or computes logits.

        Args:
            inputs: An float tensor with shape [batch_size, length, feature_dim, channels].

        Returns:
            A float tensor with shape [batch, new_length, new_feature_dim].
        """
        _ = kwargs
        audio_feature_dim = inputs.get_shape().as_list()[2]
        assert inputs.get_shape().ndims == 4
        manual_padding = [[0, 0], [self._kernel_size // 2, self._kernel_size // 2],
                          [self._kernel_size // 2, self._kernel_size // 2], [0, 0]]
        conv_out = inputs
        new_feature_dim = audio_feature_dim
        for conv_layer, norm_layer in zip(self._conv_layers, self._norm_layers):
            conv_layer_output = conv_layer(tf.pad(conv_out, manual_padding))
            if norm_layer is not None:
                conv_layer_output = norm_layer(conv_layer_output)
            conv_out = tf.nn.relu(conv_layer_output)
            new_feature_dim = (new_feature_dim + self._strides - 1) // self._strides
        new_feature_dim *= self._channels
        conv_reshape = tf.reshape(conv_out, tf.concat([tf.shape(conv_out)[:2], [new_feature_dim]], axis=0))
        output = self._dense_layer(conv_reshape)
        return output


class Wav2vec2ConvBlock(tf.keras.layers.Layer):
    """ The convolution block of Wav2vec2FeatureExtractor. """

    def __init__(self, dim, kernel, stride, dropout_rate=0.,
                 use_bias=False, norm_type=None, name=None):
        """

        Args:
            dim: The output dimension of this convolution layer.
            kernel: The kernel size.
            stride: The stride.
            dropout_rate: The dropout rate.
            use_bias: Whether to include bias in conv encoder.
            norm_type: The type of layer normalization, "layer" or "group" or None.
            name: The name of this layer
        """
        self._params = extract_constructor_params(locals(), verbose=False)
        super(Wav2vec2ConvBlock, self).__init__(name=name)
        self._dim = dim
        self._kernel = kernel
        self._stride = stride
        self._dropout_rate = dropout_rate
        self._use_bias = use_bias
        self._norm_type = norm_type

    def get_config(self):
        return self._params

    def build(self, input_shape):
        """ Builds the layer."""
        self._conv_layer = tf.keras.layers.Conv1D(
            self._dim, kernel_size=self._kernel,
            strides=self._stride, use_bias=self._use_bias,
            kernel_initializer=tf.keras.initializers.HeNormal(),
            name="conv")
        self._norm_layer = None
        if self._norm_type == "layer":
            self._norm_layer = tf.keras.layers.LayerNormalization(
                epsilon=1.e-5, dtype="float32", name="ln")
        elif self._norm_type is not None:
            assert self._norm_type == "group"
            import tensorflow_addons as tfa
            self._norm_layer = tfa.layers.GroupNormalization(
                self._dim, epsilon=1.e-5, axis=-1, dtype="float32", name="gn")

    def call(self, inputs, is_training=False):
        """ Applies the convolution.

        Args:
            inputs: A Tensor of shape [batch, width, channels]
            is_training: Whether is under training.

        Returns:
            The convolution output.
        """
        conv_out = self._conv_layer(inputs)
        if is_training and self._dropout_rate > 0:
            conv_out = tf.nn.dropout(conv_out, rate=self._dropout_rate)
        if self._norm_layer is not None:
            conv_out = self._norm_layer(conv_out)
        # The original fairseq wav2vec use the non-approximated version
        return gelu(conv_out, non_approximate=True)


class Wav2vec2FeatureExtractor(tf.keras.layers.Layer):
    """ Subsampling for raw audios. """

    def __init__(self,
                 conv_layers,
                 dropout=0.0,
                 mode="default",
                 conv_bias=False,
                 verbose=False,
                 name=None):
        """ Initializes wav2vec2's convolution layers.

        Args:
            conv_layers: A list of convolution layers, each of which is in form of [dim, kernel, stride].
            dropout: The dropout rate of each conv layer.
            mode: The mode for feature extractor. "default" has a single group norm with d
                groups in the first conv block, whereas layer_norm has layer norms in
                every block (meant to use with normalize=True)
            conv_bias: Whether to include bias in conv encoder.
            verbose: A boolean, whether to logging the parameters.
            name: The name of the layer.
        """
        self._params = extract_constructor_params(locals(), verbose=True)
        super(Wav2vec2FeatureExtractor, self).__init__(name=name)
        self._conv_layers_setting = conv_layers
        self._dropout = dropout
        self._mode = mode
        self._conv_bias = conv_bias
        self._conv_layers = []

    def get_config(self):
        return self._params

    def build(self, input_shape):
        """ Builds the layer."""
        for i, (dim, kernel, stride) in enumerate(self._conv_layers_setting):
            norm_type = None
            if self._mode == "layer_norm":
                norm_type = "layer"
            elif self._mode == "default" and i == 0:
                norm_type = "group"
            self._conv_layers.append(Wav2vec2ConvBlock(
                dim, kernel, stride, dropout_rate=self._dropout,
                use_bias=self._conv_bias, norm_type=norm_type, name=f"conv_block{i}"))
        super(Wav2vec2FeatureExtractor, self).build(input_shape)

    def call(self, inputs, is_training=False):
        """ Applies the convolutional feature extration.

        Args:
            inputs: An float tensor with shape [batch_size, width].
            is_training: Whether is under training.

        Returns:
            A float tensor with shape [batch, new_width, output_channels].
        """
        output = tf.expand_dims(inputs, axis=-1)
        for conv_layer in self._conv_layers:
            output = conv_layer(output, is_training=is_training)
        return output


class PositionalConv(tf.keras.layers.Layer):
    """ Defines the pos_conv described in Wav2vec2. """

    def __init__(self, dim, kernel_size, groups,
                 dropout=0, name=None):
        super(PositionalConv, self).__init__(name=name)
        self._dim = dim
        self._kernel_size = kernel_size
        self._groups = groups
        self._dropout = dropout

    def build(self, input_shape):
        class _Wav2vec2ConvWeightNorm(tf.keras.layers.Wrapper):
            """Performs weight normalization.
            This is modified from tfa.layers.WeightNormalization.

            """

            def __init__(self, layer: tf.keras.layers, **kwargs):
                super(_Wav2vec2ConvWeightNorm, self).__init__(layer, **kwargs)
                self._track_trackable(layer, name="layer")

            def build(self, input_shape):
                """Build `Layer`"""
                input_shape = tf.TensorShape(input_shape)
                self.input_spec = tf.keras.layers.InputSpec(shape=[None] + input_shape[1:])

                if not self.layer.built:
                    self.layer.build(input_shape)

                kernel_layer = self.layer
                kernel = kernel_layer.kernel

                self.layer_depth = int(kernel.shape[0])
                self.kernel_norm_axes = list(range(kernel.shape.rank - 1))
                self.g = self.add_weight(
                    name="g",
                    shape=(self.layer_depth,),
                    initializer="ones",
                    # dtype=kernel.dtype,
                    trainable=True,
                )
                self.v = kernel
                self._initialized = self.add_weight(
                    name="initialized",
                    shape=None,
                    initializer="zeros",
                    dtype=tf.dtypes.bool,
                    trainable=False)
                self.built = True

            def call(self, inputs):
                """Call `Layer`"""

                def _do_nothing():
                    return tf.identity(self.g)

                def _update_weights():
                    # Ensure we read `self.g` after _update_weights.
                    with tf.control_dependencies(self._initialize_weights(inputs)):
                        return tf.identity(self.g)

                g = tf.cond(self._initialized, _do_nothing, _update_weights)

                with tf.name_scope("compute_weights"):
                    # Replace kernel by normalized weight variable.
                    kernel = tf.transpose(tf.nn.l2_normalize(tf.transpose(self.v, [2, 1, 0]),
                                                             axis=self.kernel_norm_axes) * g, [2, 1, 0])
                    self.layer.kernel = kernel
                    update_kernel = tf.identity(self.layer.kernel)

                    # Ensure we calculate result after updating kernel.
                    with tf.control_dependencies([update_kernel]):
                        outputs = self.layer(inputs)
                        return outputs

            def compute_output_shape(self, input_shape):
                return tf.TensorShape(self.layer.compute_output_shape(input_shape).as_list())

            def _initialize_weights(self, inputs):
                """Initialize weight g.

                The initial value of g could either from the initial value in v,
                or by the input value if self.data_init is True.
                """
                with tf.control_dependencies(
                    [
                        tf.debugging.assert_equal(  # pylint: disable=bad-continuation
                            self._initialized, False, message="The layer has been initialized."
                        )
                    ]
                ):
                    assign_tensors = self._init_norm()
                    assign_tensors.append(self._initialized.assign(True))
                    return assign_tensors

            def _init_norm(self):
                """Set the weight g with the norm of the weight vector."""
                with tf.name_scope("init_norm"):
                    v_flat = tf.reshape(self.v, [-1, self.layer_depth])
                    v_norm = tf.linalg.norm(v_flat, axis=0)
                    g_tensor = self.g.assign(tf.cast(tf.reshape(v_norm, (self.layer_depth,)), "float32"))
                    return [g_tensor]

        self._pos_conv = _Wav2vec2ConvWeightNorm(
            tf.keras.layers.Conv1D(
                self._dim, kernel_size=self._kernel_size, groups=self._groups,
                kernel_initializer=tf.initializers.random_normal(
                    mean=0, stddev=math.sqrt(4. / (self._kernel_size * self._dim))),
                name="conv"), name="wn")
        self._norm_layer = tf.keras.layers.LayerNormalization(
            epsilon=1.e-5, dtype="float32", name="ln")

    def call(self, inputs, inputs_padding=None, is_training=False):
        """

        Args:
            inputs: A tensor with shape [batch, time, channels]
            inputs_padding: A tensor with the same shape as `inputs`, where 1. denotes the padding.
            is_training: Whether is under training.
        Returns:

        """
        if inputs_padding is not None:
            inputs = inputs * (1. - tf.expand_dims(tf.cast(inputs_padding, dtype=inputs.dtype), -1))
        padding_num = self._kernel_size // 2
        x = tf.pad(inputs, [[0, 0], [padding_num, padding_num], [0, 0]])
        x_conv = self._pos_conv(x)[:, : tf.shape(inputs)[1] - self._kernel_size % 2, :]
        x_conv = gelu(x_conv, non_approximate=True) + inputs
        x_conv = self._norm_layer(x_conv)
        if is_training:
            x_conv = tf.nn.dropout(x_conv, self._dropout)
        return x_conv
