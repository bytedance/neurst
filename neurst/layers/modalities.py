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

from neurst.utils.configurable import extract_constructor_params


class WordEmbeddingSharedWeights(tf.keras.layers.Layer):
    """Calculates input embeddings and pre-softmax linear with shared weights. """

    def __init__(self,
                 embedding_dim,
                 vocab_size,
                 share_softmax_weights=False,
                 verbose=False,
                 name=None):
        """ Initializes simple word embedding layer.

        Args:
            embedding_dim: An int scalar, the embedding dimension.
            vocab_size: An int scalar, the size of vocabulary.
            share_softmax_weights: A boolean, whether to share
                embedding table with target softmax weight.
            verbose: A boolean, whether to logging the parameters.
            name: The name of the layer.
        """
        self._params = extract_constructor_params(locals(), verbose=False)
        super(WordEmbeddingSharedWeights, self).__init__(name=name)
        self._embedding_dim = embedding_dim
        self._vocab_size = vocab_size
        self._share_softmax_weights = share_softmax_weights

    @property
    def embedding_dim(self):
        return self._embedding_dim

    @property
    def vocab_size(self):
        return self._vocab_size

    def get_config(self):
        return self._params

    def build(self, input_shape):
        """ Builds the embedding table and softmax bias
            if share_softmax_weights=True."""
        if self._share_softmax_weights:
            scope_name = "shared"
        else:
            scope_name = "emb"
        with tf.name_scope(scope_name):
            self._shared_weights = self.add_weight(
                "weights",
                shape=(self._vocab_size, self._embedding_dim),
                trainable=True,
                initializer=tf.random_normal_initializer(
                    mean=0., stddev=self._embedding_dim ** -0.5))
            if self._share_softmax_weights:
                self._bias = self.add_weight(
                    "bias",
                    shape=(self._vocab_size,),
                    trainable=True)
        super(WordEmbeddingSharedWeights, self).__init__(input_shape)

    def _bottom(self, x):
        """ Embedding lookup.

        Args:
            x: A 1/2-d Tensor to be embedded.

        Returns: A 2/3-d Tensor according to `x`.
        """
        emb = tf.gather(self._shared_weights, x)
        return emb

    def _top(self, x):
        """ Computes logits on the top layer.

        Args:
            x: A Tensor with shape [..., hidden]

        Returns: A logits Tensor with shape [..., vocab_size].
        """
        original_shape = tf.shape(x)
        logits = tf.matmul(tf.reshape(x, [-1, self._embedding_dim]),
                           self._shared_weights, transpose_b=True) + self._bias

        return tf.reshape(logits, tf.concat(
            [original_shape[:-1], [self._vocab_size]], axis=0))

    def call(self, inputs, mode="embedding", **kwargs):
        """ Gets token em beddings or computes logits.

        Args:
            inputs: An int tensor with shape [batch_size, length] or [batch, ].
            mode: A string, a valid value is one of "embedding" and "linear".

        Returns:
            A float tensor with shape [batch, length, embedding_dim]
            or [batch, embedding_dim] when mode == "embedding" ;
            A float tensor with shape [batch, length, vocab_size]
            when mode == "linear".
        """
        _ = kwargs
        if mode == "embedding":
            return self._bottom(inputs)
        elif mode == "linear":
            return self._top(inputs)
        else:
            raise ValueError("mode = {} is not valid.".format(mode))


class AudioConvSubsamplingLayer(tf.keras.layers.Layer):
    """ Subsampling for audio features. """

    def __init__(self,
                 embedding_dim,
                 channels=256,
                 kernel_size=3,
                 strides=2,
                 layer_norm=True,
                 verbose=False,
                 name=None):
        """ Initializes the layer for subsample the audio feature.

        Args:
            embedding_dim: An int scalar, the embedding dimension.
            channels: The channel size of the convolution layer.
            kernel_size: The kernel size of the convolution layer.
            strides: The stride size of the convolution layer.
            layer_norm: Whether to apply layer normalization.
            verbose: A boolean, whether to logging the parameters.
            name: The name of the layer.
        """
        self._params = extract_constructor_params(locals(), verbose=True)
        super(AudioConvSubsamplingLayer, self).__init__(name=name)
        self._embedding_dim = embedding_dim
        self._channels = channels
        self._kernel_size = kernel_size
        self._layer_norm = layer_norm
        self._strides = strides

    @property
    def embedding_dim(self):
        return self._embedding_dim

    def get_config(self):
        return self._params

    def build(self, input_shape):
        """ Builds the layer."""
        self._conv_layer1 = tf.keras.layers.Conv2D(
            filters=self._channels,
            kernel_size=(self._kernel_size, self._kernel_size),
            strides=(self._strides, self._strides),
            padding="VALID",
            activation=None,
            name="conv1")
        self._conv_layer2 = tf.keras.layers.Conv2D(
            filters=self._channels,
            kernel_size=(self._kernel_size, self._kernel_size),
            strides=(self._strides, self._strides),
            padding="VALID",
            activation=None,
            name="conv2")
        if self._layer_norm:
            self._norm_layer1 = tf.keras.layers.LayerNormalization(
                epsilon=1e-6, dtype="float32", name="ln1")
            self._norm_layer2 = tf.keras.layers.LayerNormalization(
                epsilon=1e-6, dtype="float32", name="ln2")
        self._dense_layer = tf.keras.layers.Dense(
            self._embedding_dim,
            activation=None,
            use_bias=True,
            name="output_dense")
        super(AudioConvSubsamplingLayer, self).__init__(input_shape)

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
        conv1 = self._conv_layer1(tf.pad(inputs, manual_padding))
        if self._layer_norm:
            conv1 = self._norm_layer1(conv1)
        conv1 = tf.nn.relu(conv1)
        conv2 = self._conv_layer2(tf.pad(conv1, manual_padding))
        if self._layer_norm:
            conv2 = self._norm_layer2(conv2)
        conv2 = tf.nn.relu(conv2)
        new_feature_dim = ((audio_feature_dim + self._strides - 1)
                           // self._strides + self._strides - 1) // self._strides * self._channels
        conv2_reshape = tf.reshape(conv2, tf.concat([tf.shape(conv2)[:2], [new_feature_dim]], axis=0))
        output = self._dense_layer(conv2_reshape)
        return output
