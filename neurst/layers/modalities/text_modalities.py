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

from neurst.layers.quantization.quant_layers import QuantLayer
from neurst.utils import compat
from neurst.utils.configurable import extract_constructor_params


class WordEmbeddingSharedWeights(QuantLayer):
    """Calculates input embeddings and pre-softmax linear with shared weights. """

    def __init__(self,
                 embedding_dim,
                 vocab_size,
                 share_softmax_weights=False,
                 use_bias=True,
                 verbose=False,
                 name=None):
        """ Initializes simple word embedding layer.

        Args:
            embedding_dim: An int scalar, the embedding dimension.
            vocab_size: An int scalar, the size of vocabulary.
            share_softmax_weights: A boolean, whether to share
                embedding table with target softmax weight.
            use_bias: A boolean, whether to use bias with target
                softmax weight.
            verbose: A boolean, whether to logging the parameters.
            name: The name of the layer.
        """
        self._params = extract_constructor_params(locals(), verbose=False)
        super(WordEmbeddingSharedWeights, self).__init__(name=name)
        self._embedding_dim = embedding_dim
        self._vocab_size = vocab_size
        self._share_softmax_weights = share_softmax_weights
        self._use_bias = use_bias

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
            self._bias = None
            if self._share_softmax_weights and self._use_bias:
                self._bias = self.add_weight(
                    "bias",
                    shape=(self._vocab_size,),
                    # initializer=tf.zeros_initializer,
                    trainable=True)
        super(WordEmbeddingSharedWeights, self).build(input_shape)

    def _bottom(self, x):
        """ Embedding lookup.

        Args:
            x: A 1/2-d Tensor to be embedded.

        Returns: A 2/3-d Tensor according to `x`.
        """
        emb = tf.gather(self.quant_weight(self._shared_weights), x)
        return emb

    def _top(self, x):
        """ Computes logits on the top layer.

        Args:
            x: A Tensor with shape [..., hidden]

        Returns: A logits Tensor with shape [..., vocab_size].
        """
        original_shape = tf.shape(x)
        logits = tf.matmul(tf.reshape(x, [-1, self._embedding_dim]),
                           tf.cast(self.quant_weight(self._shared_weights), x.dtype),
                           transpose_b=True)
        if self._bias is not None:
            logits += self._bias

        # logits = self.quant(logits, name="logits")

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


class TokenTypeEmbedding(tf.keras.layers.Layer):
    """Calculates input embeddings and pre-softmax linear with shared weights. """

    def __init__(self,
                 embedding_dim,
                 vocab_size,
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
        super(TokenTypeEmbedding, self).__init__(name=name)
        self._embedding_dim = embedding_dim
        self._vocab_size = vocab_size
        self._emb_table = None

    @property
    def embedding_dim(self):
        return self._embedding_dim

    @property
    def vocab_size(self):
        return self._vocab_size

    def get_config(self):
        return self._params

    def build(self, input_shape):
        with tf.name_scope("token_type_emb"):
            self._emb_table = self.add_weight(
                "weights",
                shape=(self._vocab_size, self._embedding_dim),
                trainable=True,
                initializer=tf.random_normal_initializer(
                    mean=0., stddev=self._embedding_dim ** -0.5))
        super(TokenTypeEmbedding, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """ Gets token em beddings or computes logits.

        Args:
            inputs: An int tensor with shape [batch_size, length] or [batch, ].

        Returns:
            A float tensor with shape [batch, length, embedding_dim] or [batch, embedding_dim].
        """
        _ = kwargs
        # This vocab will be small so we always do one-hot here, since it is always
        # faster for a small vocabulary.
        flat_token_type_ids = tf.reshape(inputs, [-1])
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=self.vocab_size)
        token_type_embeddings = tf.matmul(tf.cast(one_hot_ids, self._emb_table.dtype),
                                          self._emb_table)
        token_type_embeddings = tf.reshape(token_type_embeddings,
                                           tf.concat([tf.shape(inputs), [self.embedding_dim]], axis=0))
        return token_type_embeddings


class BertEmbedding(tf.keras.layers.Layer):
    """ The bert embedding layer. """

    def __init__(self,
                 embedding_dim,
                 vocab_size,
                 max_positions,
                 token_types,
                 dropout_rate=0.0,
                 epsilon=1e-12,
                 name=None):
        self._params = extract_constructor_params(locals(), verbose=False)
        super(BertEmbedding, self).__init__(name=name)
        self._embedding_dim = embedding_dim
        self._vocab_size = vocab_size
        self._max_positions = max_positions
        self._token_types = token_types
        self._dropout_rate = dropout_rate
        self._epsilon = epsilon

    @property
    def embedding_dim(self):
        return self._embedding_dim

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def token_types(self):
        return self._token_types

    def get_config(self):
        return self._params

    def build(self, input_shape):
        self._word_embedding = self.add_weight(
            "word_embedding",
            shape=(self._vocab_size, self._embedding_dim),
            trainable=True,
            initializer=tf.random_normal_initializer(
                mean=0., stddev=self._embedding_dim ** -0.5))
        self._token_type_embedding = self.add_weight(
            "token_type_embedding",
            shape=(self._token_types, self._embedding_dim),
            trainable=True,
            initializer=tf.random_normal_initializer(
                mean=0., stddev=self._embedding_dim ** -0.5))
        self._position_embedding = self.add_weight(
            "position_embedding",
            shape=(self._max_positions, self._embedding_dim),
            initializer=tf.random_normal_initializer(
                mean=0., stddev=self._embedding_dim ** -0.5),
            trainable=True)
        self._norm_layer = tf.keras.layers.LayerNormalization(
            epsilon=self._epsilon, dtype="float32", name="ln")
        super(BertEmbedding, self).build(input_shape)

    def call(self, tokens, tokens_type=None, is_training=False):
        """ Gets bert embedding.

        Args:
            tokens: An int tensor with shape [batch_size, length].
            tokens_type: An int tensor of shape [batch_size, length].

        Returns:
            A float tensor with shape [batch, length, embedding_dim].
        """
        assert tokens.get_shape().ndims == 2
        embeddings = tf.gather(self._word_embedding, tokens)
        # This vocab will be small so we always do one-hot here, since it is always
        # faster for a small vocabulary.
        if tokens_type is None:
            tokens_type = tf.zeros_like(tokens, dtype=tf.int64)
        flat_token_type_ids = tf.reshape(tokens_type, [-1])
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=self.token_types)
        token_type_embeddings = tf.matmul(tf.cast(one_hot_ids, compat.CUSTOM_GLOBAL_FLOATX),
                                          tf.cast(self._token_type_embedding, compat.CUSTOM_GLOBAL_FLOATX))
        token_type_embeddings = tf.reshape(token_type_embeddings,
                                           tf.concat([tf.shape(tokens_type), [self.embedding_dim]], axis=0))
        position_embeddings = tf.expand_dims(tf.slice(self._position_embedding, [0, 0],
                                                      [tf.shape(tokens)[1], -1]), axis=0)
        bert_emb = self._norm_layer(embeddings + token_type_embeddings + position_embeddings)
        if is_training:
            bert_emb = tf.nn.dropout(bert_emb, rate=self._dropout_rate)
        return bert_emb
