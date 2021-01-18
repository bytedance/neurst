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

from neurst.utils.configurable import extract_constructor_params


class WordEmbeddingSharedWeights(nn.Module):
    """Calculates input embeddings and pre-softmax linear with shared weights. """

    def __init__(self,
                 embedding_dim,
                 vocab_size,
                 share_softmax_weights=False,
                 use_bias=True,
                 verbose=False):
        """ Initializes simple word embedding layer.

        Args:
            embedding_dim: An int scalar, the embedding dimension.
            vocab_size: An int scalar, the size of vocabulary.
            share_softmax_weights: A boolean, whether to share
                embedding table with target softmax weight.
            use_bias: A boolean, whether to use bias with target
                softmax weight.
            verbose: A boolean, whether to logging the parameters.
        """
        self._params = extract_constructor_params(locals(), verbose=False)
        super(WordEmbeddingSharedWeights, self).__init__()
        self._embedding_dim = embedding_dim
        self._vocab_size = vocab_size
        self._share_softmax_weights = share_softmax_weights
        self._use_bias = use_bias
        self._shared_weights = nn.Parameter(nn.init.normal_(
            torch.empty(vocab_size, embedding_dim),
            mean=0., std=embedding_dim ** -0.5), requires_grad=True)
        self._bias = None
        if self._share_softmax_weights and self._use_bias:
            self._bias = nn.Parameter(torch.zeros(vocab_size), requires_grad=True)

    @property
    def embedding_dim(self):
        return self._embedding_dim

    @property
    def vocab_size(self):
        return self._vocab_size

    def get_config(self):
        return self._params

    def _bottom(self, x):
        """ Embedding lookup.

        Args:
            x: A 1/2-d Tensor to be embedded.

        Returns: A 2/3-d Tensor according to `x`.
        """
        return self._shared_weights[x.to(torch.long)]

    def _top(self, x):
        """ Computes logits on the top layer.

        Args:
            x: A Tensor with shape [..., hidden]

        Returns: A logits Tensor with shape [..., vocab_size].
        """
        original_shape = list(x.size())
        logits = torch.matmul(torch.reshape(x, [-1, self._embedding_dim]),
                              self._shared_weights.transpose(1, 0))
        if self._bias is not None:
            logits += self._bias

        return torch.reshape(logits, original_shape[:-1] + [self._vocab_size])

    def forward(self, inputs, mode="embedding", **kwargs):
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
