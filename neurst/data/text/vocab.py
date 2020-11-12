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
import copy

import tensorflow as tf


class Vocab(object):
    def __init__(self, tokens, extra_tokens=None, lowercase=False):
        """ Initialize vocabulary

        Args:
            tokens: a list of words
            extra_tokens: extra tokens appended to the end
            lowercase: whether transfer to lowercase
        """
        assert isinstance(tokens, list), (
            "`tokens` must be a list of string tokens")
        if lowercase:
            uniq_tokens = []
            for token in tokens:
                token = token.lower()
                if token not in uniq_tokens:
                    uniq_tokens.append(token)
            tokens = uniq_tokens
        self._token_list = copy.deepcopy(tokens)
        if isinstance(extra_tokens, list):
            for token in extra_tokens:
                if token not in self._token_list:
                    self._token_list.append(token)
        self._token_to_id_dict = dict([
            (w, i) for i, w in enumerate(self._token_list)])
        self._lowercase = lowercase
        self._extra_tokens = copy.deepcopy(extra_tokens)

    @property
    def vocab_size(self):
        return len(self._token_list)

    @staticmethod
    def load_from_file(vocab_path, extra_tokens=None, lowercase=False):
        """ Init vocabulary from file

        Args:
            vocab_path: A file path of the vocabulary
            extra_tokens: extra tokens appended to the end
            lowercase: whether transfer to lowercase

        Returns: A `Vocab` instance
        """
        tokens = []
        with tf.io.gfile.GFile(vocab_path) as f:
            for line in f:
                a = line.strip().split('\t')
                word = a[0]
                if word.strip() == "":
                    continue
                tokens.append(word.strip())
        return Vocab(tokens, extra_tokens, lowercase)

    def map_token_to_id(self, tokens, unknown_default=None):
        """ Map a token to int id, if unk return unk_id

        Args:
            tokens: a word token or a list of tokens
            unknown_default: the default id if token not exists

        Returns: int id or a list of int ids
        """

        def _map(token2id, lc, t):
            if lc and t not in self._extra_tokens:
                t = t.lower()
            try:
                return token2id[t]
            except KeyError:
                return unknown_default

        if isinstance(tokens, list):
            return [_map(self._token_to_id_dict, self._lowercase, t)
                    for t in tokens]
        assert isinstance(tokens, str)
        return _map(self._token_to_id_dict, self._lowercase, tokens)

    def map_id_to_token(self, ids):
        """ Map an integer(s) to word token(s)

        Args:
            ids: An integer or a list of integers

        Returns: a string token or list of strings
        """
        if isinstance(ids, list):
            return [self._token_list[i] for i in ids]
        assert isinstance(ids, int)
        return self._token_list[ids]
