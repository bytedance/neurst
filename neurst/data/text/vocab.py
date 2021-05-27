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
import json
import random

import tensorflow as tf

from neurst.utils.misc import temp_download


class PaddingMode(object):
    DEFAULT = 1
    EOS_AS_PADDING = 2


class Vocab(object):
    def __init__(self, tokens, extra_tokens=None, lowercase=False):
        """ Initialize vocabulary

        Args:
            tokens: A list of word tokens.
            extra_tokens: A list of word tokens appended to the end of the vocab, which
                will not be affected by the `lowercase` option.
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
        self._extra_tokens = extra_tokens

    @property
    def tokens(self):
        return self._token_list

    @property
    def vocab_size(self):
        return len(self._token_list)

    def add_word(self, w, lowercase=False):
        """ Adds word to the end of the vocabulary. """
        if lowercase and self._lowercase:
            w = w.lower()
        if w not in self._token_list:
            self._token_list.append(w)
            self._token_to_id_dict[w] = len(self._token_list) - 1

    @staticmethod
    def load_tokens(vocab_path=None, tokens=None):
        skip_empty = True
        if not ((vocab_path is None) ^ (tokens is None)):
            raise ValueError("Either `vocab_path` or `tokens` should be provided.")
        if vocab_path:
            if vocab_path.startswith("http://") or vocab_path.startswith("https://"):
                vocab_path = temp_download(vocab_path)
            with tf.io.gfile.GFile(vocab_path) as f:
                if vocab_path.endswith(".json"):  # for gpt vocabulary
                    tokens = list(json.load(f).keys())
                    skip_empty = False
                else:
                    tokens = [line.strip("\n") for line in f]
        cleaned_tokens = []
        assert isinstance(tokens, list)
        for word in tokens:
            if (len(word) > 1 and ((word.startswith("'") and word.endswith("'"))
                                   or (word.startswith('"') and word.endswith('"')))):
                word = word[1:-1]
            else:
                if word.strip() != "" and skip_empty:
                    word = word.strip().split()[0]
            if word == "" and skip_empty:
                continue
            cleaned_tokens.append(word)
        return cleaned_tokens

    @staticmethod
    def get_unique(codebook, token):
        while token in codebook:
            token += str(random.choice(list(range(0, 10))))
        return token

    @staticmethod
    def load_from_file(vocab_path, extra_tokens=None, lowercase=False):
        """ Init vocabulary from file

        Args:
            vocab_path: A file path of the vocabulary
            extra_tokens: extra tokens appended to the end
            lowercase: whether transfer to lowercase

        Returns: A `Vocab` instance
        """
        return Vocab(Vocab.load_tokens(vocab_path, tokens=None), extra_tokens, lowercase)

    def map_token_to_id(self, tokens, unknown_default=None):
        """ Map a token to int id, if unk return unk_id

        Args:
            tokens: a word token or a list of tokens
            unknown_default: the default id if token not exists

        Returns: int id or a list of int ids
        """

        def _map(t):
            if self._lowercase and t not in self._extra_tokens:
                t = t.lower()
            try:
                return self._token_to_id_dict[t]
            except KeyError:
                return unknown_default

        if isinstance(tokens, list):
            return [_map(t) for t in tokens]
        assert isinstance(tokens, str)
        return _map(tokens)

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
