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
import random

import tensorflow as tf

from neurst.data.text.vocab import Vocab
from neurst.utils.configurable import extract_constructor_params


class SymbolsMapper(object):
    def __init__(self,
                 vocab_path=None,
                 tokens=None,
                 max_len=0,
                 lowercase=False,
                 bos_token="<SEQ_BEG>",
                 eos_token="<SEQ_END>",
                 unk_token="<UNK>",
                 delimiter=" ",
                 reverse=False):
        """ Initialize SymbolsMapper

        Args:
            vocab_path: The path to the vocabulary file. Only one of `vocab_path` and `tokens` should be provided.
            tokens: The word tokens. Only one of `vocab_path` and `tokens` should be provided.
            max_len: The maximum sequence length. Sequence larger than this will be truncated.
            lowercase: A bool, whether to lowercase the word tokens.
            bos_token: The begin-of-sentence token.
            eos_token: The end-of-sentence token.
            unk_token: The token indicating unknown word.
            reverse: A bool, whether to reverse the sequence or not.
        """
        if not ((vocab_path is None) ^ (tokens is None)):
            raise ValueError("Either `vocab_path` or `tokens` should be provided.")
        this_locals = copy.copy(locals())
        if tokens is None:
            with tf.io.gfile.GFile(vocab_path, "r") as fp:
                tokens = [line.strip() for line in fp]
            this_locals["tokens"] = tokens
            this_locals["vocab_path"] = None
        self._params = extract_constructor_params(this_locals, verbose=False)
        # extract tokens
        cleaned_tokens = []
        for t in tokens:
            t = t.strip()
            if ((t.startswith("'") and t.endswith("'"))
                or (t.startswith('"') and t.endswith('"'))):
                word = t[1:-1]
            else:
                word = t.strip().split()[0].strip()
            if word:
                cleaned_tokens.append(word)
        assert unk_token, "must provide `unk_token`"
        extra_tokens = [unk_token]
        # add bos
        assert bos_token != unk_token
        extra_tokens.append(bos_token)
        # add eos
        assert eos_token != unk_token != bos_token
        while eos_token in cleaned_tokens:
            eos_token += str(random.choice(list(range(0, 10))))
        extra_tokens.append(eos_token)
        self.vocab = Vocab(tokens=cleaned_tokens, extra_tokens=extra_tokens,
                           lowercase=lowercase)
        self.max_len = max_len
        self.eos_id = self.vocab.map_token_to_id(eos_token)
        self.bos_id = self.vocab.map_token_to_id(bos_token)
        self.unk_id = self.vocab.map_token_to_id(unk_token)
        self.reverse = reverse
        self.delimiter = delimiter

    @property
    def meta_data(self):
        return {
            "vocab_size": self.vocab.vocab_size,
            "eos_id": self.eos_id,
            "bos_id": self.bos_id,
            "unk_id": self.unk_id,
            "pad_id": self.eos_id,
        }

    def get_config(self):
        return self._params

    def map_token_to_id(self, text, return_str=False,
                        with_bos=False, with_eos=True):
        """ Map word tokens to id list

        Args:
            text: a string of a list of string tokens
            return_str: a bool, whether to return a string or not (a list).
            with_bos: a bool, whether to automatically plus bos
                token at the front or not.
            with_eos: a bool, whether to automatically plus eos
                token at the end or not.

        Returns: A list of word ids or a `delimiter` joined string.
        """
        if isinstance(text, str):
            text = text.strip().split()
        assert isinstance(text, list), (type(text))
        token_ids = self.vocab.map_token_to_id(text, unknown_default=self.unk_id)
        if self.reverse:
            token_ids = token_ids[::-1]
        if with_bos:
            token_ids = [self.bos_id] + token_ids
        if with_eos:
            token_ids += [self.eos_id]
        if return_str:
            return self.delimiter.join(
                [str(x) for x in token_ids])
        return token_ids

    def map_id_to_token(self, text, return_str=False,
                        reverse=True):
        """ Map token ids to token string

        Args:
            text: a string or a list of word token ids
            return_str: a bool, whether to return a string or not (a list).
            reverse: a bool, whether to recover the 'reverse' operation
                at `map_token_to_id` method.

        Returns:
            A `delimiter` joined string or a list of word tokens.
        """
        if isinstance(text, str):
            text = text.strip().split()
        text = [int(x) for x in text]
        if text[0] == self.bos_id:
            text = text[1:]
        try:
            eos_pos = text.index(self.eos_id)
            text = text[:eos_pos]
        except ValueError:
            pass
        token_list = self.vocab.map_id_to_token(text)
        if reverse and self.reverse:
            token_list = token_list[::-1]
        if return_str:
            return self.delimiter.join(token_list)
        return token_list
