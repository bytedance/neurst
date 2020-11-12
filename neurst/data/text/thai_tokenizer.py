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
""" Defines thai tokenizer."""
from neurst.data.text import Tokenizer, register_tokenizer


@register_tokenizer
class ThaiTokenizer(Tokenizer):

    def __init__(self, language="th", glossaries=None):
        """ Initializes. """
        _ = language
        language = "th"
        try:
            from thai_segmenter import tokenize as thai_tokenize
            self._thai_tokenize = thai_tokenize
        except ImportError:
            raise ImportError('Please install Thai tokenizer with: pip install thai-segmenter')
        self._thai_tokenize("")
        super(ThaiTokenizer, self).__init__(language=language, glossaries=glossaries)

    def tokenize(self, text, return_str=False):
        """ Tokenize a text. """
        res = self._thai_tokenize(self._convert_to_str(text))
        if return_str:
            res = " ".join(res)
        return res

    def detokenize(self, words, return_str=True):
        """ Recovers the result of `tokenize(words)`.

        Args:
            words: A list of strings, i.e. tokenized text.
            return_str: returns a string if True, a list of tokens otherwise.

        Returns: The recovered sentence string.
        """
        if isinstance(words, str):
            words = words.strip().split()
        if return_str:
            words = "".join(words)
        return words
