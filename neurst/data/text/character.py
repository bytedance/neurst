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
import re

from neurst.data.text import register_tokenizer
from neurst.data.text.tokenizer import Tokenizer


@register_tokenizer("char")
class Character(Tokenizer):
    CHAR_COMPILER = re.compile(r"([\u2E80-\u9FFF\uA000-\uA4FF\uAC00-\uD7FF\uF900-\uFAFF])")

    def __init__(self, language, glossaries=None):
        super(Character, self).__init__(
            language=language, glossaries=glossaries)

    @staticmethod
    def is_cjk(language):
        return language in ["zh", "ja", "ko"]

    @staticmethod
    def to_character(text, language=None):
        text = Character._convert_to_str(text)
        if Character.is_cjk(language):
            return Character.cjk_to_character(text)
        return " ".join(text)

    @staticmethod
    def cjk_to_character(text):
        """ CJK sentence to character-level.

        Args:
            text: A list of string tokens or a string.

        Returns: A string.
        """
        text = Character._convert_to_str(text)
        res = Character.CHAR_COMPILER.sub(r" \1 ", text)
        # tokenize period and comma unless preceded by a digit
        res = re.sub(r'([^0-9])([\.,])', r'\1 \2 ', res)

        # tokenize period and comma unless followed by a digit
        res = re.sub(r'([\.,])([^0-9])', r' \1 \2', res)

        # tokenize dash when preceded by a digit
        res = re.sub(r'([0-9])(-)', r'\1 \2 ', res)

        # one space only between words
        res = re.sub(r'\s+', r' ', res)

        # no leading space
        res = re.sub(r'^\s+', r'', res)

        # no trailing space
        res = re.sub(r'\s+$', r'', res)
        return res

    def tokenize(self, text, return_str=False):
        return self._output_wrapper(
            self.to_character(text, language=self.language),
            return_str=return_str)

    def detokenize(self, text, return_str=True):
        if not self.is_cjk(self.language):
            raise NotImplementedError(f"detokenize fn in Character "
                                      f"is not implemented for language={self.language}")
        return self._output_wrapper(
            self.cjk_deseg(self._convert_to_str(text)),
            return_str=return_str)
