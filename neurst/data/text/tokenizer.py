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
from abc import ABCMeta, abstractmethod

import six
from absl import logging


@six.add_metaclass(ABCMeta)
class Tokenizer(object):
    """ Base class for text tokenizer. """
    REGISTRY_NAME = "tokenizer"

    def __init__(self, language, glossaries=None, **kwargs):
        """ Initializes the tokenizer. """
        _ = kwargs
        self._language = language
        self._glossaries = glossaries
        if self._glossaries is None:
            self._glossaries = []
        if len(self._glossaries) > 0:
            logging.info("WARNING: note that not all tokenizers support glossaries.")

    @property
    def language(self):
        """ The language. """
        return self._language

    def init_subtokenizer(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def tokenize(self, text, return_str=False):
        """ Tokenizes the text. """
        raise NotImplementedError

    @abstractmethod
    def detokenize(self, text, return_str=True):
        """ Detokenizes the text. """
        raise NotImplementedError

    @staticmethod
    def _convert_to_str(text, delimiter=" "):
        """ Converts the `text` to a string.

        Args:
            text: A string or a list of string.
            delimiter: The delimater for join.

        Returns: A string.
        """
        if isinstance(text, str):
            return " ".join(text.strip().split())
        elif isinstance(text, list):
            return delimiter.join(text)
        else:
            raise ValueError("Not supported type of text: {}".format(type(text)))

    @staticmethod
    def _convert_to_list(text):
        """ Converts the `text` to a list of string.

        Args:
            text: A string or a list of string.

        Returns: A list of string.
        """
        if isinstance(text, str):
            return text.strip().split()
        elif isinstance(text, list):
            return text
        else:
            raise ValueError("Not supported type of text: {}".format(type(text)))

    @staticmethod
    def _output_wrapper(text, return_str, delimiter=" "):
        """ Converts the output `text` to string if `return_str` is True,
            else converters it to a list.

        Args:
            text: A string or a list of string.
            return_str: A bool, whether to convert to string.
            delimiter: The delimiter string for join.

        Returns:
            A string or a list of string according to `return_str`.
        """
        if isinstance(text, str):
            if return_str:
                return text
            else:
                return text.split()
        elif isinstance(text, list):
            if return_str:
                return delimiter.join(text)
            else:
                return text
        else:
            raise ValueError("Not supported type of text: {}".format(type(text)))

    @staticmethod
    def cjk_deseg(text):
        """ Desegment function for Chinese, Japanese and Korean.

        Args:
            text: A string.

        Returns:
            The desegmented string.
        """

        def _strip(matched):
            return matched.group(1).strip()

        char_space_pattern1 = r"([\u2E80-\u9FFF\uA000-\uA4FF\uAC00-\uD7FF\uF900-\uFAFF]\s+)"
        char_space_pattern2 = r"(\s+[\u2E80-\u9FFF\uA000-\uA4FF\uAC00-\uD7FF\uF900-\uFAFF])"

        res = re.sub(char_space_pattern1, _strip, text)
        res = re.sub(char_space_pattern2, _strip, res)
        # no leading space
        res = re.sub(r'^\s+', r'', res)

        # no trailing space
        res = re.sub(r'\s+$', r'', res)
        return res
