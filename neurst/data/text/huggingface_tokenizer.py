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
import traceback

from absl import logging

from neurst.data.text import register_tokenizer
from neurst.data.text.tokenizer import Tokenizer

try:
    from transformers import AutoTokenizer
except ImportError:
    pass


@register_tokenizer("huggingface")
class HuggingFaceTokenizer(Tokenizer):

    def __init__(self, language, glossaries=None, subtokenizer_codes=None, **kwargs):
        super(HuggingFaceTokenizer, self).__init__(
            language=language, glossaries=glossaries, **kwargs)
        try:
            from transformers import AutoTokenizer
            _ = AutoTokenizer
        except ImportError:
            raise ImportError('Please install transformers with: pip3 install transformers')
        self._built = False
        self._codes = subtokenizer_codes

    def init_subtokenizer(self, codes):
        """ Lazily initializes huggingface tokenizer. """
        self._codes = codes

    def _lazy_init(self):
        codes = self._codes
        success = False
        fail_times = 0
        while not success:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(codes)
                success = True
            except Exception as e:
                fail_times += 1
                logging.info("AutoTokenizer.from_pretrained fails for {0} times".format(fail_times))
                if fail_times >= 5:
                    logging.info(traceback.format_exc())
                    raise e

        self._built = True

    def tokenize(self, text, return_str=False):
        if not self._built:
            self._lazy_init()
            if not self._built:
                raise ValueError("call `init_subtokenizer` at first to initialize the tokenizer.")
        return self._output_wrapper(
            self._tokenizer.tokenize(self._convert_to_str(text)), return_str=return_str)

    def detokenize(self, text, return_str=True):
        if not self._built:
            self._lazy_init()
            if not self._built:
                raise ValueError("call `init_subtokenizer` at first to initialize the tokenizer.")
        return self._output_wrapper(
            self._tokenizer.convert_tokens_to_string(self._convert_to_list(text)), return_str=return_str)
