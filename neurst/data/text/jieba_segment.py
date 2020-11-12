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
from absl import logging

from neurst.data.text import register_tokenizer
from neurst.data.text.tokenizer import Tokenizer


@register_tokenizer
class Jieba(Tokenizer):

    def __init__(self, language="zh", glossaries=None):
        super(Jieba, self).__init__(
            language=language, glossaries=glossaries)
        if self._glossaries and len(self._glossaries) > 0:
            logging.info("WARNING: now `glossaries` has no effect on Jieba.")
        try:
            import jieba
            self._cut_fn = jieba.lcut
        except ImportError:
            raise ImportError('Please install jieba with: pip3 install jieba')

    def tokenize(self, text, return_str=False):
        return self._output_wrapper(self._cut_fn(self._convert_to_str(text)),
                                    return_str=return_str)

    def detokenize(self, text, return_str=True):
        return self._output_wrapper(
            self.cjk_deseg(self._convert_to_str(text)), return_str=return_str)
