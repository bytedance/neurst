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
from neurst.data.text import register_tokenizer
from neurst.data.text.tokenizer import Tokenizer


@register_tokenizer("moses")
class MosesTokenizer(Tokenizer):

    def __init__(self, language, glossaries=None,
                 aggressive_dash_splits=True, escape=False):
        super(MosesTokenizer, self).__init__(
            language=language, glossaries=glossaries)
        self._aggressive_dash_splits = aggressive_dash_splits
        self._escape = escape
        try:
            from sacremoses import MosesDetokenizer as MDetok
            from sacremoses import MosesTokenizer as MTok
            self._tok = MTok(lang=self.language)
            self._detok = MDetok(lang=self.language)
        except ImportError:
            raise ImportError('Please install Moses tokenizer with: pip3 install sacremoses')

    def tokenize(self, text, return_str=False):
        return self._tok.tokenize(self._convert_to_str(text),
                                  aggressive_dash_splits=self._aggressive_dash_splits,
                                  return_str=return_str,
                                  escape=self._escape,
                                  protected_patterns=self._glossaries)

    def detokenize(self, text, return_str=True):
        return self._detok.detokenize(self._convert_to_list(text),
                                      return_str=return_str,
                                      unescape=True)
