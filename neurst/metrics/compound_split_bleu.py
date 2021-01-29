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

import numpy as np

from neurst.metrics import register_metric
from neurst.metrics.bleu import BLEU


@register_metric
class CompoundSplitBleu(BLEU):

    def __init__(self, *args, **kwargs):
        """ Initializes.

        Args:
            language: The language.
        """
        _ = args
        _ = kwargs
        super(CompoundSplitBleu, self).__init__(*args, **kwargs)

    @staticmethod
    def _tokenize(ss, tok_fn, lc=False):
        res = super(CompoundSplitBleu, CompoundSplitBleu)._tokenize(ss, tok_fn, lc)
        if isinstance(res[0], str):
            return [re.sub(r"(\S)-(\S)", r"\1 ##AT##-##AT## \2", x) for x in res]
        return [[re.sub(r"(\S)-(\S)", r"\1 ##AT##-##AT## \2", x) for x in xx] for xx in res]

    def get_value(self, result):
        if isinstance(result, (float, np.float32, np.float64)):
            return result
        if self._flag in result:
            return result[self._flag]
        if self._flag.lower() in result:
            return result[self._flag.lower()]
        return result["compound_split_bleu"]

    def call(self, hypothesis, groundtruth=None):
        """ Returns the BLEU result dict. """
        return {
            "compound_split_bleu": self.tok_bleu(hypothesis, groundtruth),
            "uncased_compound_split_bleu": self.tok_bleu(hypothesis, groundtruth, lc=True)}
