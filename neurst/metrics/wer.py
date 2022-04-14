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
import numpy as np

from neurst.data.data_pipelines.data_pipeline import lowercase_and_remove_punctuations
from neurst.data.text.moses_tokenizer import MosesTokenizer
from neurst.metrics import register_metric
from neurst.metrics.metric import Metric


def _wer(ref, hypo):
    errors = np.zeros([len(ref) + 1, len(hypo) + 1, 3])
    errors[0, :, 1] = np.arange(len(hypo) + 1)
    errors[:, 0, 2] = np.arange(len(ref) + 1)
    substitution = np.array([1, 0, 0])
    insertion = np.array([0, 1, 0])
    deletion = np.array([0, 0, 1])
    for r, ref in enumerate(ref):
        for d, dec in enumerate(hypo):
            errors[r + 1, d + 1] = min((
                errors[r, d] + (ref != dec) * substitution,
                errors[r + 1, d] + insertion,
                errors[r, d + 1] + deletion), key=np.sum)

    return tuple(errors[-1, -1])


@register_metric(["cer", "CER", "Cer"])
class Wer(Metric):
    def __init__(self, language="en", *args, **kwargs):
        _ = args
        _ = kwargs
        self._tokenizer = MosesTokenizer(language)
        self._language = language
        self._metric_key = "CER" if language in ["zh", "ja"] else "WER"

        super(Wer, self).__init__()

    def set_groundtruth(self, groundtruth):
        """ Setup inside groundtruth.

        Args:
            groundtruth: A list of references,
                [sent0_ref, sent1_ref, ...]
        """
        self._references = [lowercase_and_remove_punctuations(
            self._language, self._tokenizer.tokenize(x, return_str=True),
            lowercase=True, remove_punctuation=True) for x in groundtruth]

    def greater_or_eq(self, result1, result2):
        return self.get_value(result1) <= self.get_value(result2)

    def get_value(self, result):
        if isinstance(result, (float, np.float32, np.float64)):
            return result
        return result[self._metric_key]

    def call(self, hypothesis, groundtruth=None):
        """ Calculate wer

        Args:
            hypothesis: A list of hypothesis texts.
            groundtruth: A list of reference texts.

        Returns:
            A tuple(wer, substitutions, insertions, deletions)
        """
        if groundtruth is None:
            groundtruth = self._references
        else:
            groundtruth = [lowercase_and_remove_punctuations(
                self._language, self._tokenizer.tokenize(x, return_str=True),
                lowercase=True, remove_punctuation=True) for x in groundtruth]
        hypothesis = [lowercase_and_remove_punctuations(
            self._language, self._tokenizer.tokenize(x, return_str=True),
            lowercase=True, remove_punctuation=True) for x in hypothesis]
        substitutions = 0
        insertions = 0
        deletions = 0
        numwords = 0

        for lref, lout in zip(groundtruth, hypothesis):
            # read the reference and  output
            if self._language in ["zh", "ja"]:
                reftext = [w for w in "".join(lref.strip().split())]
                output = [w for w in "".join(lout.strip().split())]
            else:
                reftext, output = lref.strip().split(), lout.strip().split()

            # compare output to reference
            s, i, d = _wer(reftext, output)
            substitutions += s
            insertions += i
            deletions += d
            numwords += len(reftext)

        substitutions /= numwords
        deletions /= numwords
        insertions /= numwords
        error = substitutions + deletions + insertions
        return {
            f"{self._metric_key}": error * 100.,
            f"{self._metric_key}-substitutions": substitutions * 100.,
            f"{self._metric_key}-insertions": insertions * 100.,
            f"{self._metric_key}-deletions": deletions * 100.
        }
