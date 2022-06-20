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
from sacremoses import MosesPunctNormalizer

from neurst.utils.configurable import extract_constructor_params

PUNC_PATTERN = re.compile(r"[,\.\!\(\);:、\?\-\+=\"/><《》\[\]，。：；「」【】{}`@#\$%\^&\*]")
PUNC_NORMERS = dict()


def lowercase_and_remove_punctuations(language, text, lowercase=True, remove_punctuation=True):
    if lowercase:
        text = text.lower()
    if language not in ["zh", "ja"]:
        if language not in PUNC_NORMERS:
            PUNC_NORMERS[language] = MosesPunctNormalizer(lang=language)
        text = PUNC_NORMERS[language].normalize(text)
        text = text.replace("' s ", "'s ").replace(
            "' ve ", "'ve ").replace("' m ", "'m ").replace("' t ", "'t ").replace("' re ", "'re ")
    if remove_punctuation:
        text = PUNC_PATTERN.sub(" ", text)
    text = " ".join(text.strip().split())
    return text


@six.add_metaclass(ABCMeta)
class DataPipeline(object):
    REGISTRY_NAME = "data_pipeline"

    def __init__(self, **kwargs):
        self._params = extract_constructor_params(locals(), verbose=False)

    def get_config(self) -> dict:
        return self._params

    @property
    @abstractmethod
    def meta(self) -> dict:
        """ The meta data. """
        return {}

    def recover(self, input):
        """ Recovers one data sample. """
        return self.decode(input)

    def process(self, input, is_processed=False):
        return self.encode(input, is_processed)

    @abstractmethod
    def decode(self, input):
        """ Decodes one data sample (from token IDs to sentence). """
        raise NotImplementedError

    @abstractmethod
    def encode(self, input, is_processed=False):
        """ Encodes one data sample to token IDs. """
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, input):
        """ Preprocesses one data sample (e.g., tokenize). """
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, input):
        """ Postprocesses one data sample (e.g., detokenize). """
        raise NotImplementedError

    def text_pre_normalize(self, language, input, is_processed=False):
        if is_processed:
            return input
        output = lowercase_and_remove_punctuations(language, input,
                                                   self._params.get("lowercase", False),
                                                   self._params.get("remove_punctuation", False))
        return output
