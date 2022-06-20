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
import os

from neurst.data.data_pipelines import register_data_pipeline
from neurst.data.data_pipelines.data_pipeline import DataPipeline
from neurst.data.text.huggingface_tokenizer import HuggingFaceTokenizer
from neurst.data.text.vocab import PaddingMode, Vocab
from neurst.utils.converters.openai_gpt2 import OpenAIGPT2


@register_data_pipeline("gpt2")
class GPT2DataPipeline(DataPipeline, Vocab):

    def __init__(self,
                 language="en",
                 tokens=None,
                 vocab_path=None,
                 **kwargs):
        """ Initializes the data pipeline from OpenAI released GPT-2.

        Args:
            language: The language.
            tokens: A list of word tokens.
            vocab_path: The path to the vocabulary file.
        """
        if tokens is None and vocab_path is None:
            path = OpenAIGPT2.download("117M")
            vocab_path = os.path.join(path, "encoder.json")
        Vocab.__init__(self, Vocab.load_tokens(vocab_path, tokens), lowercase=False)
        DataPipeline.__init__(self, language=language, tokens=self.tokens, vocab_path=None, **kwargs)
        self._language = language
        self._tokenizer = HuggingFaceTokenizer(language=language)
        self._tokenizer.init_subtokenizer("gpt2")
        self._eos_id = Vocab.map_token_to_id(self, "<|endoftext|>")

    @property
    def meta(self):
        return {
            "vocab_size": self.vocab_size,
            "eos_id": self._eos_id,
            "pad_id": self._eos_id,
            "bos_id": self._eos_id,
            "padding_mode": PaddingMode.EOS_AS_PADDING,
            "language": self._language
        }

    def preprocess(self, input):
        input = DataPipeline.text_pre_normalize(self, self._language, input, is_processed=False)
        return self._tokenizer.tokenize(input, return_str=True)

    def postprocess(self, input):
        return self._tokenizer.detokenize(input, return_str=True)

    def encode(self, input, is_processed=False):
        """ Process one data sample.

        Args:
            input: A text string.
            is_processed: Whether the data sample is already processed.

        Returns:
            A list of generated token IDs.
        """
        if not is_processed:
            input = self.preprocess(input)
        if isinstance(input, str):
            input = input.strip().split()
        token_ids = [x for x in Vocab.map_token_to_id(self, input) if x is not None]
        return token_ids + [self._eos_id]

    def decode(self, input):
        """ Recover one data sample.

        Args:
            input: A list of token ids, the output of neural model.

        Returns:
            A string, the recovered text.
        """
        try:
            eos_pos = input.index(self._eos_id)
            input = input[:eos_pos]
        except ValueError:
            pass
        output = Vocab.map_id_to_token(self, input)
        return self.postprocess(output)
