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
from neurst.utils.converters.google_bert import GoogleBert


@register_data_pipeline("bert")
class BertDataPipeline(DataPipeline, Vocab):

    def __init__(self,
                 name,
                 language="en",
                 vocab_path=None,
                 tokens=None,
                 **kwargs):
        """ Initializes the data pipeline for text data.

        Args:
            name: The key of the BERT model, for creating the tokenizer and loading vocabulary.
            language: The language.
            tokens: A list of word tokens.
            vocab_path: The path to the vocabulary file.
        """
        if tokens is None and vocab_path is None:
            path = GoogleBert.download(name)
            if path is None:
                raise ValueError(f"Unknown BERT model name={name} for downloading.")
            vocab_path = os.path.join(path, "vocab.txt")
        else:
            if tokens is not None:
                vocab_path = None
            tokens = Vocab.load_tokens(vocab_path, tokens)
            vocab_path = None
            # to handle with customized vocabulary
            for spec_token in ["[UNK]", "[CLS]", "[SEP]", "[MASK]", "[PAD]"]:
                if spec_token not in tokens:
                    tokens.insert(0, spec_token)
            assert tokens[0] == "[PAD]"
        Vocab.__init__(self, Vocab.load_tokens(vocab_path, tokens), lowercase=False)
        DataPipeline.__init__(self, name=name, language=language, tokens=self.tokens,
                              vocab_path=None, **kwargs)
        self._language = language
        self._tokenizer = HuggingFaceTokenizer(language=language)
        self._tokenizer.init_subtokenizer(name)
        self._unk_id = Vocab.map_token_to_id(self, "[UNK]")
        self._pad_id = Vocab.map_token_to_id(self, "[PAD]")
        self._cls_id = Vocab.map_token_to_id(self, "[CLS]")
        self._sep_id = Vocab.map_token_to_id(self, "[SEP]")
        self._mask_id = Vocab.map_token_to_id(self, "[MASK]")

    @property
    def meta(self):
        return {
            "language": self._language,
            "vocab_size": self.vocab_size,
            "pad_id": self._pad_id,
            "cls_id": self._cls_id,
            "unk_id": self._unk_id,
            "sep_id": self._sep_id,
            "mask_id": self._mask_id,
            "bos_id": self._cls_id,
            "padding_mode": PaddingMode.DEFAULT
        }

    def preprocess(self, input):
        text = DataPipeline.text_pre_normalize(self, self._language, input, is_processed=False)
        return self._tokenizer.tokenize(text, return_str=True)

    def postprocess(self, input):
        return self._tokenizer.detokenize(input, return_str=True)

    def decode(self, input):
        raise NotImplementedError("No need to call recover for BertDataPipeline")

    def encode(self, input, is_processed=False):
        """ Process one data sample.

        Args:
            input: A string for text_a or a dict containing text_a and text_b.
            is_processed: Whether the data sample is already processed.

        Returns:
            A list of generated token IDs.
        """

        def _process(text):
            if not is_processed:
                text = self.preprocess(text)
            if isinstance(text, str):
                text = text.strip().split()
            token_ids = Vocab.map_token_to_id(self, text, unknown_default=self._unk_id)
            return token_ids + [self._sep_id]

        return_ids = [self._cls_id]
        if isinstance(input, dict):
            return_ids.extend(_process(input["text_a"]))
            return_ids.extend(_process(input["text_b"]))
        else:
            return_ids.extend(_process(input))
        return return_ids
