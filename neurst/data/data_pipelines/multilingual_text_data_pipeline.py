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
import yaml

from neurst.data.data_pipelines import register_data_pipeline
from neurst.data.data_pipelines.data_pipeline import DataPipeline
from neurst.data.text.spm import SentencePiece
from neurst.data.text.vocab import PaddingMode, Vocab


@register_data_pipeline("multilingual_text")
class MultilingualTextDataPipeline(DataPipeline, Vocab):

    def __init__(self,
                 vocab_path,
                 spm_model,
                 languages,
                 reverse_sequence=False,
                 **kwargs):
        """ Initializes the data pipeline for text data.

        Args:
            vocab_path: The path to the vocabulary file, or a list of word tokens.
            spm_model: The path to the sentence piece model.
            languages: A list of languages. The corresponding language tags will automatically
                append to the vocabulary.
            reverse_sequence: A bool, whether to reverse the sequence.
        """
        DataPipeline.__init__(self, vocab_path=vocab_path, languages=languages,
                              reverse_sequence=reverse_sequence, **kwargs)
        self._reverse_sequence = reverse_sequence
        self._tokenizer = SentencePiece()
        self._tokenizer.init_subtokenizer(spm_model)
        if isinstance(vocab_path, list):
            tokens = Vocab.load_tokens(tokens=vocab_path)
        else:
            tokens = Vocab.load_tokens(vocab_path=vocab_path)
        if isinstance(languages, str):
            languages = yaml.load(languages, Loader=yaml.FullLoader)
        assert isinstance(languages, list), (
            f"`languages` must be a list of strings, but got {languages}")
        lang2tags = {}
        for lang in languages:
            lang2tags[lang] = Vocab.get_unique(tokens, "<" + lang + ">")
        unk_token = Vocab.get_unique(tokens, "<UNK>")
        bos_token = Vocab.get_unique(tokens, "<SEQ_BEG>")
        eos_token = Vocab.get_unique(tokens, "<SEQ_END>")
        assert unk_token != bos_token != eos_token
        Vocab.__init__(self, tokens, [unk_token, bos_token, eos_token] + list(lang2tags.values()),
                       lowercase=False)
        self._eos_id = Vocab.map_token_to_id(self, eos_token)
        self._bos_id = Vocab.map_token_to_id(self, bos_token)
        self._unk_id = Vocab.map_token_to_id(self, unk_token)
        self._lang_ids = {lang: Vocab.map_token_to_id(self, lang2tags[lang])
                          for lang in languages}

    @property
    def meta(self):
        return {
            "lang2id": self._lang_ids,
            "vocab_size": self.vocab_size,
            "eos_id": self._eos_id,
            "bos_id": self._bos_id,
            "unk_id": self._unk_id,
            "pad_id": self._eos_id,
            "padding_mode": PaddingMode.EOS_AS_PADDING
        }

    def preprocess(self, input):
        input = DataPipeline.text_pre_normalize(self, "en", input, is_processed=False)
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
        token_ids = Vocab.map_token_to_id(self, input, unknown_default=self._unk_id)
        if self._reverse_sequence:
            token_ids = token_ids[::-1]
        return token_ids + [self._eos_id]

    def decode(self, input):
        """ Recover one data sample.

        Args:
            input: A list of token ids, the output of neural model.

        Returns:
            A string, the recovered text.
        """
        input = [int(x) for x in input]
        if input[0] == self._bos_id or input[0] in self._lang_ids.values():
            input = input[1:]
        try:
            eos_pos = input.index(self._eos_id)
            input = input[:eos_pos]
        except ValueError:
            pass
        token_list = Vocab.map_id_to_token(self, input)
        if self._reverse_sequence:
            token_list = token_list[::-1]
        return self.postprocess(token_list)
