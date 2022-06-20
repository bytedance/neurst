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

from neurst.data.data_pipelines import register_data_pipeline
from neurst.data.data_pipelines.data_pipeline import DataPipeline
from neurst.data.text import build_tokenizer
from neurst.data.text.vocab import PaddingMode, Vocab


@register_data_pipeline(["simple_text", "TranscriptDataPipeline", "transcript_data_pipeline"])
class TextDataPipeline(DataPipeline, Vocab):

    def __init__(self,
                 vocab_path,
                 language="en",
                 tokenizer=None,
                 subtokenizer=None,
                 subtokenizer_codes=None,
                 glossaries=None,
                 reverse_sequence=False,
                 bos_id=None,
                 eos_id=None,
                 unk_id=None,
                 pad_id=None,
                 **kwargs):
        """ Initializes the data pipeline for text data.

        Args:
            language: The language.
            vocab_path: The path to the vocabulary file, or a list of word tokens.
            tokenizer: The tokenizer name.
            subtokenizer: The name of tokenizer for subword encoding.
            subtokenizer_codes: The subword codes.
            glossaries: The glossaries that will not be split by tokenizer/subtokenizer.
            reverse_sequence: A bool, whether to reverse the sequence.
        """
        DataPipeline.__init__(self, vocab_path=vocab_path, language=language,
                              tokenizer=tokenizer, subtokenizer=subtokenizer,
                              subtokenizer_codes=subtokenizer_codes,
                              glossaries=glossaries,
                              reverse_sequence=reverse_sequence,
                              **kwargs)
        self._language = language
        self._reverse_sequence = reverse_sequence
        self._tokenizer = build_tokenizer(tokenizer, language=language, glossaries=glossaries)
        self._subtokenizer = None
        self._subtokenizer = build_tokenizer(
            subtokenizer, language=language, glossaries=glossaries, vocabulary=vocab_path)
        if self._subtokenizer is not None:
            if subtokenizer_codes is None:
                logging.info("No codes provided for subtokenizer: {}. "
                             "We assume this was done on purpose.".format(subtokenizer))
            else:
                self._subtokenizer.init_subtokenizer(subtokenizer_codes)
        if isinstance(vocab_path, list):
            tokens = Vocab.load_tokens(tokens=vocab_path)
        else:
            tokens = Vocab.load_tokens(vocab_path=vocab_path)
        unk_token = Vocab.get_unique(tokens, "<UNK>") if unk_id is None else tokens[unk_id]
        bos_token = Vocab.get_unique(tokens, "<SEQ_BEG>") if bos_id is None else tokens[bos_id]
        eos_token = Vocab.get_unique(tokens, "<SEQ_END>") if eos_id is None else tokens[eos_id]
        pad_token = eos_token if pad_id is None else tokens[pad_id]
        assert unk_token != bos_token != eos_token
        Vocab.__init__(self, tokens, [unk_token, bos_token, eos_token, pad_token], lowercase=False)
        self._eos_id = Vocab.map_token_to_id(self, eos_token)
        self._bos_id = Vocab.map_token_to_id(self, bos_token)
        self._unk_id = Vocab.map_token_to_id(self, unk_token)
        self._pad_id = Vocab.map_token_to_id(self, pad_token)

    @property
    def meta(self):
        return {
            "language": self._language,
            "vocab_size": self.vocab_size,
            "eos_id": self._eos_id,
            "bos_id": self._bos_id,
            "unk_id": self._unk_id,
            "pad_id": self._eos_id,
            "padding_mode": (PaddingMode.EOS_AS_PADDING
                             if self._eos_id == self._pad_id else PaddingMode.DEFAULT)
        }

    def preprocess(self, input):
        input = DataPipeline.text_pre_normalize(self, self._language, input, is_processed=False)
        if self._tokenizer:
            input = self._tokenizer.tokenize(input, return_str=True)
        if self._subtokenizer:
            input = self._subtokenizer.tokenize(input, return_str=True)
        return input

    def postprocess(self, input):
        output = input
        if self._subtokenizer is not None:
            output = self._subtokenizer.detokenize(output, return_str=True)
        if self._tokenizer is not None:
            output = self._tokenizer.detokenize(output, return_str=True)
        return output

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
            input = input.split()
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
        if input[0] == self._bos_id:
            input = input[1:]
        try:
            eos_pos = input.index(self._eos_id)
            input = input[:eos_pos]
        except ValueError:
            pass
        token_list = Vocab.map_id_to_token(self, input)
        if self._reverse_sequence:
            token_list = token_list[::-1]
        return self.postprocess(" ".join(token_list))
