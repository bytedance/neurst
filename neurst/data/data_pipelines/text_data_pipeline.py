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
import copy

from absl import logging

from neurst.data.data_pipelines import register_data_pipeline
from neurst.data.data_pipelines.data_pipeline import DataPipeline
from neurst.data.text import build_tokenizer
from neurst.data.text.symbols_mapper import SymbolsMapper


@register_data_pipeline("simple_text")
class TextDataPipeline(DataPipeline):

    def __init__(self,
                 vocab_path,
                 language="en",
                 tokenizer=None,
                 subtokenizer=None,
                 subtokenizer_codes=None,
                 glossaries=None,
                 reverse_sequence=False,
                 **kwargs):
        """ Initializes the data pipeline for text data.

        Args:
            language: The language.
            vocab_path: The path to the vocabulary file.
            tokenizer: The tokenizer name.
            subtokenizer: The name of tokenizer for subword encoding.
            subtokenizer_codes: The subword codes.
            glossaries: The glossaries that will not be split by tokenizer/subtokenizer.
            reverse_sequence: A bool, whether to reverse the sequence.
        """
        super(TextDataPipeline, self).__init__(
            vocab_path=vocab_path,
            language=language,
            tokenizer=tokenizer,
            subtokenizer=subtokenizer,
            subtokenizer_codes=subtokenizer_codes,
            glossaries=glossaries,
            reverse_sequence=reverse_sequence,
            **kwargs)
        self._language = language
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
        self._symbols_mapper = SymbolsMapper(vocab_path=vocab_path, reverse=reverse_sequence)

    @property
    def meta(self):
        meta = copy.deepcopy(self._symbols_mapper.meta_data)
        meta["language"] = self._language
        return meta

    def process(self, input, is_processed=False):
        """ Process one data sample.

        Args:
            input: A text string.
            is_processed: Whether the data sample is already processed.

        Returns:
            A list of generated token IDs.
        """
        if not is_processed:
            if self._tokenizer:
                input = self._tokenizer.tokenize(input)
            if self._subtokenizer:
                input = self._subtokenizer.tokenize(input, return_str=False)
        return self._symbols_mapper.map_token_to_id(
            input, return_str=False, with_bos=False, with_eos=True)

    def recover(self, input):
        """ Recover one data sample.

        Args:
            input: A list of token ids, the output of neural model.

        Returns:
            A string, the recovered text.
        """

        if self._subtokenizer is None:
            output = self._symbols_mapper.map_id_to_token(input, return_str=True)
        else:
            output = self._symbols_mapper.map_id_to_token(input, return_str=False)
            output = self._subtokenizer.detokenize(output, return_str=True)
        if self._tokenizer:
            output = self._tokenizer.detokenize(output, return_str=True)
        return output
