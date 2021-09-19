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
import time

import tensorflow as tf
from absl import logging

from neurst.data.text import register_tokenizer
from neurst.data.text.tokenizer import Tokenizer
from neurst.utils.misc import download_with_tqdm


@register_tokenizer("spm")
class SentencePiece(Tokenizer):

    def __init__(self, language=None, glossaries=None, subtokenizer_codes=None, **kwargs):
        _ = kwargs
        super(SentencePiece, self).__init__(
            language=language, glossaries=glossaries)
        try:
            import sentencepiece
            self._sp = sentencepiece.SentencePieceProcessor()
        except ImportError:
            raise ImportError('Please install SentencePiece with: pip install sentencepiece')
        self._built = False
        self._codes = subtokenizer_codes

    def _lazy_init(self):
        codes = self._codes
        from_local = False
        if codes.startswith("hdfs://"):
            local_path = os.path.join(os.path.dirname(__file__), "spm{}.model".format(time.time()))
            logging.info("Copying spm model: {} to local: {}".format(codes, local_path))
            tf.io.gfile.copy(codes, local_path, overwrite=True)
            codes = local_path
            from_local = True
        elif codes.startswith("http"):
            local_path = os.path.join(os.path.dirname(__file__), "spm{}.model".format(time.time()))
            logging.info("Downloading spm model to local: {}".format(local_path))
            download_with_tqdm(codes, local_path)
            codes = local_path
            from_local = True
        status = self._sp.Load(codes)
        assert status, "Fail to load spm model: {}".format(codes)
        if from_local:
            tf.io.gfile.remove(codes)
        self._built = True

    def init_subtokenizer(self, codes):
        """ Lazily initializes sentence piece. """
        self._codes = codes

    def tokenize(self, text, return_str=False):
        if not self._built:
            self._lazy_init()
            if not self._built:
                raise ValueError("call `init_subtokenizer` at first to initialize the SentencePiece.")
        return self._output_wrapper(
            self._sp.EncodeAsPieces(self._convert_to_str(text)), return_str=return_str)

    def detokenize(self, text, return_str=True):
        if not self._built:
            self._lazy_init()
            if not self._built:
                raise ValueError("call `init_subtokenizer` at first to initialize the SentencePiece.")
        return self._output_wrapper(
            self._sp.DecodePieces(self._convert_to_list(text)), return_str=return_str)
