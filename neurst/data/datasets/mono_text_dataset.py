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
import tensorflow as tf
from absl import logging

from neurst.data.datasets import register_dataset
from neurst.data.datasets.text_gen_dataset import TextGenDataset
from neurst.utils.compat import DataStatus
from neurst.utils.flags_core import Flag


@register_dataset(["mono_text", "monolingual_text"])
class MonoTextDataset(TextGenDataset):

    def __init__(self, args):
        """ Initializes the dataset. """
        super(MonoTextDataset, self).__init__(trg_lang=args["data_lang"])
        self._data_file = args["data_file"]
        assert self._data_file, "`data_file` must be provided for MonoTextDataset."
        self._data_is_processed = args["data_is_processed"]

    @staticmethod
    def class_or_method_args():
        return [
            Flag("data_file", dtype=Flag.TYPE.STRING, help="The text file"),
            Flag("data_is_processed", dtype=Flag.TYPE.BOOLEAN,
                 help="Whether the text data is already processed."),
            Flag("data_lang", dtype=Flag.TYPE.STRING, default=None, help="The language of the text."),
        ]

    @property
    def status(self):
        if self._data_is_processed:
            return DataStatus.PROCESSED
        return DataStatus.RAW

    def build_iterator(self, map_func=None, shard_id=0, total_shards=1):
        """ Reads data from files and returns the iterator.

        Args:
            map_func: A function mapping a dataset element to another dataset element.
            shard_id: Generator yields on the `shard_id`-th shard of the whole dataset.
            total_shards: The number of total shards.
        """
        if total_shards > 1:
            total_samples = self.num_samples
            samples_per_part = total_samples // total_shards
            range_begin = samples_per_part * shard_id
            if shard_id == total_shards - 1:
                range_end = total_samples + 1
                logging.info(f"Iterate on dataset from {range_begin} "
                             f"to the end (total {total_samples}).")
            else:
                range_end = range_begin + samples_per_part
                logging.info(f"Iterate on dataset from {range_begin} "
                             f"to {range_end} (total {total_samples}).")

        def gen():
            fp = tf.io.gfile.GFile(self._data_file)
            n = 0
            for src in fp:
                n += 1
                data = {"tokens": " ".join(src.strip().split())}
                if self.trg_lang is not None:
                    data["lang"] = self.trg_lang
                if total_shards > 1:
                    if n < range_begin:
                        continue
                    if n >= range_end:
                        break
                if map_func is not None:
                    data = map_func(data)
                yield data
            fp.close()

        return gen
