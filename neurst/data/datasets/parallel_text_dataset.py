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
from abc import ABCMeta, abstractmethod

import six
import tensorflow as tf
from absl import logging

from neurst.data.datasets import register_dataset
from neurst.data.datasets.dataset import TFRecordDataset
from neurst.data.datasets.text_gen_dataset import TextGenDataset
from neurst.utils.compat import DataStatus
from neurst.utils.flags_core import Flag


@six.add_metaclass(ABCMeta)
class AbstractParallelDataset(TextGenDataset):
    """ The abstract dataset for parallel text.
    The element spec must be
        {
            'feature': tf.TensorSpec(shape=(None,), dtype=tf.int64),
            'label': tf.TensorSpec(shape=(None,), dtype=tf.int64)
         }
    """

    def __init__(self):
        self._sources = None
        super(AbstractParallelDataset, self).__init__()

    @property
    @abstractmethod
    def status(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def build_iterator(self, map_func=None, shard_id=0, total_shards=1):
        """ Returns the iterator of the dataset.

        Args:
            map_func: A function mapping a dataset element to another dataset element.
            shard_id: Generator yields on the `shard_id`-th shard of the whole dataset.
            total_shards: The number of total shards.
        """
        raise NotImplementedError

    @property
    def sources(self):
        """ Returns a list of source texts. """
        return self._sources


@register_dataset("parallel_text")
class ParallelTextDataset(AbstractParallelDataset):

    def __init__(self, args):
        """ Initializes the dataset. """
        super(ParallelTextDataset, self).__init__()
        self._src_file = args["src_file"]
        assert self._src_file, "`src_file` must be provided for ParallelTextDataset."
        self._trg_file = args["trg_file"]
        self._data_is_processed = args["data_is_processed"]

    @staticmethod
    def class_or_method_args():
        return [
            Flag("src_file", dtype=Flag.TYPE.STRING, help="The source text file"),
            Flag("trg_file", dtype=Flag.TYPE.STRING, help="The target text file"),
            Flag("data_is_processed", dtype=Flag.TYPE.BOOLEAN,
                 help="Whether the text data is already processed."),
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
            fsrc = tf.io.gfile.GFile(self._src_file)
            ftrg = None if self._trg_file is None else tf.io.gfile.GFile(self._trg_file)
            n = 0
            for src in fsrc:
                n += 1
                data = {"feature": src.strip()}
                if ftrg is not None:
                    data["label"] = ftrg.readline().strip()
                if total_shards > 1:
                    if n < range_begin:
                        continue
                    if n >= range_end:
                        break
                if map_func is not None:
                    data = map_func(data)
                yield data
            fsrc.close()
            if ftrg is not None:
                ftrg.close()

        return gen

    @property
    def sources(self):
        """ Returns a list of sources. """
        if self._sources is None and self._src_file:
            with tf.io.gfile.GFile(self._src_file) as fp:
                self._sources = [line.strip() for line in fp]
        return self._sources

    @property
    def targets(self):
        """ Returns a list of targets. """
        if self._targets is None and self._trg_file:
            with tf.io.gfile.GFile(self._trg_file) as fp:
                self._targets = [line.strip() for line in fp]
        return self._targets


@register_dataset("parallel_tfrecord")
class ParallelTFRecordDataset(TFRecordDataset, AbstractParallelDataset):

    @property
    def status(self):
        return DataStatus.PROJECTED

    @property
    def fields(self):
        return {"feature": tf.io.VarLenFeature(tf.int64),
                "label": tf.io.VarLenFeature(tf.int64)}
