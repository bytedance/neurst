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
from abc import ABCMeta, abstractmethod

import six
import tensorflow as tf
import yaml
from absl import logging

from neurst.data.dataset_utils import glob_tfrecords
from neurst.data.datasets import register_dataset
from neurst.data.datasets.data_sampler import DataSampler, build_data_sampler
from neurst.data.datasets.dataset import TFRecordDataset
from neurst.data.datasets.text_gen_dataset import TextGenDataset
from neurst.utils.compat import DataStatus
from neurst.utils.flags_core import Flag, ModuleFlag
from neurst.utils.misc import temp_download


@six.add_metaclass(ABCMeta)
class AbstractParallelDataset(TextGenDataset):
    """ The abstract dataset for parallel text.
    The element spec must be
        {
            'feature': tf.TensorSpec(shape=(None,), dtype=tf.int64),
            'label': tf.TensorSpec(shape=(None,), dtype=tf.int64)
         }
    """

    def __init__(self, src_lang=None, trg_lang=None):
        self._sources = None
        self._src_lang = src_lang
        super(AbstractParallelDataset, self).__init__(trg_lang=trg_lang)

    @property
    def src_lang(self):
        return self._src_lang

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
        super(ParallelTextDataset, self).__init__(src_lang=args["src_lang"], trg_lang=args["trg_lang"])
        if args["src_file"] and args["src_file"].startswith("http"):
            self._src_file = temp_download(args["src_file"])
        else:
            self._src_file = args["src_file"]
        assert self._src_file, "`src_file` must be provided for ParallelTextDataset."
        if args["trg_file"] and args["trg_file"].startswith("http"):
            self._trg_file = temp_download(args["trg_file"])
        else:
            self._trg_file = args["trg_file"]
        if args["raw_trg_file"] and args["raw_trg_file"].startswith("http"):
            self._raw_trg_file = temp_download(args["raw_trg_file"])
        else:
            self._raw_trg_file = args["raw_trg_file"]
        self._data_is_processed = args["data_is_processed"]

    @staticmethod
    def class_or_method_args():
        return [
            Flag("src_file", dtype=Flag.TYPE.STRING, help="The source text file"),
            Flag("trg_file", dtype=Flag.TYPE.STRING, help="The target text file"),
            Flag("raw_trg_file", dtype=Flag.TYPE.STRING, help="The raw target text file"),
            Flag("data_is_processed", dtype=Flag.TYPE.BOOLEAN,
                 help="Whether the text data is already processed."),
            Flag("src_lang", dtype=Flag.TYPE.STRING, default=None, help="The source language"),
            Flag("trg_lang", dtype=Flag.TYPE.STRING, default=None, help="The target language"),
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
                data = {"feature": " ".join(src.strip().split())}
                if ftrg is not None:
                    data["label"] = " ".join(ftrg.readline().strip().split())
                if self.src_lang is not None:
                    data["src_lang"] = self.src_lang
                if self.trg_lang is not None:
                    data["trg_lang"] = self.trg_lang
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

    @property
    def raw_targets(self):
        """ Returns a list of targets. """
        if self._raw_targets is None and self._raw_trg_file:
            with tf.io.gfile.GFile(self._raw_trg_file) as fp:
                self._raw_targets = [line.strip() for line in fp]
        return self._raw_targets


@register_dataset("multiple_parallel_text")
class MultipleParallelTextDataset(AbstractParallelDataset):
    """ For unbalanced datasets. """

    def __init__(self, args):
        """ Initializes the dataset. """
        super(MultipleParallelTextDataset, self).__init__(
            src_lang=args["src_lang"], trg_lang=args["trg_lang"])
        self._data_files = args["data_files"]
        if isinstance(self._data_files, str):
            self._data_files = yaml.load(args["data_files"], Loader=yaml.FullLoader)
        assert isinstance(self._data_files, dict)
        self._data_is_processed = args["data_is_processed"]
        self._data_sampler = build_data_sampler(args)

    @staticmethod
    def class_or_method_args():
        return [
            Flag("data_files", dtype=Flag.TYPE.STRING,
                 help="A dict of parallel data files. The key is the dataset name while "
                      "the value is a dict containing `src_file` and `trg_file`."),
            Flag("data_is_processed", dtype=Flag.TYPE.BOOLEAN,
                 help="Whether the text data is already processed."),
            Flag("src_lang", dtype=Flag.TYPE.STRING, default=None, help="The source language"),
            Flag("trg_lang", dtype=Flag.TYPE.STRING, default=None, help="The target language"),
            ModuleFlag(DataSampler.REGISTRY_NAME, default=None,
                       help="The sampler for unbalanced datasets.")
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

        def get_data(s, t):
            data = {"feature": " ".join(s.strip().split()),
                    "label": " ".join(t.strip().split())}
            if self.src_lang is not None:
                data["src_lang"] = self.src_lang
            if self.trg_lang is not None:
                data["trg_lang"] = self.trg_lang
            if map_func is not None:
                data = map_func(data)
            return data

        def gen():
            fps = dict()
            for k, elem in self._data_files.items():
                fps[k] = (tf.io.gfile.GFile(elem["src_file"]),
                          tf.io.gfile.GFile(elem["trg_file"]))
            n = 0
            if self._data_sampler is None:
                for _, (fsrc, ftrg) in fps.items():
                    for s, t in zip(fsrc, ftrg):
                        n += 1
                        if total_shards > 1:
                            if n < range_begin:
                                continue
                            if n >= range_end:
                                break
                        yield get_data(s, t)
                    fsrc.close()
                    ftrg.close()
            else:
                while True:
                    n += 1
                    choice = self._data_sampler()
                    s = fps[choice][0].readline()
                    t = fps[choice][1].readline()
                    if s == "" or t == "":
                        fps[choice][0].seek(0)
                        fps[choice][1].seek(0)
                        s = fps[choice][0].readline()
                        t = fps[choice][1].readline()
                        assert s and t
                    if total_shards > 1:
                        if n < range_begin:
                            continue
                        if n >= range_end:
                            break
                    yield get_data(s, t)

        return gen


@register_dataset("parallel_tfrecord")
class ParallelTFRecordDataset(TFRecordDataset, AbstractParallelDataset):

    @property
    def status(self):
        return DataStatus.PROJECTED

    @property
    def fields(self):
        return {"feature": tf.io.VarLenFeature(tf.int64),
                "label": tf.io.VarLenFeature(tf.int64)}


@register_dataset
class InMemoryParallelTFRecordDataset(ParallelTFRecordDataset):

    def __init__(self, args):
        new_data_path = f"ram://parallel_text_tfrecord{time.time()}/"
        record_files = glob_tfrecords(args["data_path"])
        total_size = len(record_files)
        for idx, f in enumerate(record_files):
            tf.io.gfile.copy(f, os.path.join(new_data_path, "train-%5.5d-of%5.5d" % (idx, total_size)))
        args["data_path"] = new_data_path
        super(InMemoryParallelTFRecordDataset, self).__init__(args)
