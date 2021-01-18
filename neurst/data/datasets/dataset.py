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

from neurst.data.dataset_utils import load_tfrecords
from neurst.utils.compat import get_distributed_worker_setting
from neurst.utils.flags_core import Flag
from neurst.utils.misc import to_numpy_or_python_type


@six.add_metaclass(ABCMeta)
class Dataset(object):
    """ Abstract dataset class for handling data io. """
    REGISTRY_NAME = "dataset"

    def __init__(self):
        """ Initializes the dataset."""
        self._count = None

    @property
    @abstractmethod
    def status(self):
        raise NotImplementedError

    def build(self, auto_shard=False, map_func=None, map_output_dtypes=None,
              shuffle=True) -> tf.data.Dataset:
        """ Reads data from files and build the tf dataset.

        Args:
            auto_shard: Whether to automatically shard the dataset.
            map_func: A function mapping a dataset element to another dataset element.
            map_output_dtypes: A list/tuple of dtypes after applying `map_func`.
            shuffle: Whether to shuffle the TF records files.

        Returns: A tf.data.Dataset.
        """
        dataset = tf.data.Dataset.from_generator(self.build_iterator(map_func),
                                                 output_types=map_output_dtypes)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset = dataset.with_options(options)

        if auto_shard:
            worker_id, num_workers, strategy = get_distributed_worker_setting()
            if num_workers > 1 and strategy in ["horovod", "byteps"]:
                logging.info("Shard %d of the whole dataset(total %d workers).", worker_id, num_workers)
                dataset = dataset.shard(num_workers, worker_id)
        return dataset

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
    def num_samples(self):
        if self._count is None:
            cnt = 0
            for _ in self.build_iterator(map_func=None)():
                cnt += 1
            self._count = cnt
        return self._count


@six.add_metaclass(ABCMeta)
class TFRecordDataset(Dataset):
    """ Abstract dataset class for handling data io. """

    def __init__(self, args):
        super(TFRecordDataset, self).__init__()
        self._data_path = args["data_path"]
        self._shuffle_dataset = args["shuffle_dataset"]

    @staticmethod
    def class_or_method_args():
        return [
            Flag("data_path", dtype=Flag.TYPE.STRING, help="The path to TF records."),
            Flag("shuffle_dataset", dtype=Flag.TYPE.BOOLEAN,
                 help="Whether to shuffle the TF records files. "
                      "Note that some parts may be lost under MultiWorkerMirroredStrategy if set True."),
        ]

    @property
    @abstractmethod
    def fields(self) -> dict:
        """ The fields of the TF Records, e.g. {"feature": tf.io.VarLenFeature(tf.int64)}. """
        raise NotImplementedError

    @property
    @abstractmethod
    def status(self):
        raise NotImplementedError

    def build(self, auto_shard=False, map_func=None, map_output_dtypes=None,
              shuffle=True) -> tf.data.Dataset:
        """ Reads data from files and build the tf dataset.

        Args:
            auto_shard: Whether to automatically shard the dataset.
            map_func: A function mapping a dataset element to another dataset element.
            map_output_dtypes: A list/tuple of dtypes after applying `map_func`.
            shuffle: Whether to shuffle the TF records files.

        Returns: A tf.data.Dataset.
        """
        _ = map_output_dtypes
        return load_tfrecords(self._data_path, shuffle=self._shuffle_dataset and shuffle,
                              deterministic=(not shuffle),
                              auto_shard=auto_shard, map_func=map_func,
                              name_to_features=self.fields)

    def build_iterator(self, map_func=None, shard_id=0, total_shards=1):
        """ Returns the iterator of the dataset. """

        def gen():
            ds = load_tfrecords(self._data_path, shuffle=False, auto_shard=False,
                                name_to_features=self.fields,
                                sharding_index=shard_id, num_shards=total_shards)
            for x in ds:
                data = to_numpy_or_python_type(x, bytes_as_str=True)
                if map_func is not None:
                    data = map_func(data)
                yield data

        return gen
