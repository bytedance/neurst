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
import random

import tensorflow as tf
import yaml
from absl import logging

from neurst.data.datasets import Dataset, build_dataset, register_dataset
from neurst.data.datasets.data_sampler import DataSampler, build_data_sampler
from neurst.utils.flags_core import Flag, ModuleFlag


@register_dataset
class MixedTrainDataset(Dataset):
    """ Mixed datasets for training. """

    def __init__(self, args):
        """ Initializes the dataset. """
        super(MixedTrainDataset, self).__init__()
        self._data_files = args["data_files"]
        if isinstance(self._data_files, str):
            self._data_files = yaml.load(args["data_files"], Loader=yaml.FullLoader)
        assert isinstance(self._data_files, dict)
        self._data_sampler = build_data_sampler(args)
        common_properties = args["common_properties"]
        if common_properties is None:
            common_properties = {}
        elif isinstance(common_properties, str):
            common_properties = yaml.load(common_properties, Loader=yaml.FullLoader)
        assert isinstance(common_properties, dict)
        self._custom_dss = dict()
        self._status = None
        for name, ds in self._data_files.items():
            self._custom_dss[name] = build_dataset(
                args["data_class"], **ds, **common_properties)
            if self._status is None:
                self._status = self._custom_dss[name].status
            else:
                assert self._status == self._custom_dss[name].status, (
                    "Status of each dataset are supposed to be the same.")
        self._data_sampler = build_data_sampler(args)

    @staticmethod
    def class_or_method_args():
        return [
            Flag("data_files", dtype=Flag.TYPE.STRING,
                 help="A dict of data files. The key is the dataset name while "
                      "the value is a dict containing arguments indicating data files."),
            Flag("data_class", dtype=Flag.TYPE.STRING,
                 help="The dataset class for the data files."),
            Flag("common_properties", dtype=Flag.TYPE.STRING, default=None,
                 help="Other common properties for building a dataset."),
            ModuleFlag(DataSampler.REGISTRY_NAME, default=None,
                       help="The sampler for unbalanced datasets.")
        ]

    @property
    def status(self):
        return self._status

    def build(self, auto_shard=False, map_func=None, map_output_dtypes=None,
              shuffle=True):
        try:
            if self._data_sampler is None:
                return tf.data.experimental.sample_from_datasets(
                    [v.build(auto_shard, map_func, map_output_dtypes, shuffle)
                     for _, v in self._custom_dss.items()])
            else:
                weights = [self._data_sampler.normalized_sample_weights[k]
                           for k, _ in self._custom_dss.items()]
                return tf.data.experimental.sample_from_datasets(
                    [v.build(auto_shard, map_func, map_output_dtypes, shuffle).repeat()
                     for _, v in self._custom_dss.items()], weights=weights)
        except AttributeError:
            logging.info("Fail to use `tf.data.experimental.sample_from_datasets`. "
                         "We recommend you to upgrade TensorFlow to >= 2.4.")
            return super(MixedTrainDataset, self).build(auto_shard, map_func,
                                                        map_output_dtypes, shuffle)

    def build_iterator(self, map_func=None, shard_id=0, total_shards=1):
        """ Reads data from files and returns the iterator.

        Args:
            map_func: A function mapping a dataset element to another dataset element.
            shard_id: Generator yields on the `shard_id`-th shard of the whole dataset.
            total_shards: The number of total shards.
        """

        def gen():
            iterators = {k: v.build_iterator(map_func, shard_id, total_shards)()
                         for k, v in self._custom_dss.items()}
            key_set = list(iterators.keys())
            if self._data_sampler is None:
                while True:
                    choice = random.choice(key_set)
                    try:
                        yield next(iterators[choice])
                    except StopIteration:
                        key_set.remove(choice)
                        if len(key_set) == 0:
                            break
            else:
                while True:
                    choice = self._data_sampler()
                    try:
                        yield next(iterators[choice])
                    except StopIteration:
                        iterators[choice] = self._custom_dss[choice].build_iterator(
                            map_func, shard_id, total_shards)()
                        yield next(iterators[choice])

        return gen
