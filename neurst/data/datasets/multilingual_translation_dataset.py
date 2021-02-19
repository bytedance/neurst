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
import random

import tensorflow as tf
from absl import logging

from neurst.data.dataset_utils import load_tfrecords
from neurst.data.datasets import register_dataset
from neurst.data.datasets.data_sampler import DataSampler, build_data_sampler
from neurst.data.datasets.data_sampler.temperature_sampler import TemperatureSampler
from neurst.data.datasets.parallel_text_dataset import AbstractParallelDataset
from neurst.utils.compat import DataStatus
from neurst.utils.flags_core import Flag, ModuleFlag


@register_dataset(["mutilingual_translation_tfrecord", "m2m_tfrecord"])
class MultilingualTranslationTFRecordDataset(AbstractParallelDataset):
    """ The multilingual translation dataset for training. """

    def __init__(self, args):
        """ Initializes the dataset. """
        super(MultilingualTranslationTFRecordDataset, self).__init__()
        self._path = args["path"]
        self._data_sampler = build_data_sampler(args)
        self._auto_switch_langs = args["auto_switch_langs"]

    @staticmethod
    def class_or_method_args():
        return [
            Flag("path", dtype=Flag.TYPE.STRING,
                 help="The path to multilingual datasets. "
                      "The record files should be stored in sub directories, which are named by src2trg, "
                      "e.g. rootpath/en2de, rootpath/en2fr..."),
            Flag("auto_switch_langs", dtype=Flag.TYPE.BOOLEAN,
                 help="Whether to switch source and target langauges (which will doubled the dataset)."),
            ModuleFlag(DataSampler.REGISTRY_NAME, default=TemperatureSampler.__name__,
                       help="The sampler for unbalanced datasets.")
        ]

    @property
    def status(self):
        return DataStatus.PROJECTED

    def build_iterator(self, map_func=None, shard_id=0, total_shards=1):
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
        if not tf.io.gfile.exists(self._path):
            raise ValueError(f"Fail to find data path: {self._path}.")
        lang2numpy_iter = dict()
        for path in tf.io.gfile.glob(os.path.join(self._path, "*")):
            lang_pair = path.strip().split("/")[-1]
            langs = lang_pair.strip().split("2")
            if len(langs) == 1:
                langs = lang_pair.strip().split("-")
                reversed_lang_pair = langs[1] + "-" + langs[0]
            else:
                reversed_lang_pair = langs[1] + "2" + langs[0]
            if tf.io.gfile.isdir(path) and len(langs) == 2:
                lang2numpy_iter[lang_pair] = load_tfrecords(
                    file_path=os.path.join(path, "*"),
                    shuffle=True, deterministic=False,
                    auto_shard=False, map_func=map_func,
                    name_to_features={"feature": tf.io.VarLenFeature(tf.int64),
                                      "label": tf.io.VarLenFeature(tf.int64)},
                    auxiliary_elements={"src_lang": langs[0], "trg_lang": langs[1]}).repeat().as_numpy_iterator()
                if self._auto_switch_langs:
                    lang2numpy_iter[reversed_lang_pair] = load_tfrecords(
                        file_path=os.path.join(path, "*"),
                        shuffle=True, deterministic=False,
                        auto_shard=False, map_func=map_func,
                        name_to_features={"feature": tf.io.VarLenFeature(tf.int64),
                                          "label": tf.io.VarLenFeature(tf.int64)},
                        feature_name_mapping={"label": "feature", "feature": "label"},
                        auxiliary_elements={"trg_lang": langs[0], "src_lang": langs[1]}).repeat().as_numpy_iterator()
            else:
                logging.info(f"Ignore {path}.")

        def gen():
            keys = list(lang2numpy_iter.keys())
            while True:
                if self._data_sampler is None:
                    choice = random.choice(keys)
                else:
                    choice = self._data_sampler()
                yield lang2numpy_iter[choice].next()

        # TODO: to see https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
        # and https://www.tensorflow.org/api_docs/python/tf/data/experimental/sample_from_datasets
        dataset = tf.data.Dataset.from_generator(gen, output_types=map_output_dtypes)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset = dataset.with_options(options)
        return dataset
