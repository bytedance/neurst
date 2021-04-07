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

from neurst.utils.configurable import ModelConfigs


@six.add_metaclass(ABCMeta)
class Converter(object):
    """ Abstract class for converting models and tasks. """
    REGISTRY_NAME = "converter"

    @classmethod
    def new(cls, *args, **kwargs):
        _ = args
        _ = kwargs
        return cls

    @staticmethod
    @abstractmethod
    def convert_model_config(path):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def convert_task_config(path):
        raise NotImplementedError

    @staticmethod
    def download(key):
        _ = key
        return None

    @staticmethod
    @abstractmethod
    def convert_checkpoint(path, save_path):
        raise NotImplementedError

    @classmethod
    def convert(cls, from_path, to_path):
        if (from_path.startswith("http://") or from_path.startswith("https://")
            or (not tf.io.gfile.exists(from_path))):
            path = cls.download(from_path)
            if path is None:
                raise ValueError(f"Fail to find model to download: {from_path}")
            from_path = path
        try:
            cfgs = cls.convert_model_config(from_path)
        except NotImplementedError:
            cfgs = {}
        try:
            cfgs.update(cls.convert_task_config(from_path))
        except NotImplementedError:
            pass
        ModelConfigs.dump(cfgs, to_path)
        cls.convert_checkpoint(from_path, to_path)
