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
from typing import Tuple

import six
import tensorflow as tf

from neurst.data.datasets import Dataset
from neurst.layers.quantization.quant_layers import QuantLayer
from neurst.utils.configurable import copy_dict_list
from neurst.utils.flags_core import COMMON_DATA_ARGS


@six.add_metaclass(ABCMeta)
class Task(object):
    """ The task object binds the data pipelines. """

    REGISTRY_NAME = "task"

    def __init__(self, args):
        """ Initializes with configuration. """
        self._args = args

    def model_configs(self, model):
        return {
            "model.class": model.__class__.__name__,
            "model.params": copy_dict_list(model.args),
            "task.class": self.__class__.__name__,
            "task.params": self.get_config(),
            "enable_quant": QuantLayer.enable_quant,
            "quant_params": QuantLayer.get_global_config()
        }

    def create_inputs(self, mode):
        """ Creates keras input placeholders. """
        dtypes, signatures = self.inputs_signature(mode)
        if isinstance(dtypes, dict):
            inps = dict()
            for name in dtypes:
                inps[name] = tf.keras.layers.Input(tuple(signatures[name][1:]),
                                                   dtype=dtypes[name], name=name)
            return inps
        else:
            inps = []
            for d, s in zip(tf.nest.flatten(dtypes), tf.nest.flatten(signatures)):
                inps.append(tf.keras.layers.Input(tuple(s[1:]), dtype=d))
            return tf.nest.pack_sequence_as(dtypes, inps)

    @abstractmethod
    def inputs_signature(self, mode) -> Tuple[dict, dict]:
        """ Defines the types and signatures of the dataset inputs. """
        raise NotImplementedError("Task must implement the `inputs_signature` method.")

    @abstractmethod
    def example_to_input(self, batch_of_data: dict, mode) -> dict:
        """ Converts a batch of data into the model readable structure. """
        raise NotImplementedError("Task must implement the `example_to_input` method.")

    @abstractmethod
    def get_data_preprocess_fn(self, mode, data_status, args=None) -> callable:
        """ Returns a callable function that preprocess the data sample
            according to this task. """
        raise NotImplementedError("Task must implement the `get_data_preprocess_fn` method.")

    def get_data_postprocess_fn(self, data_status, **kwargs) -> callable:
        """ Returns a callable function that postprocess the data sample
            according to this task. """
        raise NotImplementedError

    @abstractmethod
    def create_and_batch_tfds(self,
                              ds: Dataset,
                              mode,
                              args=None,
                              num_replicas_in_sync=1) -> tf.data.Dataset:
        """ Batch dataset. """
        raise NotImplementedError("Task must implement the `create_and_batch_tfds` method.")

    @abstractmethod
    def get_config(self):
        return {}

    @staticmethod
    def class_or_method_args():
        """ Returns a list of args for flag definition. """
        return [x for x in COMMON_DATA_ARGS]

    @abstractmethod
    def build_model(self, args, name=None, **kwargs):
        """Build a new model instance."""
        raise NotImplementedError("Task must implement the `build_model` method.")

    def build_metric_layer(self):
        """ Builds a list of metric layers for logging and tensorboard. """
        return []

    def get_eval_metric(self, args, name="metric", ds=None):
        """ Returns a neurst.metrics.metric.Metric object for evaluation."""
        return None
