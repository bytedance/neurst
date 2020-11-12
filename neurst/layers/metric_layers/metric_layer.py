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
from abc import abstractmethod

import tensorflow as tf

from neurst.layers.metric_layers import METRIC_REDUCTION, register_metric


class MetricLayer(tf.keras.layers.Layer):
    """ The base class of metric layer for verbose and """

    def __init__(self):
        super(MetricLayer, self).__init__()
        self._layer_metrics = {}

    def build_metric_reduction(self, name, reduction):
        register_metric(name, reduction)
        if reduction == METRIC_REDUCTION.SUM:
            self._layer_metrics[name] = tf.keras.metrics.Sum(name)
        elif reduction == METRIC_REDUCTION.MEAN:
            self._layer_metrics[name] = tf.keras.metrics.Mean(name)
        else:
            raise NotImplementedError(f"Unknown reduction name: {reduction}.")

    @abstractmethod
    def calculate(self, input, output):
        """ Calculates metric values according to model input and output. """
        raise NotImplementedError

    def call(self, inputs):
        """ Registers metrics by calling `self.add_metric()`

        Args:
            inputs: A list of [model inputs, model outputs]

        Returns:
            The model outputs.
        """
        model_inp, model_out = inputs
        ms = self.calculate(model_inp, model_out)
        if not isinstance(ms, dict):
            assert len(self._layer_metrics) == 1, "The number of metrics mismatch."
            for k, v in self._layer_metrics.items():
                ms = {k: ms}
        for name, aggr in self._layer_metrics.items():
            m = aggr(ms[name])
            self.add_metric(m)
        return model_out
