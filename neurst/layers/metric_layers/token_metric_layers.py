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

from neurst.layers.metric_layers import METRIC_REDUCTION
from neurst.layers.metric_layers.metric_layer import MetricLayer


class SequenceTokenMetricLayer(MetricLayer):
    """Custom a layer of metrics for Transformer model."""

    def __init__(self, name_prefix, key=None):
        super(SequenceTokenMetricLayer, self).__init__()
        self._name_prefix = "" if name_prefix is None else (name_prefix + "_")
        self._key = key or name_prefix

    def build(self, input_shape):
        super(SequenceTokenMetricLayer, self).build(input_shape)
        self.build_metric_reduction(self._name_prefix + "tokens", METRIC_REDUCTION.SUM)
        self.build_metric_reduction(self._name_prefix + "real_tokens", METRIC_REDUCTION.SUM)

    def calculate(self, input, output):
        """ Calculates metric values according to model input and output. """
        x = input[self._key]
        ms = {self._name_prefix + "tokens": tf.cast(tf.shape(x)[0] * tf.shape(x)[1], tf.float32)}
        if self._name_prefix + "padding" in input:
            x_len = ms[self._name_prefix + "tokens"] - tf.cast(
                tf.reduce_sum(input[self._name_prefix + "padding"]), tf.float32)
        else:
            x_len = tf.reduce_sum(tf.cast(input[self._name_prefix + "length"], tf.float32))
        ms[self._name_prefix + "real_tokens"] = x_len
        return ms


class AudioFramesMetricLayer(MetricLayer):
    """Custom a layer of metrics for Speech Transformer model."""

    def __init__(self, name_prefix):
        super(AudioFramesMetricLayer, self).__init__()
        self._name_prefix = name_prefix

    def build(self, input_shape):
        """" Builds metric layer. """
        super(AudioFramesMetricLayer, self).build(input_shape)
        self.build_metric_reduction(self._name_prefix + "_tokens", METRIC_REDUCTION.SUM)
        self.build_metric_reduction(self._name_prefix + "_real_tokens", METRIC_REDUCTION.SUM)

    def calculate(self, input, output):
        """ Calculates metric values according to model input and output. """
        x = input[self._name_prefix]
        x_len = input[self._name_prefix + "_length"]
        ms = {self._name_prefix + "_tokens": tf.cast(tf.shape(x)[0] * tf.shape(x)[1], tf.float32),
              self._name_prefix + "_real_tokens": tf.cast(tf.reduce_sum(x_len), tf.float32)}
        return ms


class BatchCountMetricLayer(MetricLayer):
    """Custom a layer of metrics for Transformer model."""

    def __init__(self, key):
        super(BatchCountMetricLayer, self).__init__()
        self._key = key

    def build(self, input_shape):
        super(BatchCountMetricLayer, self).build(input_shape)
        self.build_metric_reduction("samples", METRIC_REDUCTION.SUM)

    def calculate(self, input, output):
        """ Calculates metric values according to model input and output. """
        x = input[self._key]
        return {"samples": tf.cast(tf.shape(x)[0], tf.float32)}
