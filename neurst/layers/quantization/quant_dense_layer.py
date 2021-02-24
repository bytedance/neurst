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

from neurst.layers.quantization.quant_layers import QuantLayer


class QuantDense(tf.keras.layers.Dense, QuantLayer):
    """ `tf.keras.layers.Dense` with quantization. """

    def __init__(self, activation_quantizer=None, *args, **kwargs):
        tf.keras.layers.Dense.__init__(self, *args, **kwargs)
        QuantLayer.__init__(self, name=self.name)
        self._quant_op = None
        if activation_quantizer is not None:
            self._quant_op = self.add_activation_quantizer(self.name + "_activ", activation_quantizer)

    def build(self, input_shape):
        tf.keras.layers.Dense.build(self, input_shape)
        self.add_weight_quantizer(self.kernel)
        self.v = self.kernel
        self.built = True

    def call(self, inputs):
        self.kernel = tf.cast(self.quant_weight(self.v), inputs.dtype)
        return tf.keras.layers.Dense.call(self, inputs)

    def __call__(self, *args, **kwargs):
        output = tf.keras.layers.Dense.__call__(self, *args, **kwargs)
        if self._quant_op is None:
            return output
        return self._quant_op(output)
