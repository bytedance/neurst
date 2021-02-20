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
from absl import logging


class QuantLayer(tf.keras.layers.Layer):
    """ Base layer of keras layers which use quantization. """

    enable_quant = None
    quant_strategy = None
    quant_bits = None
    quant_weight_clip_max = None
    quant_act_clip_max = None

    @staticmethod
    def global_init(enable_quant,
                    quant_strategy="min/max",
                    quant_bits=8,
                    quant_weight_clip_max=1.0,
                    quant_act_clip_max=16.0,
                    **kwargs):
        """Initialize the global parameters for quantization. """
        QuantLayer.enable_quant = enable_quant
        if QuantLayer.enable_quant:
            QuantLayer.quant_strategy = quant_strategy
            QuantLayer.quant_bits = quant_bits
            if QuantLayer.quant_strategy == "min/max":
                QuantLayer.quant_weight_clip_max = quant_weight_clip_max
                QuantLayer.quant_act_clip_max = quant_act_clip_max
                logging.info("Enable min/max quantization:")
                logging.info(f"  quant_bits: {quant_bits}")
                logging.info(f"  quant_weight_clip_max: {quant_weight_clip_max}")
                logging.info(f"  quant_act_clip_max: {quant_act_clip_max}")
            else:
                raise ValueError("Only support min/max quantization currently.")
        if len(kwargs) > 0:
            logging.info(f"Unknown args of quantization: {kwargs}")

    @staticmethod
    def get_global_config():
        return {
            "quant_strategy": QuantLayer.quant_strategy,
            "quant_bits": QuantLayer.quant_bits,
            "quant_weight_clip_max": QuantLayer.quant_weight_clip_max,
            "quant_act_clip_max": QuantLayer.quant_act_clip_max
        }

    def __init__(self, *args, **kwargs):
        super(QuantLayer, self).__init__(*args, **kwargs)
        # Used to collect the quantizers used in this layer.
        self.traced = {}

    def add_weight_quantizer(self, weight):
        """ Add quantizer for weight if quantization is enabled. """
        name = weight.name.split(":")[0].split("/")[-1]
        if QuantLayer.enable_quant and "bias" not in name:
            # Add maximal value for cliping the weights.
            # The minimal value can be calculated using symmetric quantization.
            weight_clip_max = super(QuantLayer, self).add_weight(
                name=name + "_clip_max",
                trainable=True,
                regularizer=tf.keras.regularizers.l2(l=0.001),
                initializer=tf.constant_initializer(QuantLayer.quant_weight_clip_max),
                aggregation=tf.VariableAggregation.MEAN)
            self.traced[name] = WeightQuantization(weight_clip_max)
            return self.traced[name]
        return None

    def add_activation_quantizer(self, name, activation):
        """ Add quantizer for activation output if quantization is enabled

        Args:
            name: Name of the quantizer.
            activation: act, softmax or relu.
        """
        if not QuantLayer.enable_quant:
            return None
        activation_clip_max = None
        if activation != "softmax":
            activation_clip_max = super(QuantLayer, self).add_weight(
                name=name + "_clip_max",
                trainable=True,
                regularizer=tf.keras.regularizers.l2(l=0.01),
                initializer=tf.constant_initializer(QuantLayer.quant_act_clip_max),
                aggregation=tf.VariableAggregation.MEAN)
        self.traced[name] = ActivationQuantization(activation, activation_clip_max)
        return self.traced[name]

    def add_weight(self, *args, **kwargs):
        """ Add keras weight and weight quantizer if quantization is enable. """
        weight = super(QuantLayer, self).add_weight(*args, **kwargs)
        self.add_weight_quantizer(weight)
        return weight

    def quant_weight(self, weight):
        if not QuantLayer.enable_quant:
            return weight
        name = weight.name.split(":")[0].split("/")[-1]
        if name not in self.traced:
            raise ValueError(f"must call `add_weight_quantizer` for {name} in advance.")
        return self.traced[name](weight)

    def quant(self, inputs, name):
        """ quantize the inputs and return the quantized results.

        Args:
            inputs: A float tensor which needs to be quantized.
            name: Used to specify which quantizer will be used.
                The name must be the same as the one in above ``add_quantizer`` functions.

        Returns:
            If quantization is disable, directly return the original inputs.
            Otherwise return the quantized outputs accroding to the quantizer name.
        """
        if not QuantLayer.enable_quant:
            return inputs
        if name is None:
            raise ValueError("`name` of the quantizer must be provided.")
        return self.traced[name](inputs)


def _quant_tensor(inputs, min_w, max_w):
    """ quantize the inputs and return the quantized results.

    Args:
        inputs: A float tensor which needs to be quantized.
        min_w: Minimum of inputs.
        max_w: Maximum of inputs.

    Returns:
        Quantized outputs using algorithms described in https://arxiv.org/abs/1712.05877.
    """

    def apply():
        return tf.quantization.fake_quant_with_min_max_vars(
            inputs, min_w, max_w, num_bits=QuantLayer.quant_bits)

    return apply()


class WeightQuantization(tf.keras.layers.Layer):
    def __init__(self, clip_max):
        super(WeightQuantization, self).__init__()
        self.clip_max = clip_max

    def call(self, inputs):
        # The quantization can only be calculated in fp32.
        dtype = inputs.dtype
        inputs = tf.cast(inputs, tf.float32)

        # Calculate the minimum for cliping using symmetric quantization.
        weight_clip_max = tf.maximum(self.clip_max, 0.0)
        weight_clip_max = tf.cast(weight_clip_max, tf.float32)
        bits_tmp = float(2 ** (QuantLayer.quant_bits - 1))
        weight_clip_min = -weight_clip_max * bits_tmp / (bits_tmp - 1)

        # Quantization.
        outputs = _quant_tensor(inputs, weight_clip_min, weight_clip_max)
        outputs = tf.cast(outputs, dtype)
        return outputs


class ActivationQuantization(tf.keras.layers.Layer):
    def __init__(self, activation, clip_max=None):
        super(ActivationQuantization, self).__init__()
        self.activation = activation
        if self.activation not in ["relu", "softmax"]:
            self.activation = "act"
        # if self.activation not in ["act", "softmax", "relu"]:
        #     raise ValueError("`activation` should be one of (act, softmax, relu).")
        self.activation_clip_max = clip_max

    def call(self, inputs):
        dtype = inputs.dtype
        inputs = tf.cast(inputs, tf.float32)

        if self.activation == "act" or self.activation == "relu":
            activation_clip_max = tf.maximum(self.activation_clip_max, 0.0)
            activation_clip_max = tf.cast(activation_clip_max, tf.float32)
            if self.activation == "relu":
                activation_clip_min = 0.0
            else:
                bits_tmp = float(2 ** (QuantLayer.quant_bits - 1))
                activation_clip_min = -activation_clip_max * bits_tmp / (bits_tmp - 1)
        elif self.activation == "softmax":
            quant_bits = QuantLayer.quant_bits
            activation_clip_max = float(2 ** quant_bits - 1) / (2 ** quant_bits)
            activation_clip_min = 0.0
        else:
            raise ValueError("`type` should be one of (act, softmax, relu).")

        outputs = _quant_tensor(inputs, activation_clip_min, activation_clip_max)
        outputs = tf.cast(outputs, dtype)
        return outputs

# class WeightQuantization(tf.keras.layers.Layer):
#     def __init__(self, name):
#         super(WeightQuantization, self).__init__(name=name)
#
#     def build(self, input_shape):
#         """ Add maximal value for cliping the weights.
#             The minimal value can be calculated using symmetric quantization. """
#         self.weight_clip_max = self.add_weight(
#             name="clip_max",
#             trainable=True,
#             regularizer=tf.keras.regularizers.l2(l=0.001),
#             initializer=tf.constant_initializer(QuantLayer.quant_weight_clip_max),
#             aggregation=tf.VariableAggregation.MEAN)
#         super(WeightQuantization, self).build(input_shape)
#
#     def call(self, inputs):
#         # The quantization can only be calculated in fp32.
#         dtype = inputs.dtype
#         inputs = tf.cast(inputs, tf.float32)
#
#         # Calculate the minimum for cliping using symmetric quantization.
#         weight_clip_max = tf.maximum(self.weight_clip_max, 0.0)
#         weight_clip_max = tf.cast(weight_clip_max, tf.float32)
#         bits_tmp = float(2 ** (QuantLayer.quant_bits - 1))
#         weight_clip_min = -weight_clip_max * bits_tmp / (bits_tmp - 1)
#
#         # Quantization.
#         outputs = _quant_tensor(inputs, weight_clip_min, weight_clip_max)
#         outputs = tf.cast(outputs, dtype)
#         return outputs
#
#
# class ActivationQuantization(tf.keras.layers.Layer):
#     def __init__(self, name, activation):
#         super(ActivationQuantization, self).__init__(name=name)
#         self.activation = activation
#         if self.activation not in ["relu", "softmax"]:
#             self.activation = "act"
#         # if self.activation not in ["act", "softmax", "relu"]:
#         #     raise ValueError("`activation` should be one of (act, softmax, relu).")
#
#     def build(self, input_shape):
#         """ Add maximal value for cliping the activations.
#             The minimal value can be calculated using symmetric quantization. """
#         if self.activation == "act" or self.activation == "relu":
#             self.activation_clip_max = self.add_weight(
#                 name="clip_max",
#                 trainable=True,
#                 regularizer=tf.keras.regularizers.l2(l=0.01),
#                 initializer=tf.constant_initializer(QuantLayer.quant_act_clip_max),
#                 aggregation=tf.VariableAggregation.MEAN)
#         super(ActivationQuantization, self).build(input_shape)
#
#     def call(self, inputs):
#         dtype = inputs.dtype
#         inputs = tf.cast(inputs, tf.float32)
#
#         if self.activation == "act" or self.activation == "relu":
#             activation_clip_max = tf.maximum(self.activation_clip_max, 0.0)
#             activation_clip_max = tf.cast(activation_clip_max, tf.float32)
#             if self.activation == "relu":
#                 activation_clip_min = 0.0
#             else:
#                 bits_tmp = float(2 ** (QuantLayer.quant_bits - 1))
#                 activation_clip_min = -activation_clip_max * bits_tmp / (bits_tmp - 1)
#         elif self.activation == "softmax":
#             quant_bits = QuantLayer.quant_bits
#             activation_clip_max = float(2 ** quant_bits - 1) / (2 ** quant_bits)
#             activation_clip_min = 0.0
#         else:
#             raise ValueError("`type` should be one of (act, softmax, relu).")
#
#         outputs = _quant_tensor(inputs, activation_clip_min, activation_clip_max)
#         outputs = tf.cast(outputs, dtype)
#         return outputs
