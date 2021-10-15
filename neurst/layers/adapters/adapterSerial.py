""" Base Adapter class. """
from __future__ import absolute_import, division, print_function

import tensorflow as tf

# tf.random.Generator = None # Patch for a bug, https://stackoverflow.com/questions/62696815/tensorflow-core-api-v2-random-has-no-attribute-generator

from neurst.layers.adapters import register_adapter
from neurst.layers.adapters.adapter import Adapter


@register_adapter
class AdapterSerial(Adapter):
    """Serial Adapter
     reimplement of Simple, scalable adaptation for neural machine translation (EMNLP-IJCNLP 2019 )
    """

    def __init__(self,
                 hidden_size_inner,
                 hidden_size_outter,
                 dropout_rate=.3,
                 name="SerialAdapter", ):
        """ Initializes the parameters of the Serial Adapter.
        """
        super(AdapterSerial, self).__init__(
            hidden_size_inner=hidden_size_inner,
            hidden_size_outter=hidden_size_outter,
            dropout_rate=dropout_rate,
            name=name,
        )
        self.inner_layer = None
        self.outter_layer = None
        self._norm_layer = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, dtype="float32", name="output_ln")

    def build(self, input_shape):
        params = self.get_config()
        self.inner_layer = tf.keras.layers.Dense(units=params["hidden_size_inner"], activation=tf.nn.relu)
        self.outter_layer = tf.keras.layers.Dense(units=params["hidden_size_outter"], activation=None)
        super(Adapter, self).build(input_shape)

    def call(self, inputs, is_training=True):
        params = self.get_config()
        z = self._norm_layer(inputs)
        z = self.inner_layer(z)
        z = tf.cast(z, inputs.dtype)
        if is_training:
            z = tf.nn.dropout(z, params["dropout_rate"])
        h_out = tf.cast(self.outter_layer(z), inputs.dtype) + inputs
        return h_out
