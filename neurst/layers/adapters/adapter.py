""" Base Adapter class. """
from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod

import six
import tensorflow as tf
from neurst.utils.configurable import extract_constructor_params


@six.add_metaclass(ABCMeta)
class Adapter(tf.keras.layers.Layer):
    """Base class for Adapter """

    def __init__(self, name=None, **kwargs):
        """ Initializes the parameters of the decoder. """
        self._params = extract_constructor_params(locals(), verbose=False)
        super(Adapter, self).__init__(name=name)

    def build(self, input_shape):
        super(Adapter, self).build(input_shape)

    def get_config(self):
        return self._params

    @abstractmethod
    def call(self, inputs, is_training=True):
        raise NotImplementedError
