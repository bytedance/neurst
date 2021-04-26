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
import numpy
import tensorflow as tf


def gelu(x, non_approximate=False):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
        x: float Tensor to perform activation.
        non_approximate: use tanh approximation
    Returns:
        `x` with the GELU activation applied.
    """
    if non_approximate:
        # TODO: check fp16
        # https://github.com/tensorflow/tensorflow/issues/25052
        if x.dtype.base_dtype.name == "float16":
            fp32_x = tf.cast(x, tf.float32)
        else:
            fp32_x = x
        cdf = 0.5 * (1.0 + tf.math.erf(fp32_x / numpy.sqrt(2.0)))

        if x.dtype.base_dtype.name == "float16":
            return x * tf.saturate_cast(cdf, tf.float16)

        return x * cdf
    cdf = 0.5 * (1.0 + tf.tanh(
        (numpy.sqrt(2 / numpy.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def glu(x):
    """ Gated linear unit. """
    a, b = tf.split(x, axis=-1, num_or_size_splits=2)
    return a * tf.nn.sigmoid(b)


def get_activation(activ):
    if callable(activ):
        return activ
    if activ is None:
        return None
    if activ == "tanh":
        return tf.nn.tanh
    elif activ == "relu":
        return tf.nn.relu
    elif activ == "gelu" or activ == "gelu_approx":
        return lambda x: gelu(x, non_approximate=False)
    elif activ == "gelu_nonapprox":
        return lambda x: gelu(x, non_approximate=True)
    else:
        raise ValueError("Unknown activation: {}".format(activ))
