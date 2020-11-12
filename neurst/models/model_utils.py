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

from neurst.utils import compat


def deduce_data_length(data_tensor, eos_id):
    """ Gets the length by data tensor.

    Args:
        data_tensor: A Tensor with shape [batch_size, max_len]
        eos_id: The id of the end-of-sentence symbol. Note that the
            EOS here is the last word of the vocabulary (with
            the maximum ID).

    Returns:
        The length Tensor with shape [batch_size, ]
    """
    return tf.reduce_sum(tf.cast(
        tf.less(data_tensor, eos_id), tf.int32), axis=1) + 1


def deduce_inputs_padding(data_tensor, eos_id):
    """ Creates a bias tensor according to the padding tensor for attention.

    Args:
        data_tensor: A Tensor with shape [batch_size, max_len]
        eos_id: The id of the end-of-sentence symbol. Note that the
            EOS here is the last word of the vocabulary (with
            the maximum ID).

    Returns:
        A float tensor with shape [batch_size, max_len],
        indicating the padding positions, where 1.0 for padding and
        0.0 for non-padding.
    """
    with tf.name_scope("padding"):
        length = deduce_data_length(data_tensor, eos_id)
        padding = 1. - tf.sequence_mask(
            lengths=tf.cast(length, tf.int32),
            maxlen=tf.cast(tf.shape(data_tensor)[1], tf.int32),
            dtype=tf.dtypes.as_dtype(compat.CUSTOM_GLOBAL_FLOATX))  # 1.0 for padding
    return padding


def _summary_model_variables(varname_shape_list, output_stream=logging.info):
    output_stream("variable name  | # parameters")
    var_sizes = dict()
    var_hiro_names = dict()
    for var_name, var_shape_list in varname_shape_list:
        var_names = var_name.split(":")[0].split("/")
        var_total_size = 1
        for _s in var_shape_list:
            var_total_size *= _s
        if var_names[0] not in var_hiro_names:
            var_hiro_names[var_names[0]] = dict()
        _tmp_var_names = var_hiro_names[var_names[0]]
        for i in range(1, len(var_names)):
            _tmp_name = "/".join(var_names[:i])
            if _tmp_name not in var_sizes:
                var_sizes[_tmp_name] = 0
            var_sizes[_tmp_name] += var_total_size
            if i == len(var_names) - 1:
                _tmp_var_names[var_names[i]] = var_shape_list
            else:
                if var_names[i] not in _tmp_var_names:
                    _tmp_var_names[var_names[i]] = dict()
                _tmp_var_names = _tmp_var_names[var_names[i]]
        var_sizes["/".join(var_names)] = var_total_size

    def _get_size(v):
        killo = v / 1000.
        milli = v / 1000000.
        if milli >= 1:
            return "%.2fm" % milli
        elif killo >= 1:
            return "%.2fk" % killo
        return v

    def _recursive_print(accumulate_name_list, recuv_varname_dict, indent=0):
        for k, v in recuv_varname_dict.items():
            accumulate_name_list.append(k)
            prefix = " "
            if indent > 0:
                prefix = "".join([" "] * (indent + 1))
            this_name = "/".join(accumulate_name_list)
            if isinstance(v, dict):
                output_stream(prefix + this_name + " (--/{} params)".format(_get_size(var_sizes[this_name])))
                _recursive_print(accumulate_name_list, v, indent + 2)
            else:
                output_stream(prefix + this_name + " ({}, {} params)".format(
                    "x".join([str(_v) for _v in v]), _get_size(var_sizes[this_name])))
            accumulate_name_list.pop(-1)

    _recursive_print([], var_hiro_names)


def summary_model_variables(keras_model):
    """ Prints model variables and shapes.

    Args:
        keras_model: A keras model.
    """
    varname_shape_list = [
        (var.name, var.shape.as_list()) for var in keras_model.weights]
    _summary_model_variables(varname_shape_list)
