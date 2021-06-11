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
import re

import tensorflow as tf
from absl import logging

from neurst.data.text.vocab import PaddingMode
from neurst.utils import compat


def deduce_text_length(data_tensor, pad_id, padding_mode):
    """ Gets the length by text data tensor.

    Args:
        data_tensor: A Tensor with shape [batch_size, max_len]
        pad_id: The id of the PAD symbol.
        padding_mode: A PaddingMode value.

    Returns:
        The length Tensor with shape [batch_size, ]
    """
    if padding_mode == PaddingMode.DEFAULT:
        return tf.reduce_sum(tf.cast(
            tf.not_equal(data_tensor, pad_id), tf.int32), axis=1)
    elif padding_mode == PaddingMode.EOS_AS_PADDING:
        return tf.argmin(tf.cast(
            tf.not_equal(data_tensor, pad_id), tf.int32), axis=-1) + 1
    else:
        raise NotImplementedError


def input_length_to_nonpadding(lengths, max_len, dtype=None):
    """ Creates a bias tensor according to the non-padding tensor for cross entropy.

    Args:
        length: A Tensor with shape [batch_size, ], indicating the true length.
        max_len: A scalar tensor indicating the maximum length.

    Returns:
        A float tensor with shape [batch_size, max_len],
        indicating the padding positions, where 0.0 for padding and
        1.0 for non-padding.
    """
    return tf.sequence_mask(
        lengths=tf.cast(lengths, tf.int32),
        maxlen=tf.cast(max_len, tf.int32),
        dtype=(dtype or tf.dtypes.as_dtype(compat.CUSTOM_GLOBAL_FLOATX)))  # 1.0 for non-padding


def input_length_to_padding(lengths, max_len, dtype=None):
    """ Creates a bias tensor according to the padding tensor for attention.

    Args:
        length: A Tensor with shape [batch_size, ], indicating the true length.
        max_len: A scalar tensor indicating the maximum length.

    Returns:
        A float tensor with shape [batch_size, max_len],
        indicating the padding positions, where 1.0 for padding and
        0.0 for non-padding.
    """
    with tf.name_scope("padding"):
        return 1. - input_length_to_nonpadding(lengths, max_len, dtype)


def _summary_model_variables(varname_shape_list, freeze_variables=None,
                             output_stream=logging.info):
    output_stream("variable name  | # parameters")
    var_sizes = dict()
    var_hiro_names = dict()
    var_trainable = dict()
    non_trainable_sizes = 0
    for var_name, var_shape_list, trainable in varname_shape_list:
        if (trainable and freeze_variables is not None
            and re.search(freeze_variables, var_name) is not None):
            trainable = False
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
        var_trainable["/".join(var_names)] = trainable
        if not trainable:
            non_trainable_sizes += var_total_size

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
                extra = (f", {_get_size(var_sizes[this_name]-non_trainable_sizes)} trainable params"
                         if non_trainable_sizes > 0 and indent == 0 else "")
                output_stream(prefix + this_name + " (--/{} params{})".format(
                    _get_size(var_sizes[this_name]), extra))
                _recursive_print(accumulate_name_list, v, indent + 2)
            else:
                extra = "" if var_trainable[this_name] else ", non-trainable"
                output_stream(prefix + this_name + " ({}, {} params{})".format(
                    "x".join([str(_v) for _v in v]), _get_size(var_sizes[this_name]), extra))
            accumulate_name_list.pop(-1)

    _recursive_print([], var_hiro_names)


def summary_model_variables(keras_model, freeze_variables=None):
    """ Prints model variables and shapes.

    Args:
        keras_model: A keras model.
    """
    varname_shape_list = [
        (var.name, var.shape.as_list(), var.trainable) for var in keras_model.weights]
    _summary_model_variables(varname_shape_list, freeze_variables)
