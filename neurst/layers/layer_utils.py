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

from neurst.utils import compat


def input_padding_to_bias(input_padding):
    """ Creates a bias tensor according to the padding tensor for attention.

    Args:
        input_padding: A float tensor with shape [batch_size, max_length],
                indicating the padding positions, where 1.0 for padding and
                0.0 for non-padding.

    Returns:
        Attention bias tensor with shape [batch_size, max_length]
    """
    with tf.name_scope("attention_bias"):
        bias = input_padding * compat.FLOAT_MIN
    return bias


def lower_triangle_attention_bias(length, dtype=None):
    """ Create a bias tensor for decoder self attention.

      Allows a query to attend to all positions up to and including its own.
    Args:
        length: A scalar.

    Returns: A float Tensor of shape [1, 1, length, length], with FLOAT_MIN in
      padding positions and 0 in non-padding positions.

    """
    if dtype is None:
        dtype = compat.CUSTOM_GLOBAL_FLOATX
    with tf.name_scope("decoder_self_attention_bias"):
        lower_triangle = tf.cast(
            tf.linalg.band_part(tf.ones([length, length]), -1, 0), dtype=dtype)
        bias = tf.reshape(compat.FLOAT_MIN * (1. - lower_triangle),
                          [1, 1, length, length])
    return bias


def waitk_attention_bias(memory_length, waitk_lagging, query_length=None, dtype=None):
    """ Creates a bias tensor for decoder self attention with lagging.

    Args:
        memory_length: The length of memory tensor.
        waitk_lagging: The lagging.
        query_length: The length of queries or the position of the query.

    Returns: A float Tensor of shape [query_length, memory_length] if `query_length` is None,
        else a float Tensor of shape [memory_length, ],  with FLOAT_MIN in padding positions
        and 0 in non-padding positions.

    """
    with tf.name_scope("decoder_waitk_self_attention_bias"):
        if query_length is None:
            return compat.FLOAT_MIN * (1. - tf.sequence_mask(
                tf.minimum(waitk_lagging, memory_length), maxlen=memory_length,
                dtype=(dtype or compat.CUSTOM_GLOBAL_FLOATX)))
        waitk_non_padding = tf.cast(
            tf.linalg.band_part(tf.ones([query_length, memory_length]), -1,
                                tf.minimum(waitk_lagging - 1, memory_length)),
            dtype=(dtype or compat.CUSTOM_GLOBAL_FLOATX))
        return compat.FLOAT_MIN * (1. - waitk_non_padding)


def stack_beam_size(x, beam_size):
    """ Tiles a given tensor by beam_size.

    Args:
        x: A tensor with shape [batch_size, ...].
        beam_size: An int scalar.

    Returns:
        The tiled tensor with shape [batch_size * beam_size, ...]

    Raises:
        AssertionError: if the shape of tensor does not match
          [batch_size, 1, 1, timesteps] when tensor.ndims == 4.
        NotImplementedError: if tensor.ndims > 4.
    """
    assert compat.is_tf_tensor(x)
    original_shape = tf.shape(x)
    x = tf.expand_dims(x, axis=1)
    tile_dims = [1] * x.shape.ndims
    tile_dims[1] = beam_size
    tiled_x = tf.tile(x, tile_dims)
    tiled_shape = tf.concat([[-1], original_shape[1:]], axis=0)
    return tf.reshape(tiled_x, tiled_shape)


def static_shape_list(tensor):
    """Return a list of the tensor's shape, and ensure no None values in list."""
    # Get statically known shape (may contain None's for unknown dimensions)
    shape = tensor.get_shape().as_list()

    # Ensure that the shape values are not None
    dynamic_shape = tf.shape(tensor)
    for i in range(len(shape)):  # pylint: disable=consider-using-enumerate
        if shape[i] is None:
            shape[i] = dynamic_shape[i]
    return shape


def static_tensorshape(tensor):
    """ Returns the static TensorShape. """
    return tf.TensorShape(tensor.get_shape().as_list())


def dynamic_tensorshape_except_last_dim(tensor):
    """ Returns a tf.TensorShape with only last dim having the static shape. """
    shape_list = static_shape_list(tensor)
    # Only the last
    for i in range(len(shape_list) - 1):
        shape_list[i] = None

    if compat.is_tf_tensor(shape_list[-1]):
        shape_list[-1] = None
    return tf.TensorShape(shape_list)


def one_entry_bias(on_entry, num_entries,
                   on_value, off_value, dtype=None):
    """ Builds a bias vector to be added to log_probs for special use.

    Args:
        on_entry: A python integer.
        num_entries: A python integer.
        on_value: A scalar defining the value to fill in
            output when `index = on_entry`.
        off_value: A scalar defining the value to fill in
            output when `index != on_entry`.
        dtype: The tensor type.

    Returns: A bias vector with shape [num_entries, ].
    """
    if dtype is None:
        dtype = compat.CUSTOM_GLOBAL_FLOATX
    bias = tf.one_hot(
        [on_entry], num_entries,
        on_value=tf.convert_to_tensor(on_value, dtype=dtype),
        off_value=tf.convert_to_tensor(off_value, dtype=dtype),
        dtype=tf.dtypes.as_dtype(dtype))
    return tf.squeeze(bias, axis=0)


def tile_tensor(tensor, size, axis=0):
    """ Stacks a given tensor `size` times on a specific axis.

    For example, tensor=[1, 2, 3, 4], beam_size=3, axis=0 get the tensor
    [ [1, 2, 3, 4],
      [1, 2, 3, 4],
      [1, 2, 3, 4] ]

    tensor=[[1, 2, 3], [3, 4, 5]], beam_size=1, axis=1 get the tensor
    [ [[1, 2, 3]], [[3, 4, 5]] ]

    Args:
        tensor: A Tensor.
        size: A python integer, the size to be stacked.
        axis: A python integer.

    Returns: A Tensor.
    """
    tensor = tf.expand_dims(tensor, axis=axis)
    tile_dims = [1] * tensor.get_shape().ndims
    tile_dims[axis] = size
    return tf.tile(tensor, tile_dims)


def compute_batch_indices(batch_size, k):
    """ Computes the i'th coordinate that contains the batch index for gathers.

    Batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..]. It says which
    batch the beam item is in. This will create the i of the i,j coordinate
    needed for the gather.

    Args:
        batch_size: A python integer, the batch size.
        k: A python integer, the beam width.

    Returns: A Tensor.
    """
    # [beam_size, batch_size]: [[0, 1, 2,..., batch_size], [0, 1, 2,..., batch_size], ...]
    batch_pos = tile_tensor(tf.range(batch_size), k)
    batch_pos = tf.transpose(batch_pos)
    return batch_pos
