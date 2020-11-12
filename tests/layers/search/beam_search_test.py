import numpy
import tensorflow as tf

from neurst.layers.layer_utils import one_entry_bias, stack_beam_size, tile_tensor
from neurst.utils import compat


def tf1codebase_stack_beam_size(tensors, beam_size):
    """ Stacks the tensors `beam_size` times at specific dimension.

    Args:
        tensors: A Tensor of a list/tuple/dict of Tensors. For each Tensor, the first
          dimension must be batch_size, otherwise, unknow errors may occur.
        beam_size: A python integer, the beam width.

    Returns: A Tensor or a list/tuple of Tensors with the same structure
      as `tensors`.

    Raises:
        AssertionError: if the shape of tensor does not match
          [batch_size, 1, 1, timesteps] when tensor.ndims == 4.
        NotImplementedError: if tensor.ndims > 4.
    """

    def _stack(x):
        assert isinstance(x, tf.Tensor)
        batch_size = tf.shape(x)[0]
        x_ndims = x.get_shape().ndims
        last_dim = x.get_shape().as_list()[-1]
        if last_dim is None:
            last_dim = tf.shape(x)[-1]
        if x_ndims == 3:
            final_shape = [beam_size * batch_size, -1, last_dim]
            stacked_x = tf.reshape(tf.tile(x, [1, beam_size, 1]), final_shape)
            return stacked_x
        elif x_ndims == 2:
            final_shape = [batch_size * beam_size, last_dim]
            return tf.reshape(tf.tile(x, [1, beam_size]), final_shape)
        elif x_ndims == 1:
            return tf.reshape(
                tf.transpose(tf.tile([x], [beam_size, 1])), [-1])
        elif x_ndims == 4:
            assert x.get_shape().as_list()[1] == x.get_shape().as_list()[2] == 1, (
                "this only matches the bias tensor with shape [batch_size, 1, 1, timesteps]")
            return tf.expand_dims(
                _stack(tf.squeeze(x, axis=1)), axis=1)
        else:
            raise NotImplementedError("Not implemented the capability for ndims={}".format(x_ndims))

    return tf.nest.pack_sequence_as(
        tensors,
        tf.nest.map_structure(
            _stack, tf.nest.flatten(tensors)))


def tf1codebase_finished_beam_one_entry_bias(on_entry, num_entries, dtype=None):
    """ Builds a bias vector to be added to log_probs of a finished beam.

    The returned vector with shape [`num_entries`, ]. Only the `on_entry`
    has value 0, and the others are FLOAT_MIN.

    For example, on_entry=3 and num_entries=6 get the vector
    [FLOAT_MIN, FLOAT_MIN, FLOAT_MIN, 0.0, FLOAT_MIN, FLOAT_MIN]

    Args:
        on_entry: A python integer.
        num_entries: A python integer.
        dtype:

    Returns: A bias vector.
    """
    if dtype is None:
        dtype = compat.CUSTOM_GLOBAL_FLOATX
    bias = tf.one_hot(
        [on_entry], num_entries,
        on_value=0., off_value=compat.FLOAT_MIN)
    if dtype.base_dtype.name != "float32":
        bias = tf.cast(bias, dtype=dtype)
    return tf.squeeze(bias, axis=0)


def tf1codebase_unk_beam_one_entry_bias(on_entry, num_entries, dtype=None):
    """ Builds a bias vector to be added to log_probs of the UNK token in the vocab.

    The returned vector with shape [`num_entries`, ]. Only the `on_entry`
    has value FLOAT_MIN, and the others are 0.

    For example, on_entry=3 and num_entries=6 get the vector
    [FLOAT_MIN, FLOAT_MIN, FLOAT_MIN, 0.0, FLOAT_MIN, FLOAT_MIN]

    Args:
        on_entry: A python integer.
        num_entries: A python integer.
        dtype:

    Returns: A bias vector.
    """
    if dtype is None:
        dtype = compat.CUSTOM_GLOBAL_FLOATX
    bias = tf.one_hot(
        [on_entry], num_entries,
        on_value=compat.FLOAT_MIN, off_value=0.)
    if dtype.base_dtype.name != "float32":
        bias = tf.cast(bias, dtype=dtype)
    return tf.squeeze(bias, axis=0)


def tf1codebase_expand_to_beam_size(tensor, beam_size, axis=0):
    """ Stacks a given tensor `beam_size` times on a specific axis.

    For example, tensor=[1, 2, 3, 4], beam_size=3, axis=0 get the tensor
    [ [1, 2, 3, 4],
      [1, 2, 3, 4],
      [1, 2, 3, 4] ]

    tensor=[[1, 2, 3], [3, 4, 5]], beam_size=1, axis=1 get the tensor
    [ [[1, 2, 3]], [[3, 4, 5]] ]

    Args:
        tensor: A Tensor.
        beam_size: A python integer, the beam width.
        axis: A python integer.

    Returns: A Tensor.
    """
    tensor = tf.expand_dims(tensor, axis=axis)
    tile_dims = [1] * tensor.get_shape().ndims
    tile_dims[axis] = beam_size
    return tf.tile(tensor, tile_dims)


def test_fn_stack_beam_size():
    beam_size = 4
    batch_size = 3
    dim = 5
    time_step = 7
    num_heads = 8
    tensor_1d = tf.convert_to_tensor(numpy.random.randint(0, 10, (batch_size,)), dtype=tf.int32)
    tensor_2d = tf.convert_to_tensor(numpy.random.rand(batch_size, dim), dtype=tf.float32)
    tensor_3d = tf.convert_to_tensor(numpy.random.rand(batch_size, time_step, dim), dtype=tf.float32)
    tensor_4d_bias = tf.convert_to_tensor(numpy.random.rand(batch_size, 1, 1, dim), dtype=tf.float32)
    tensor_4d = tf.convert_to_tensor(numpy.random.rand(batch_size, time_step, num_heads, dim), dtype=tf.float32)
    tensor_3d_from_4d = tf.reshape(tensor_4d, [batch_size, time_step, -1])
    assert (stack_beam_size(tensor_1d, beam_size).numpy()
            == tf1codebase_stack_beam_size(tensor_1d, beam_size).numpy()).all()
    assert (stack_beam_size(tensor_2d, beam_size).numpy()
            == tf1codebase_stack_beam_size(tensor_2d, beam_size).numpy()).all()
    assert (stack_beam_size(tensor_3d, beam_size).numpy()
            == tf1codebase_stack_beam_size(tensor_3d, beam_size).numpy()).all()
    assert (stack_beam_size(tensor_4d_bias, beam_size).numpy()
            == tf1codebase_stack_beam_size(tensor_4d_bias, beam_size).numpy()).all()
    assert (stack_beam_size(tensor_4d, beam_size).numpy()
            == tf.reshape(tf1codebase_stack_beam_size(tensor_3d_from_4d, beam_size),
                          [-1, time_step, num_heads, dim]).numpy()).all()


def test_fn_one_entry_bias():
    vocab_size = 10
    eos_id = 9
    unk_id = 7
    assert (tf1codebase_finished_beam_one_entry_bias(on_entry=eos_id,
                                                     num_entries=vocab_size, dtype=tf.float32).numpy()
            == one_entry_bias(on_entry=eos_id, num_entries=vocab_size, on_value=0.,
                              off_value=compat.FLOAT_MIN, dtype=tf.float32).numpy()).all()
    assert (tf1codebase_unk_beam_one_entry_bias(on_entry=unk_id,
                                                num_entries=vocab_size, dtype=tf.float32).numpy()
            == one_entry_bias(on_entry=unk_id, num_entries=vocab_size, off_value=0.,
                              on_value=compat.FLOAT_MIN, dtype=tf.float32).numpy()).all()


def test_fn_expand_tensor():
    vocab_size = 10
    eos_id = 9
    batch_size = 3
    beam_size = 4
    batch_beam_size = batch_size * beam_size
    finished_beam_bias = tf1codebase_finished_beam_one_entry_bias(
        on_entry=eos_id, num_entries=vocab_size, dtype=tf.float32)
    assert (tf1codebase_expand_to_beam_size(finished_beam_bias,
                                            batch_beam_size, axis=0).numpy()
            == tile_tensor(finished_beam_bias,
                           batch_beam_size, axis=0).numpy()).all()


if __name__ == "__main__":
    test_fn_stack_beam_size()
    test_fn_one_entry_bias()
    test_fn_expand_tensor()
