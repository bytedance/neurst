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
import functools
import inspect
import multiprocessing
import os
import tempfile
from distutils.version import LooseVersion
from urllib.request import urlretrieve

import numpy
import tensorflow as tf
from absl import logging


def deprecated(substitution):
    """This is a decorator which can be used to mark functions or classes
    as deprecated. It will result in a warning being emmitted
    when the function/class is used."""

    cls_warn = "Call to deprecated class `{name}`. "
    func_warn = "Call to deprecated function `{name}`. "
    if isinstance(substitution, str):

        def decorator(func):
            warn_info = cls_warn if inspect.isclass(func) else func_warn
            warn_info += "Please use `{substitution}` instead."

            @functools.wraps(func)
            def new_func(*args, **kwargs):
                logging.warning(warn_info.format(name=func.__name__, substitution=substitution))
                return func(*args, **kwargs)

            return new_func

        return decorator

    elif inspect.isclass(substitution) or inspect.isfunction(substitution):
        func = substitution
        warn_info = cls_warn if inspect.isclass(func) else func_warn

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            logging.warning(warn_info.format(name=func.__name__))
            return func(*args, **kwargs)

        return new_func

    else:
        raise TypeError(repr(type(substitution)))


def flatten_string_list(arg):
    """ Flattens a string list.

    Args:
        arg: A list of string or a string. The string may contains comma as a separator.

    Returns: A list of string.
    """
    if arg is None:
        return None
    return [c.strip() for cs in tf.nest.flatten(arg) for c in cs.split(",") if c.strip()]


class DummyContextManager(object):

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


def to_numpy_or_python_type(tensors, bytes_as_str=False):
    """Converts a structure of `Tensor`s to `NumPy` arrays or Python scalar types.

    For each tensor, it calls `tensor.numpy()`. If the result is a scalar value,
    it converters it to a Python type, such as a float or int, by calling
    `result.item()`.

    Numpy scalars are converted, as Python types are often more convenient to deal
    with. This is especially useful for bfloat16 Numpy scalars, which don't
    support as many operations as other Numpy values.

    Args:
      tensors: A structure of tensors.
      bytes_as_str: A boolean, whether to convert bytes elements to string.

    Returns:
      `tensors`, but scalar tensors are converted to Python types and non-scalar
      tensors are converted to Numpy arrays.
    """

    def _to_single_numpy_or_python_type(t):
        if LooseVersion(tf.__version__) < LooseVersion("2.4"):
            is_tf_tensor = isinstance(t, tf.Tensor)
        else:
            is_tf_tensor = tf.is_tensor(t)
        if is_tf_tensor:
            x = t.numpy()
            if numpy.ndim(x) == 0:
                x = x.item()
                if bytes_as_str and isinstance(x, bytes):
                    return x.decode("utf-8")
                return x
            else:
                if len(x) == 0:
                    return x
                if bytes_as_str and isinstance(x.flatten()[0], bytes):
                    return tf.nest.map_structure(lambda _x: _x.decode("utf-8"), x.tolist())[0]
                return x
        return t  # Don't turn ragged or sparse tensors to NumPy.

    return tf.nest.map_structure(_to_single_numpy_or_python_type, tensors)


class PseudoPool(object):
    def __init__(self, processes=1):
        """ If processes is 1, then don't create pool.

        Args:
            processes:
        """
        self.pool = None
        if processes > 1:
            self.pool = multiprocessing.Pool(processes=processes)
        self.processes = processes

    @staticmethod
    def parse_arg_list(n_threads, sample_list, *args):
        total_length = len(sample_list)
        samples_per_thread = total_length // n_threads + 1
        if samples_per_thread < 1000:
            samples_per_thread = total_length
        start_idx = 0
        ret = []
        while start_idx < total_length:
            arg_list = []
            arg_list.append(sample_list[start_idx:(start_idx + samples_per_thread)])
            for arg in args:
                if isinstance(arg, list) and len(arg) == len(sample_list):
                    arg_list.append(arg[start_idx:(start_idx + samples_per_thread)])
                elif not isinstance(arg, list):
                    arg_list.append(arg)
                else:
                    raise ValueError
            start_idx += samples_per_thread
            ret.append(arg_list)
        return ret

    def map(self, func, args_list):
        return [func(args) for args in args_list]

    def __enter__(self):
        if self.processes > 1:
            return self.pool
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.processes > 1:
            self.pool.terminate()


def download_with_tqdm(url, filename):
    from tqdm import tqdm

    class TqdmUpTo(tqdm):
        last_block = 0

        def update_to(self, block_num=1, block_size=1, total_size=None):
            if total_size is not None:
                self.total = total_size
            self.update((block_num - self.last_block) * block_size)
            self.last_block = block_num

    try:
        os.remove(filename)
    except FileNotFoundError:
        pass

    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024,
                  miniters=1, desc=filename) as t:
        urlretrieve(url, filename, reporthook=t.update_to, data=None)


def temp_download(url):
    tmpfile = tempfile.NamedTemporaryFile("w", delete=False)
    download_with_tqdm(url, tmpfile.name)
    inmemory_name = "ram://" + os.path.basename(tmpfile.name)
    with tf.io.gfile.GFile(inmemory_name, "wb") as fw, tf.io.gfile.GFile(tmpfile.name, "rb") as fp:
        fw.write(fp.read())
    os.remove(tmpfile.name)
    return inmemory_name


def assert_equal_numpy(tensor_a, tensor_b, epsilon=1e-6):
    """

    Args:
        tensor_a: A numpy.ndarray
        tensor_b: A numpy.ndarray
        epsilon: The epsilon
    """
    assert tensor_a.shape == tensor_b.shape
    diff = numpy.sqrt(numpy.sum((tensor_a - tensor_b) ** 2))
    assert diff < epsilon, diff
