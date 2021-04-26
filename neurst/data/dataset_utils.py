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
import math
import os

import tensorflow as tf
from absl import logging

from neurst.utils.compat import get_distributed_worker_setting
from neurst.utils.misc import deprecated, flatten_string_list

_MIN_BUCKET_BOUNDARY = 8
_BUCKET_BOUNDARY_SCALE = 1.1
_MAX_BUCKET_BOUNDARY = 256


def map_data_for_keras(dataset):
    """ Maps data for training.
        For TF v2, the 2nd parameter is omitted to make Keras training work.

    Args:
        dataset: A tf.data.Dataset object.

    Returns:
        A tf.data.Dataset object.
    """

    def _fn(*args):
        return (args,)

    return dataset.map(
        _fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


@deprecated
def _batch_examples_by_token(dataset,
                             batch_size,
                             bucket_boundaries,
                             padding_values,
                             padding_length,
                             example_length_func,
                             drop_remainder=True,
                             num_replicas_in_sync=1):
    """Group examples by similar lengths, and return batched dataset.

    Each batch of similar-length examples are padded to the same length, and may
    have different number of elements in each batch, such that:
      group_batch_size * padded_length <= batch_size.

    This decreases the number of padding tokens per batch, which improves the
    training speed.

    Args:
        dataset: Dataset of unbatched examples.
        batch_size: Max number of tokens per batch of examples.
        bucket_boundaries: A list of integers of the boundaries of each bucket.
        padding_values: A tuple of constants for padding.
        padding_length: A list/tuple of padding length, which will be passed to padded_decode.
        example_length_func: A callable function, which deduces the input examples to the maximum length.
        drop_remainder: Whether the last batch should be dropped in the case it has fewer than batch_size.
        num_replicas_in_sync: The number of GPUs or other workers. We will generate
            global batches, and each global batch is equally divisible by number of replicas.

    Returns:
      Dataset of batched examples with similar lengths.
    """
    # Get min and max boundary lists for each example. These are used to calculate
    # the `bucket_id`, which is the index at which:
    # buckets_min[bucket_id] <= len(example) < buckets_max[bucket_id]
    # Note that using both min and max lists improves the performance.
    buckets_min = [0] + bucket_boundaries[:-1]
    buckets_max = bucket_boundaries

    # Create list of batch sizes for each bucket_id, so that
    # bucket_batch_size[bucket_id] * buckets_max[bucket_id] <= batch_size
    bucket_batch_sizes = [batch_size // x // num_replicas_in_sync * num_replicas_in_sync
                          for x in buckets_max]

    # bucket_id will be a tensor, so convert this list to a tensor as well.
    bucket_batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)

    def example_to_bucket_id(examples):
        """Return int64 bucket id for this example, calculated based on length."""
        seq_length = tf.cast(example_length_func(examples), tf.int32)

        conditions_c = tf.logical_and(
            tf.less_equal(buckets_min, seq_length),
            tf.less(seq_length, buckets_max))
        bucket_id = tf.reduce_min(tf.where(conditions_c))
        return bucket_id

    def window_size_fn(bucket_id):
        """Return number of examples to be grouped when given a bucket id."""
        return bucket_batch_sizes[bucket_id]

    def batching_fn(bucket_id, grouped_dataset):
        """Batch and add padding to a dataset of elements with similar lengths."""
        bucket_batch_size = window_size_fn(bucket_id)

        # Batch the dataset and add padding so that all input sequences in the
        # examples have the same length, and all target sequences have the same
        # lengths as well. Resulting lengths of inputs and targets can differ.
        return grouped_dataset.padded_batch(
            bucket_batch_size, padding_length,
            padding_values=padding_values, drop_remainder=drop_remainder)

    return dataset.apply(tf.data.experimental.group_by_window(
        key_func=example_to_bucket_id,
        reduce_func=batching_fn,
        window_size=None,
        window_size_func=window_size_fn))


def create_batch_bucket_boundaries(max_length,
                                   min_boundary=_MIN_BUCKET_BOUNDARY,
                                   boundary_scale=_BUCKET_BOUNDARY_SCALE):
    """ Creates training batch bucket boundaries.

    Args:
        max_length: The maximum length of example in dataset.
        min_boundary: Minimum length in boundary.
        boundary_scale: Amount to scale consecutive boundaries in the list.

    Returns:
        A list of bucket boundaries.
    """
    # Create bucket boundaries list by scaling the previous boundary or adding 1
    # (to ensure increasing boundary sizes).
    bucket_boundaries = []
    x = min_boundary
    while x < max_length:
        bucket_boundaries.append(x)
        x = max(x + 1, int(x * boundary_scale))
    if bucket_boundaries[-1] < max_length + 1:
        bucket_boundaries = bucket_boundaries + [max_length + 1]
    return bucket_boundaries


def associated_bucket_boundaries(a, b):
    """ Creates training batch bucket boundaries.

    Args:
        a: A list of bucket boundaries.
        b: Another list of bucket boundaries.

    Returns:
        Two refactored lists of bucket boundaries with the same size.
    """
    length1 = len(a)
    length2 = len(b)
    if length1 == length2:
        return a, b
    elif length1 > length2:
        step_size1 = length1 * 1. / length2
        step_size2 = 1
    else:
        step_size1 = 1
        step_size2 = length2 * 1. / length1
    new_boundaries1 = []
    new_boundaries2 = []
    i = 1
    while i < min(length1, length2) + 1:
        new_boundaries1.append(a[int(math.ceil(i * step_size1)) - 1])
        new_boundaries2.append(b[int(math.ceil(i * step_size2)) - 1])
        i += 1

    return new_boundaries1, new_boundaries2


@deprecated
def load_from_tfrecord_and_auto_shard(features_file, shuffle=True,
                                      example_parse_fn=None, deterministic=True):
    """ Loads TFRecords and does autot-sharding according to worker num.

    Args:
        features_file: The TFRecords file path.
        shuffle: Whether to shuffle files.
        example_parse_fn: The example parse function for TF Record.
        deterministic: Whether the outputs need to be produced in deterministic order.

    Returns: A dataset.
    """
    _files = features_file.split(",")
    _features_files = []
    for _file in _files:
        if tf.io.gfile.isdir(_file):
            _features_files.append(os.path.join(_file, "*train*"))
        elif tf.io.gfile.exists(_file):
            _features_files.append(_file)
        else:
            _features_files.append(_file + "*")
    logging.info("Load TFRecords from {}".format(str(_features_files)))
    dataset = tf.data.Dataset.list_files(_features_files, shuffle=shuffle)
    # auto sharding
    worker_id, num_workers, strategy = get_distributed_worker_setting()
    if num_workers > 1 and strategy in ["horovod", "byteps"] and not shuffle:
        logging.info("Shard %d of the whole dataset(total %d workers).", worker_id, num_workers)
        dataset = dataset.shard(num_workers, worker_id)
    # Read files and interleave results.
    # When training, the order of the examples will be non-deterministic.
    options = tf.data.Options()
    options.experimental_deterministic = deterministic
    dataset = dataset.interleave(
        lambda f: tf.data.TFRecordDataset(f, buffer_size=32 * 1024 * 1024),
        cycle_length=10,
        num_parallel_calls=tf.data.experimental.AUTOTUNE).with_options(options)
    if example_parse_fn is None:
        return dataset
    return dataset.map(example_parse_fn,
                       num_parallel_calls=tf.data.experimental.AUTOTUNE)


def parse_tfexample(serialized_example, name_to_features,
                    name_mapping=None, map_func=None,
                    auxiliary_elements=None):
    """ Parses TF example from TF Record. """
    parsed = tf.io.parse_single_example(serialized_example, name_to_features)
    elements = {}
    for k, v in parsed.items():
        if name_mapping is None or k not in name_mapping:
            elements[k] = tf.sparse.to_dense(v)
        else:
            elements[name_mapping[k]] = tf.sparse.to_dense(v)
    if isinstance(auxiliary_elements, dict):
        elements.update(auxiliary_elements)

    if map_func is None:
        return elements
    return map_func(elements)


def glob_tfrecords(file_path):
    _files = flatten_string_list(file_path)
    _features_files = []
    for _file in _files:
        if tf.io.gfile.isdir(_file):
            _features_files.extend(tf.io.gfile.glob(os.path.join(_file, "*train*")))
        elif tf.io.gfile.exists(_file):
            _features_files.append(_file)
        else:
            _features_files.extend(tf.io.gfile.glob(_file + "*"))
    return _features_files


def load_tfrecords(file_path,
                   name_to_features,
                   shuffle=False,
                   deterministic=True,
                   feature_name_mapping=None,
                   map_func=None,
                   sharding_index=0,
                   num_shards=1,
                   auto_shard=False,
                   auxiliary_elements=None) -> tf.data.Dataset:
    """ Loads TFRecords and does autot-sharding according to worker num.

    Args:
        file_path: The TFRecords file path.
        name_to_features: A `dict` mapping feature keys to `FixedLenFeature` or
            `VarLenFeature` values.
        shuffle: Whether to shuffle files.
        deterministic: Whether the outputs need to be produced in deterministic order.
        feature_name_mapping: A dict that maps the names in `name_to_features` to aliases.
        map_func: A callable function to process the data.
        sharding_index: The manually defined index for sharding.
        num_shards: The manually defined number of shards operating in parallel.
        auto_shard: Automatically shard the TFRecord parts if True.
        auxiliary_elements: A dict containing auxiliary elements that will
            append to the data sample.

    Returns: A dataset.
    """
    _features_files = []
    for _file in flatten_string_list(file_path):
        if tf.io.gfile.isdir(_file):
            _features_files.append(os.path.join(_file, "*train*"))
        elif tf.io.gfile.exists(_file):
            _features_files.append(_file)
        else:
            _features_files.append(_file + "*")
    # shuffle = (shuffle is True) and (num_shards == 1)
    # dataset = tf.data.Dataset.list_files(_features_files, shuffle=shuffle)
    dataset = tf.data.Dataset.list_files(_features_files, shuffle=False)
    if num_shards > 1:
        logging.info("Shard %d of the whole dataset(total %d workers).", sharding_index, num_shards)
        dataset = dataset.shard(num_shards, sharding_index)
    else:
        # auto sharding
        worker_id, num_workers, strategy = get_distributed_worker_setting()
        if num_workers > 1 and strategy in ["horovod", "byteps"] and auto_shard:
            logging.info("Shard %d of the whole dataset(total %d workers).", worker_id, num_workers)
            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
            dataset = dataset.with_options(options)
            dataset = dataset.shard(num_workers, worker_id)
    logging.info("Loading TF Records from: ")
    if shuffle:
        dataset = dataset.shuffle(5000)
    for _f in dataset:
        logging.info(f"   {_f.numpy()}")
    # Read files and interleave results.
    # When training, the order of the examples will be non-deterministic.
    options = tf.data.Options()
    options.experimental_deterministic = deterministic
    dataset = dataset.interleave(
        lambda f: tf.data.TFRecordDataset(f, buffer_size=128 * 1024 * 1024),
        cycle_length=10,
        num_parallel_calls=tf.data.experimental.AUTOTUNE).with_options(options)

    if name_to_features is None:
        return dataset
    return dataset.map(lambda x: parse_tfexample(x, name_to_features, feature_name_mapping, map_func,
                                                 auxiliary_elements=auxiliary_elements),
                       num_parallel_calls=tf.data.experimental.AUTOTUNE)


def clean_dataset_by_length(dataset, data_max_lengths):
    """ Filters empty datas, or datas exceeded max length. """
    logging.info(f"Filtering empty data and datas exceeded max length={data_max_lengths}")
    return dataset.filter(
        lambda data_sample: tf.reduce_all([
            (length == -1 or length is None
             or tf.less_equal(tf.size(data_sample[k]), length))  # filter by max length
            and (length == -1 or (length != -1 and tf.size(data_sample[k]) > 1))  # filter out empty lines
            for k, length in data_max_lengths.items()
        ]))


@deprecated
def batch_sequential_dataset(dataset,
                             padding_values,
                             example_length_func=None,
                             batch_size=None,
                             batch_size_per_gpu=None,
                             batch_by_tokens=False,
                             bucket_boundaries=None,
                             data_max_lengths=None,
                             shuffer_buffer=0,
                             drop_remainder=True,
                             num_replicas_in_sync=1):
    """ Calls padded_batch under special settings for sequential dataset.

    Args:
        dataset: A parallel dataset.
        padding_values: A list of padding values, will be passed to dataset.padded_batch.
        example_length_func: A callable function that takes a dict as input and returns
            the "length" of this data sample.
        batch_size: The number of sentences or word tokens according to `batch_by_tokens`.
        batch_size_per_gpu: The per-GPU batch size.
        batch_by_tokens: A bool, whether to batch the data by word tokens.
        bucket_boundaries: A list integers indicating the boundaries of the bucket when
            `batch_by_tokens` is True.
        data_max_lengths: The maximum length of training data, None or a list/tuple of
            integers with the the size as data samples. -1 indicates scalar data with
            no 'length' checking.
        shuffer_buffer: The buffer size for shuffling.
        drop_remainder: Whether the last batch should be dropped in the case it has fewer than batch_size.
        num_replicas_in_sync: The number of GPUs or other workers. We will generate global
                batches, and each global batch is equally divisible by number of replicas.

    Returns:
        The batched dataset.
    """
    if data_max_lengths is None:
        data_max_lengths = {k: None for k in padding_values}
    assert len(data_max_lengths) == len(padding_values)

    if example_length_func is None:
        def example_length_func(examples):
            return tf.reduce_max([
                tf.size(examples[k]) for k, length in data_max_lengths.items() if length != -1])

    if batch_size is None and batch_size_per_gpu is None:
        raise ValueError("Either `batch_size` or `batch_size_per_gpu` needs to be provided.")
    elif batch_size is not None and batch_size_per_gpu is not None:
        logging.info("Both `batch_size` and `batch_size_per_gpu` are provided, use `batch_size_per_gpu`.")
    if batch_size_per_gpu is not None:
        batch_size = int(batch_size_per_gpu * num_replicas_in_sync)
    logging.info("The global batch size is {}, with batch_by_tokens={}".format(batch_size, batch_by_tokens))
    # filter out empty lines
    dataset = clean_dataset_by_length(dataset, data_max_lengths)
    dynamic_padding_length = {k: ([] if length == -1 else [None])
                              for k, length in data_max_lengths.items()}

    if batch_by_tokens:
        # shuffle
        if shuffer_buffer:
            dataset = dataset.shuffle(buffer_size=shuffer_buffer)
        max_length = max(max([_len or 0 for _len in data_max_lengths.values()]), 0)

        if not max_length:
            logging.info("Using pre-defined max length={}".format(_MAX_BUCKET_BOUNDARY))
            max_length = _MAX_BUCKET_BOUNDARY
        logging.info("Final check of the max length of the training data. "
                     "Filter out whose length is larger than {}".format(max_length))
        dataset = dataset.filter(
            lambda data_sample: tf.reduce_all([
                (length == -1) or (length is None) or tf.size(data_sample[k]) <= max_length
                for k, length in data_max_lengths.items()]))
        if bucket_boundaries is None:
            bucket_boundaries = create_batch_bucket_boundaries(max_length)

        return _batch_examples_by_token(
            dataset,
            batch_size=batch_size,
            drop_remainder=drop_remainder,
            padding_values=padding_values,
            padding_length=dynamic_padding_length,
            bucket_boundaries=bucket_boundaries,
            example_length_func=example_length_func,
            num_replicas_in_sync=num_replicas_in_sync)
    else:
        # shuffle
        if shuffer_buffer:
            dataset = dataset.shuffle(buffer_size=shuffer_buffer)
        padding_length = dynamic_padding_length
    logging.info("The padding length of the dataset is {}".format(padding_length))
    dataset = dataset.padded_batch(
        int(batch_size // num_replicas_in_sync * num_replicas_in_sync),
        padding_length, drop_remainder=drop_remainder, padding_values=padding_values)
    return dataset


def adjust_batch_size(batch_size=None,
                      batch_size_per_gpu=None,
                      bucket_boundaries=None,
                      boundaries_reduce_to_length_fn=None,
                      num_replicas_in_sync=1,
                      verbose=True):
    if batch_size is None and batch_size_per_gpu is None:
        raise ValueError("At least one of the `batch_size` and `batch_size_per_gpu` should be provided.")
    elif batch_size is not None and batch_size_per_gpu is not None:
        logging.info("Both `batch_size` and `batch_size_per_gpu` are provided, use `batch_size_per_gpu`.")
    if batch_size_per_gpu is not None:
        batch_size = int(batch_size_per_gpu * num_replicas_in_sync)
    if bucket_boundaries is None:
        batch_size = int(batch_size // num_replicas_in_sync * num_replicas_in_sync)
        if verbose:
            logging.info(f"The global batch size is {batch_size} samples.")
        return batch_size
    logging.info(f"The global batch size is {batch_size} tokens.")
    bucket_batch_sizes = []
    try:
        i = 0
        while True:
            bucket_batch_sizes.append(
                int(batch_size // boundaries_reduce_to_length_fn({k: v[i] for k, v in bucket_boundaries.items()})
                    // num_replicas_in_sync * num_replicas_in_sync))
            i += 1

    except IndexError:
        pass
    return bucket_batch_sizes


def batch_examples_by_token(dataset,
                            bucket_boundaries,
                            bucket_batch_sizes,
                            padding_values,
                            example_length_func,
                            extra_padded_shapes=None,
                            drop_remainder=True):
    """Group examples by similar lengths, and return batched dataset.

    Each batch of similar-length examples are padded to the same length, and may
    have different number of elements in each batch, such that:
      group_batch_size * padded_length <= batch_size.

    This decreases the number of padding tokens per batch, which improves the
    training speed.

    Args:
        dataset: Dataset of unbatched examples.
        bucket_batch_sizes: Max number of tokens per batch of examples or a list of batch size for each bucket.
        bucket_boundaries: A list of integers of the boundaries of each bucket.
        padding_values: A tuple of constants for padding.
        example_length_func: A callable function, which deduces the input examples to the maximum length.
        extra_padded_shapes: A dict containing extra shapes (not included in bucket boundaries) for padding.
        drop_remainder: Whether the last batch should be dropped in the case it has fewer than batch_size.

    Returns:
      Dataset of batched examples with similar lengths.
    """
    cnt = 0
    try:
        logging.info("The details of batching logic:")
        while True:
            _batch = bucket_batch_sizes
            if isinstance(bucket_batch_sizes, list):
                _batch = bucket_batch_sizes[cnt]
            _bounds = {k: v[cnt] for k, v in bucket_boundaries.items()}
            logging.info(f"   - batch={_batch}, bucket boundary={_bounds}")
            cnt += 1
    except IndexError:
        logging.info(f"  Total {cnt} input shapes are compiled.")
    if not isinstance(bucket_batch_sizes, list):
        bucket_batch_sizes = [bucket_batch_sizes] * cnt
    # bucket_id will be a tensor, so convert this list to a tensor as well.
    bucket_batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)
    bucket_boundaries = {k: tf.constant(v, dtype=tf.int32) for k, v in bucket_boundaries.items()}

    def example_to_bucket_id(examples):
        """Return int64 bucket id for this example, calculated based on length."""
        seq_length = example_length_func(examples)

        conditions_c = tf.reduce_all([
            tf.less_equal(v, bucket_boundaries[k])
            for k, v in seq_length.items()], axis=0)
        bucket_id = tf.reduce_min(tf.where(conditions_c))
        return bucket_id

    def window_size_fn(bucket_id):
        """Return number of examples to be grouped when given a bucket id."""
        return bucket_batch_sizes[bucket_id]

    def batching_fn(bucket_id, grouped_dataset):
        """Batch and add padding to a dataset of elements with similar lengths."""
        bucket_batch_size = window_size_fn(bucket_id)
        padded_shapes = {k: [v[bucket_id]] for k, v in bucket_boundaries.items()}
        if extra_padded_shapes:
            for k, v in extra_padded_shapes.items():
                padded_shapes[k] = v

        # Batch the dataset and add padding so that all input sequences in the
        # examples have the same length, and all target sequences have the same
        # lengths as well. Resulting lengths of inputs and targets can differ.
        return grouped_dataset.padded_batch(
            bucket_batch_size, padded_shapes,
            padding_values=padding_values,
            drop_remainder=drop_remainder)

    return dataset.apply(tf.data.experimental.group_by_window(
        key_func=example_to_bucket_id,
        reduce_func=batching_fn,
        window_size=None,
        window_size_func=window_size_fn))


def take_one_record(data_path):
    _file_path = flatten_string_list(data_path)[0]
    if tf.io.gfile.isdir(_file_path):
        _feature_file = os.path.join(_file_path, "*train*")
    elif tf.io.gfile.exists(_file_path):
        _feature_file = _file_path
    else:
        _feature_file = _file_path + "*"
    dataset = tf.data.Dataset.list_files([_feature_file], shuffle=False)
    dataset = dataset.interleave(
        lambda f: tf.data.TFRecordDataset(f, buffer_size=128 * 1024 * 1024),
        cycle_length=10,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    for x in dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(x.numpy())
        return example
