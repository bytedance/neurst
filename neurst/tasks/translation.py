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

from neurst.data.dataset_utils import clean_dataset_by_length
from neurst.tasks import register_task
from neurst.tasks.seq2seq import Seq2Seq
from neurst.training.training_utils import EFFICIENT_MULTIPLIER, GPU_EFFICIENT_LEVEL, minimal_multiple
from neurst.utils import compat
from neurst.utils.configurable import deep_merge_dict
from neurst.utils.flags_core import Flag


def _auto_scale_batch_size(batch_size, level):
    if level == GPU_EFFICIENT_LEVEL.LEVEL1:
        return int(batch_size * 1.03)
    elif level == GPU_EFFICIENT_LEVEL.LEVEL2:
        return int(batch_size * 1.1)
    elif level == GPU_EFFICIENT_LEVEL.LEVEL3:
        return int(batch_size * 1.33)
    elif level == GPU_EFFICIENT_LEVEL.LEVEL4:
        return int(batch_size * 1.87)
    elif level == GPU_EFFICIENT_LEVEL.LEVEL5:
        return batch_size * 2


@register_task
class Translation(Seq2Seq):
    """ Defines the translation task. """

    @staticmethod
    def class_or_method_args():
        this_args = super(Translation, Translation).class_or_method_args()
        this_args.extend([
            Flag("gpu_efficient_level", dtype=Flag.TYPE.INTEGER, default=GPU_EFFICIENT_LEVEL.LEVEL1,
                 choices=tuple(GPU_EFFICIENT_LEVEL),
                 help="The efficient level for training using XLA, from 0~5."),
            Flag("auto_scaling_batch_size", dtype=Flag.TYPE.BOOLEAN, default=None,
                 help="Whether to automatically scale up the batch size to match the real tokens "
                      "when `gpu_efficient_level` > 0")
        ])
        return this_args

    def create_and_batch_tfds(self, ds, mode, args=None, num_replicas_in_sync=1):
        """ With efficient level for training. """
        if args is None:
            args = self._args
        else:
            args = deep_merge_dict(self._args, args)
        level = args.get("gpu_efficient_level", None)
        auto_scale_batch = args.get("auto_scaling_batch_size", None)
        if (mode in [compat.ModeKeys.INFER, compat.ModeKeys.EVAL]
            or level is None or level == GPU_EFFICIENT_LEVEL.LEVEL0):
            return super(Translation, self).create_and_batch_tfds(
                ds, mode, args, num_replicas_in_sync)
        padding_values = {"feature": tf.constant(self._src_data_pipeline.meta["eos_id"], dtype=tf.int64),
                          "label": tf.constant(self._trg_data_pipeline.meta["eos_id"], dtype=tf.int64)}
        dataset = ds.build(auto_shard=True,
                           map_func=self.get_data_preprocess_fn(mode, ds.status, args),
                           map_output_dtypes=self.inputs_signature(mode)[0])
        max_src_len = args["max_src_len"]
        max_trg_len = args["max_trg_len"]
        batch_by_tokens = args["batch_by_tokens"]
        assert max_src_len, "Must provide `max_src_len` when `gpu_efficient_level` > 0"
        assert max_trg_len, "Must provide `max_trg_len` when `gpu_efficient_level` > 0"
        logging.info(f"Creating training dataset with `gpu_efficient_level`={level}.")
        dataset = clean_dataset_by_length(dataset, {"feature": max_src_len, "label": max_trg_len})
        if args["cache_dataset"]:
            dataset = dataset.cache()
        if args["shuffle_buffer"]:
            dataset = dataset.shuffle(buffer_size=args["shuffle_buffer"])
        batch_size_per_gpu = args["batch_size_per_gpu"]
        batch_size = args["batch_size"]
        if batch_size_per_gpu is None:
            batch_size_per_gpu = batch_size // num_replicas_in_sync
        if batch_by_tokens:
            assert batch_size_per_gpu > max(max_src_len, max_trg_len), (
                f"batch size per gpu({batch_size_per_gpu} must be greater than "
                f"both `max_src_len`{max_src_len} and `max_trg_len`{max_trg_len}")
        if auto_scale_batch:
            new_batch_size_per_gpu = _auto_scale_batch_size(batch_size_per_gpu, level)
            logging.info(f"Auto scaling `batch_size_per_gpu` from {batch_size_per_gpu} "
                         f"to {new_batch_size_per_gpu}")
            batch_size_per_gpu = new_batch_size_per_gpu
        max_src_len = minimal_multiple(max_src_len, EFFICIENT_MULTIPLIER[level])
        max_trg_len = minimal_multiple(max_trg_len, EFFICIENT_MULTIPLIER[level])
        max_len = max(max_src_len, max_trg_len)
        if level == GPU_EFFICIENT_LEVEL.LEVEL5:  # static batch
            if batch_by_tokens:
                batch_size_per_gpu = batch_size_per_gpu // max_len
            return dataset.padded_batch(
                int(minimal_multiple(batch_size_per_gpu, EFFICIENT_MULTIPLIER[level]) * num_replicas_in_sync),
                padded_shapes={"feature": [max_src_len], "label": [max_trg_len]},
                drop_remainder=True, padding_values=padding_values)
        else:
            bucket_boundaries = [EFFICIENT_MULTIPLIER[level] * i for i in
                                 range(1, max_len // EFFICIENT_MULTIPLIER[level] + 1)]
            if bucket_boundaries[-1] < max_len:
                bucket_boundaries.append(minimal_multiple(bucket_boundaries[-1] + 1,
                                                          EFFICIENT_MULTIPLIER[level]))
            buckets_min = [0] + bucket_boundaries[:-1]
            if batch_by_tokens:
                bucket_batch_sizes = [int(minimal_multiple(batch_size_per_gpu // bound,
                                                           EFFICIENT_MULTIPLIER[level])
                                          * num_replicas_in_sync) for bound in bucket_boundaries]
            else:
                bucket_batch_sizes = [int(minimal_multiple(batch_size_per_gpu, EFFICIENT_MULTIPLIER[level])
                                          * num_replicas_in_sync)] * len(bucket_boundaries)

            logging.info(f"There are {len(bucket_batch_sizes)} input shapes to be compiled:")
            for batc, bound in zip(bucket_batch_sizes, bucket_boundaries):
                logging.info(f"   - batch={batc}, maximum-length={bound}")
            bucket_batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)
            bucket_boundaries = tf.constant(bucket_boundaries, dtype=tf.int32)

            def example_to_bucket_id(examples):
                """Return int64 bucket id for this example, calculated based on length."""
                seq_length = tf.cast(tf.maximum(tf.size(examples["feature"]),
                                                tf.size(examples["label"])), tf.int32)

                conditions_c = tf.logical_and(
                    tf.less(buckets_min, seq_length),
                    tf.less_equal(seq_length, bucket_boundaries))
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
                    bucket_batch_size,
                    padded_shapes={"feature": [bucket_boundaries[bucket_id]],
                                   "label": [bucket_boundaries[bucket_id]]},
                    padding_values=padding_values, drop_remainder=True)

            return dataset.apply(tf.data.experimental.group_by_window(
                key_func=example_to_bucket_id,
                reduce_func=batching_fn,
                window_size=None,
                window_size_func=window_size_fn))
