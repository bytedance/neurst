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

from neurst.data import dataset_utils
from neurst.tasks import register_task
from neurst.tasks.seq2seq import Seq2Seq
from neurst.training.training_utils import (EFFICIENT_MULTIPLIER, GPU_EFFICIENT_LEVEL, maximum_lower_multiple,
                                            minimal_multiple)
from neurst.utils import compat
from neurst.utils.configurable import deep_merge_dict
from neurst.utils.flags_core import Flag


def _auto_scale_batch_size(batch_size, level):
    if level == GPU_EFFICIENT_LEVEL.LEVEL0:
        return batch_size
    elif level == GPU_EFFICIENT_LEVEL.LEVEL1:
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
            Flag("gpu_efficient_level", dtype=Flag.TYPE.INTEGER, default=GPU_EFFICIENT_LEVEL.LEVEL0,
                 choices=tuple(GPU_EFFICIENT_LEVEL),
                 help="The efficient level for training using XLA, from 0~5."),
            Flag("auto_scaling_batch_size", dtype=Flag.TYPE.BOOLEAN, default=None,
                 help="Whether to automatically scale up the batch size to match the real tokens "
                      "when `gpu_efficient_level` > 0")
        ])
        return this_args

    def create_and_batch_tfds(self, ds, mode, args=None, num_replicas_in_sync=1):
        """ With efficient level for training. """
        if mode in [compat.ModeKeys.INFER, compat.ModeKeys.EVAL]:
            return super(Translation, self).create_and_batch_tfds(
                ds, mode, args, num_replicas_in_sync)
        if args is None:
            args = self._args
        else:
            args = deep_merge_dict(self._args, args, local_overwrite=False)
        level = args.get("gpu_efficient_level", None)
        auto_scale_batch = args.get("auto_scaling_batch_size", None)
        logging.info(f"Creating training dataset with GPU efficient level={level}.")
        dataset = ds.build(map_func=self.get_data_preprocess_fn(mode, ds.status, args),
                           map_output_dtypes=self.inputs_signature(mode)[0],
                           auto_shard=True, shuffle=True)
        dataset = dataset_utils.clean_dataset_by_length(
            dataset, {"feature": args["max_src_len"], "label": args["max_trg_len"]})
        if args["cache_dataset"]:
            dataset = dataset.cache()
        if args["shuffle_buffer"]:
            dataset = dataset.shuffle(buffer_size=args["shuffle_buffer"])
        padding_values = {"feature": tf.constant(self._src_data_pipeline.meta["pad_id"], dtype=tf.int64),
                          "label": tf.constant(self._trg_data_pipeline.meta["pad_id"], dtype=tf.int64)}
        if args["max_src_len"] is None:
            raise RuntimeError("Must provide `max_src_len` for training.")
        if args["max_trg_len"] is None:
            raise RuntimeError("Must provide `max_trg_len` for training.")
        max_src_len = minimal_multiple(args["max_src_len"], EFFICIENT_MULTIPLIER[level])
        max_trg_len = minimal_multiple(args["max_trg_len"], EFFICIENT_MULTIPLIER[level])
        max_len = max(max_src_len, max_trg_len)
        batch_size = dataset_utils.adjust_batch_size(args["batch_size"], args["batch_size_per_gpu"],
                                                     num_replicas_in_sync=num_replicas_in_sync,
                                                     verbose=False)
        if auto_scale_batch:
            batch_size = _auto_scale_batch_size(batch_size, level)
            logging.info(f"Auto scaling batch size to {batch_size}.")
        if level == GPU_EFFICIENT_LEVEL.LEVEL5:  # static batch
            _batch_size = batch_size
            if args["batch_by_tokens"]:
                _batch_size = _batch_size // max_len
            logging.info("Batching dataset with fixed shape: "
                         f"batch={_batch_size} x (feature={max_src_len}, label={max_trg_len}).")
            return dataset.padded_batch(
                _batch_size // num_replicas_in_sync * num_replicas_in_sync,
                padded_shapes={"feature": [max_src_len], "label": [max_trg_len]},
                drop_remainder=True, padding_values=padding_values)
        else:
            src_bucket_boundaries = [EFFICIENT_MULTIPLIER[level] * i for i in
                                     range(1, max_src_len // EFFICIENT_MULTIPLIER[level] + 1)]
            if src_bucket_boundaries[-1] < max_src_len:
                src_bucket_boundaries.append(minimal_multiple(src_bucket_boundaries[-1] + 1,
                                                              EFFICIENT_MULTIPLIER[level]))
            trg_bucket_boundaries = [EFFICIENT_MULTIPLIER[level] * i for i in
                                     range(1, max_trg_len // EFFICIENT_MULTIPLIER[level] + 1)]
            if trg_bucket_boundaries[-1] < max_trg_len:
                trg_bucket_boundaries.append(minimal_multiple(trg_bucket_boundaries[-1] + 1,
                                                              EFFICIENT_MULTIPLIER[level]))
            src_bucket_boundaries, trg_bucket_boundaries = dataset_utils.associated_bucket_boundaries(
                src_bucket_boundaries, trg_bucket_boundaries)
            bucket_boundaries = {
                "feature": src_bucket_boundaries,
                "label": trg_bucket_boundaries
            }
            bucket_batch_sizes = dataset_utils.adjust_batch_size(
                batch_size,
                bucket_boundaries=bucket_boundaries if args["batch_by_tokens"] else None,
                boundaries_reduce_to_length_fn=lambda x: max(tf.nest.flatten(x)),
                num_replicas_in_sync=num_replicas_in_sync)
            if level != GPU_EFFICIENT_LEVEL.LEVEL0:
                if isinstance(bucket_batch_sizes, list):
                    bucket_batch_sizes = [
                        int(maximum_lower_multiple(x // num_replicas_in_sync,
                                                   EFFICIENT_MULTIPLIER[level]) * num_replicas_in_sync)
                        for x in bucket_batch_sizes]
                else:
                    bucket_batch_sizes = int(maximum_lower_multiple(
                        bucket_batch_sizes // num_replicas_in_sync,
                        EFFICIENT_MULTIPLIER[level]) * num_replicas_in_sync)
            return dataset_utils.batch_examples_by_token(
                dataset,
                bucket_boundaries=bucket_boundaries,
                bucket_batch_sizes=bucket_batch_sizes,
                padding_values=padding_values,
                example_length_func=lambda x: {k: tf.size(v) for k, v in x.items()}
            )
