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
from typing import Tuple

import tensorflow as tf
from absl import logging

import neurst.data.dataset_utils as dataset_utils
from neurst.data.data_pipelines import DataPipeline, build_data_pipeline
from neurst.data.data_pipelines.text_data_pipeline import TextDataPipeline
from neurst.data.datasets import Dataset
from neurst.data.text.vocab import PaddingMode
from neurst.layers.metric_layers.token_metric_layers import BatchCountMetricLayer, SequenceTokenMetricLayer
from neurst.models import build_model
from neurst.models.model_utils import deduce_text_length
from neurst.tasks import register_task
from neurst.tasks.task import Task
from neurst.training.training_utils import (EFFICIENT_MULTIPLIER, GPU_EFFICIENT_LEVEL, maximum_lower_multiple,
                                            minimal_multiple)
from neurst.utils import compat
from neurst.utils.configurable import deep_merge_dict
from neurst.utils.flags_core import Flag, ModuleFlag


@register_task("lm")
class LanguageModel(Task):
    """ Defines the sequence to sequence task. """

    def __init__(self, args):
        """ Initializes the task.

        Args:
            args: A dict of model configurations.
        """
        data_pipeline_cls = args.get("data_pipeline.class", TextDataPipeline)
        data_pipeline_params = args.get("data_pipeline.params", None) or {}
        self._data_pipeline = build_data_pipeline(
            data_pipeline_cls, **data_pipeline_params)
        self._begin_of_sentence = args.get("begin_of_sentence", "bos")
        super(LanguageModel, self).__init__(args)

    def get_config(self):
        return {
            "data_pipeline.class": self._data_pipeline.__class__.__name__,
            "data_pipeline.params": self._data_pipeline.get_config(),
            "begin_of_sentence": self._begin_of_sentence
        }

    @staticmethod
    def class_or_method_args():
        this_args = super(LanguageModel, LanguageModel).class_or_method_args()
        this_args.extend([
            # for creating data pipelines
            ModuleFlag(DataPipeline.REGISTRY_NAME, help="The data pipeline."),
            # for preprocessing data
            Flag("max_len", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The maximum length of training data."),
            Flag("truncate", dtype=Flag.TYPE.BOOLEAN, default=None,
                 help="Whether to truncate data to max_len."),
            # for batching dataset
            Flag("batch_by_tokens", dtype=Flag.TYPE.BOOLEAN, default=None,
                 help="Whether to batch the data by word tokens."),
            Flag("begin_of_sentence", dtype=Flag.TYPE.STRING, default="bos",
                 choices=["bos", "eos"],
                 help="The begin of sentence symbol for target side. The choice 'eos' "
                      "is for compatibility with fairseq transformer."),
            Flag("gpu_efficient_level", dtype=Flag.TYPE.INTEGER, default=GPU_EFFICIENT_LEVEL.LEVEL0,
                 choices=tuple(GPU_EFFICIENT_LEVEL),
                 help="The efficient level for training using XLA, from 0~5."),
        ])
        return this_args

    def inputs_signature(self, mode) -> Tuple[dict, dict]:
        """ Returns the input dtypes and signatures. """
        dtypes = {"tokens": tf.int64}
        signatures = {"tokens": tf.TensorShape([None, None])}
        return dtypes, signatures

    def build_model(self, args, name=None):
        """ Builds and return a keras model. """
        model = build_model(args, self._data_pipeline.meta, name=name)
        return model

    def example_to_input(self, batch_of_data: dict, mode) -> dict:
        """ Transform the data examples to model acceptable inputs.

        Args:
            batch_of_data: A data tensor with shape [batch, ...]
            mode: The running mode.

        Returns: The input data for model.
        """
        if mode == compat.ModeKeys.INFER:
            raise NotImplementedError
        input_dict = {"trg": batch_of_data["tokens"],
                      "trg_length": deduce_text_length(
                          batch_of_data["tokens"], self._data_pipeline.meta["pad_id"],
                          self._data_pipeline.meta.get("padding_mode", PaddingMode.EOS_AS_PADDING))}
        bosid = (self._data_pipeline.meta["eos_id"] if self._begin_of_sentence == "eos"
                 else self._data_pipeline.meta["bos_id"])
        bos = tf.tile([tf.convert_to_tensor(bosid, dtype=tf.int64)],
                      [tf.shape(input_dict["trg"])[0]])

        input_dict["trg_input"] = tf.concat([tf.expand_dims(bos, axis=1),
                                             batch_of_data["tokens"][:, :-1]], axis=1)
        return input_dict

    def get_data_postprocess_fn(self, data_status, **kwargs) -> callable:
        if data_status == compat.DataStatus.PROJECTED:
            return self._data_pipeline.decode
        elif data_status == compat.DataStatus.PROCESSED:
            return self._data_pipeline.postprocess
        return lambda x: x

    def get_data_preprocess_fn(self, mode, data_status=compat.DataStatus.RAW, args=None) -> callable:
        """ Preprocess data sample according to this task.

        Args:
            args: A dict containing dataset arguments.
            mode: A ModeKeys indicating the running mode.
            data_status: The status of the data sample.

        Returns: A callable function to collate (process) a data sample.
        """
        if args is None:
            args = self._args
        else:
            args = deep_merge_dict(self._args, args, local_overwrite=False)
        truncate = args.get("truncate", None)
        max_len = args.get("max_len", None)

        def _process_and_truncate(data):
            text = data["tokens"]
            if data_status != compat.DataStatus.PROJECTED:
                text = self._data_pipeline.encode(text, is_processed=(
                    data_status == compat.DataStatus.PROCESSED))
            if mode == compat.ModeKeys.TRAIN and truncate and max_len:
                if compat.is_tf_tensor(text):
                    text = tf.cond(
                        tf.less_equal(tf.size(text), max_len), lambda: text,
                        lambda: tf.concat([text[:(max_len - 1)], text[-1:]], axis=0))
                elif len(text) > max_len:
                    text = text[:(max_len - 1)] + text[-1:]
            return {"tokens": text}

        return _process_and_truncate

    def create_and_batch_tfds(self, ds: Dataset, mode,
                              args=None, num_replicas_in_sync=1) -> tf.data.Dataset:
        """ Creates a dataset according to the `mode`.

        Args:
            args: A dict containing dataset arguments.
            ds: A neurst.data.datasets.Dataset object.
            mode: A ModeKeys indicating the running mode.
            num_replicas_in_sync: The number of GPUs or other workers. We will generate global
                batches, and each global batch is equally divisible by number of replicas.

        Returns:
            A tf.data.Dataset.
        """
        if args is None:
            args = self._args
        else:
            args = deep_merge_dict(self._args, args, local_overwrite=False)
        pad = tf.constant(self._data_pipeline.meta["pad_id"], dtype=tf.int64)
        dataset = ds.build(map_func=self.get_data_preprocess_fn(mode, ds.status, args),
                           map_output_dtypes=self.inputs_signature(mode)[0],
                           auto_shard=(mode == compat.ModeKeys.TRAIN),
                           shuffle=(mode == compat.ModeKeys.TRAIN))

        if mode == compat.ModeKeys.INFER:
            raise NotImplementedError
            # logging.info("Creating test dataset.")
            # return dataset.cache().padded_batch(
            #     dataset_utils.adjust_batch_size(args["batch_size"],
            #                                     num_replicas_in_sync=num_replicas_in_sync),
            #     padded_shapes={"tokens": [None]},
            #     padding_values={"tokens": pad},
            #     drop_remainder=False)
        elif mode == compat.ModeKeys.EVAL:
            logging.info("Creating evaluation dataset.")
            return dataset.cache().padded_batch(
                dataset_utils.adjust_batch_size(args["batch_size"],
                                                num_replicas_in_sync=num_replicas_in_sync),
                padded_shapes={"tokens": [None]},
                padding_values={"tokens": pad},
                drop_remainder=False)
        else:
            logging.info("Creating training dataset.")
            level = args.get("gpu_efficient_level", None)
            logging.info(f"Creating training dataset with GPU efficient level={level}.")
            dataset = ds.build(map_func=self.get_data_preprocess_fn(mode, ds.status, args),
                               map_output_dtypes=self.inputs_signature(mode)[0],
                               auto_shard=True, shuffle=True)
            dataset = dataset_utils.clean_dataset_by_length(dataset, {"tokens": args["max_len"]})
            if args["cache_dataset"]:
                dataset = dataset.cache()
            if args["shuffle_buffer"]:
                dataset = dataset.shuffle(buffer_size=args["shuffle_buffer"])
            padding_values = {"tokens": tf.constant(self._data_pipeline.meta["pad_id"], dtype=tf.int64)}
            if args["max_len"] is None:
                raise RuntimeError("Must provide `max_len` for training.")
            max_len = minimal_multiple(args["max_len"], EFFICIENT_MULTIPLIER[level])
            batch_size = dataset_utils.adjust_batch_size(args["batch_size"], args["batch_size_per_gpu"],
                                                         num_replicas_in_sync=num_replicas_in_sync,
                                                         verbose=False)
            if level == GPU_EFFICIENT_LEVEL.LEVEL5:  # static batch
                _batch_size = batch_size
                if args["batch_by_tokens"]:
                    _batch_size = _batch_size // max_len
                logging.info(f"Batching dataset with fixed shape: batch={_batch_size} x {max_len}).")
                return dataset.padded_batch(
                    _batch_size // num_replicas_in_sync * num_replicas_in_sync,
                    padded_shapes={"tokens": [max_len]}, padding_values=padding_values,
                    drop_remainder=True)
            else:
                bucket_boundaries = [EFFICIENT_MULTIPLIER[level] * i for i in
                                     range(1, max_len // EFFICIENT_MULTIPLIER[level] + 1)]
                if bucket_boundaries[-1] < max_len:
                    bucket_boundaries.append(minimal_multiple(bucket_boundaries[-1] + 1,
                                                              EFFICIENT_MULTIPLIER[level]))
                bucket_boundaries = {"tokens": bucket_boundaries}
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

    def build_metric_layer(self):
        return [SequenceTokenMetricLayer("trg"), BatchCountMetricLayer("trg")]

    def get_eval_metric(self, args, name="metric", ds=None):
        raise NotImplementedError
