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
from neurst.data.datasets.parallel_text_dataset import AbstractParallelDataset
from neurst.data.text.vocab import PaddingMode
from neurst.layers.metric_layers.token_metric_layers import BatchCountMetricLayer, SequenceTokenMetricLayer
from neurst.metrics import build_metric
from neurst.models import build_model
from neurst.models.model_utils import deduce_text_length
from neurst.tasks import register_task
from neurst.tasks.task import Task
from neurst.utils import compat
from neurst.utils.configurable import deep_merge_dict
from neurst.utils.flags_core import Flag, ModuleFlag


@register_task("seq_to_seq")
class Seq2Seq(Task):
    """ Defines the sequence to sequence task. """

    def __init__(self, args):
        """ Initializes the task.

        Args:
            args: A dict of model configurations.
        """
        src_data_pipeline_cls = args.get("src_data_pipeline.class", TextDataPipeline)
        src_data_pipeline_params = args.get("src_data_pipeline.params", None) or {}
        self._src_data_pipeline = build_data_pipeline(
            src_data_pipeline_cls, **src_data_pipeline_params)
        trg_data_pipeline_cls = args.get("trg_data_pipeline.class", TextDataPipeline)
        trg_data_pipeline_params = args.get("trg_data_pipeline.params", None) or {}
        self._trg_data_pipeline = build_data_pipeline(
            trg_data_pipeline_cls, **trg_data_pipeline_params)
        self._target_begin_of_sentence = args.get("target_begin_of_sentence", "bos")
        super(Seq2Seq, self).__init__(args)

    def get_config(self):
        return {
            "src_data_pipeline.class": self._src_data_pipeline.__class__.__name__,
            "src_data_pipeline.params": self._src_data_pipeline.get_config(),
            "trg_data_pipeline.class": self._trg_data_pipeline.__class__.__name__,
            "trg_data_pipeline.params": self._trg_data_pipeline.get_config(),
            "target_begin_of_sentence": self._target_begin_of_sentence
        }

    @staticmethod
    def class_or_method_args():
        this_args = super(Seq2Seq, Seq2Seq).class_or_method_args()
        this_args.extend([
            # for creating data pipelines
            ModuleFlag("src_data_pipeline", DataPipeline.REGISTRY_NAME,
                       help="The source side data pipeline."),
            ModuleFlag("trg_data_pipeline", DataPipeline.REGISTRY_NAME,
                       help="The target side data pipeline."),
            # for preprocessing data
            Flag("max_src_len", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The maximum source length of training data."),
            Flag("max_trg_len", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The maximum target length of training data."),
            Flag("truncate_src", dtype=Flag.TYPE.BOOLEAN, default=None,
                 help="Whether to truncate source to max_src_len."),
            Flag("truncate_trg", dtype=Flag.TYPE.BOOLEAN, default=None,
                 help="Whether to truncate target to max_trg_len."),
            # for batching dataset
            Flag("batch_by_tokens", dtype=Flag.TYPE.BOOLEAN, default=None,
                 help="Whether to batch the data by word tokens."),
            Flag("target_begin_of_sentence", dtype=Flag.TYPE.STRING, default="bos",
                 choices=["bos", "eos"],
                 help="The begin of sentence symbol for target side. The choice 'eos' "
                      "is for compatibility with fairseq transformer.")
        ])
        return this_args

    def inputs_signature(self, mode) -> Tuple[dict, dict]:
        """ Returns the input dtypes and signatures. """
        dtypes = {"feature": tf.int64}
        signatures = {"feature": tf.TensorShape([None, None])}
        if mode == compat.ModeKeys.INFER:
            return dtypes, signatures
        dtypes["label"] = tf.int64
        signatures["label"] = tf.TensorShape([None, None])
        return dtypes, signatures

    def build_model(self, args, name=None, **kwargs):
        """ Builds and return a keras model. """
        model = build_model(args, self._src_data_pipeline.meta,
                            self._trg_data_pipeline.meta, name=name, **kwargs)
        return model

    def example_to_input(self, batch_of_data: dict, mode) -> dict:
        """ Transform the data examples to model acceptable inputs.

        Args:
            batch_of_data: A data tensor with shape [batch, ...]
            mode: The running mode.

        Returns: The input data for model.
        """
        input_dict = {"src": batch_of_data["feature"],
                      "src_length": deduce_text_length(
                          batch_of_data["feature"], self._src_data_pipeline.meta["pad_id"],
                          self._src_data_pipeline.meta.get("padding_mode", PaddingMode.EOS_AS_PADDING))}
        bosid = (self._trg_data_pipeline.meta["eos_id"] if self._target_begin_of_sentence == "eos"
                 else self._trg_data_pipeline.meta["bos_id"])
        target_bos = tf.tile([tf.convert_to_tensor(bosid, dtype=tf.int64)],
                             [tf.shape(input_dict["src"])[0]])
        if mode == compat.ModeKeys.INFER:
            input_dict["trg_input"] = target_bos
        else:
            input_dict["trg"] = batch_of_data["label"]
            input_dict["trg_length"] = deduce_text_length(
                batch_of_data["label"], self._trg_data_pipeline.meta["pad_id"],
                self._trg_data_pipeline.meta.get("padding_mode", PaddingMode.EOS_AS_PADDING))
            input_dict["trg_input"] = tf.concat([tf.expand_dims(target_bos, axis=1),
                                                 batch_of_data["label"][:, :-1]], axis=1)
        return input_dict

    def get_data_postprocess_fn(self, data_status, is_src=False, **kwargs) -> callable:
        dp = self._src_data_pipeline if is_src else self._trg_data_pipeline
        if data_status == compat.DataStatus.PROJECTED:
            return dp.decode
        elif data_status == compat.DataStatus.PROCESSED:
            return dp.postprocess
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
        truncate_src = args.get("truncate_src", None)
        truncate_trg = args.get("truncate_trg", None)
        max_src_len = args.get("max_src_len", None)
        max_trg_len = args.get("max_trg_len", None)

        def _process_and_truncate(text, dp, trunc, max_len):
            if data_status != compat.DataStatus.PROJECTED:
                text = dp.encode(text, is_processed=(data_status == compat.DataStatus.PROCESSED))
            if mode == compat.ModeKeys.TRAIN and trunc and max_len:
                if compat.is_tf_tensor(text):
                    text = tf.cond(
                        tf.less_equal(tf.size(text), max_len), lambda: text,
                        lambda: tf.concat([text[:(max_len - 1)], text[-1:]], axis=0))
                elif len(text) > max_len:
                    text = text[:(max_len - 1)] + text[-1:]
            return text

        if mode == compat.ModeKeys.INFER:
            return lambda data: {
                "feature": _process_and_truncate(data["feature"],
                                                 self._src_data_pipeline,
                                                 truncate_src,
                                                 max_src_len)}
        return lambda data: {
            "feature": _process_and_truncate(data["feature"],
                                             self._src_data_pipeline,
                                             truncate_src,
                                             max_src_len),
            "label": _process_and_truncate(data["label"],
                                           self._trg_data_pipeline,
                                           truncate_trg,
                                           max_trg_len)}

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
        src_pad = tf.constant(self._src_data_pipeline.meta["pad_id"], dtype=tf.int64)
        trg_pad = tf.constant(self._trg_data_pipeline.meta["pad_id"], dtype=tf.int64)

        assert isinstance(ds, AbstractParallelDataset), (
            "The dataset for SeqToSeq task must inherit AbstractParallelDataset.")

        dataset = ds.build(map_func=self.get_data_preprocess_fn(mode, ds.status, args),
                           map_output_dtypes=self.inputs_signature(mode)[0],
                           auto_shard=(mode == compat.ModeKeys.TRAIN),
                           shuffle=(mode == compat.ModeKeys.TRAIN))

        if mode == compat.ModeKeys.INFER:
            logging.info("Creating test dataset.")
            return dataset.padded_batch(
                dataset_utils.adjust_batch_size(args["batch_size"],
                                                num_replicas_in_sync=num_replicas_in_sync),
                padded_shapes={"feature": [None]},
                padding_values={"feature": src_pad},
                drop_remainder=False)
        elif mode == compat.ModeKeys.EVAL:
            logging.info("Creating evaluation dataset.")
            return dataset.padded_batch(
                dataset_utils.adjust_batch_size(args["batch_size"],
                                                num_replicas_in_sync=num_replicas_in_sync),
                padded_shapes={"feature": [None], "label": [None]},
                padding_values={"feature": src_pad, "label": trg_pad},
                drop_remainder=False)
        else:
            logging.info("Creating training dataset.")
            dataset = dataset_utils.clean_dataset_by_length(
                dataset, {"feature": args["max_src_len"], "label": args["max_trg_len"]})
            if args["cache_dataset"]:
                dataset = dataset.cache()
            if args["shuffle_buffer"]:
                dataset = dataset.shuffle(buffer_size=args["shuffle_buffer"])
            padding_values = {"feature": src_pad, "label": trg_pad}
            if args["max_src_len"] is None:
                raise RuntimeError("Must provide `max_src_len` for training.")
            if args["max_trg_len"] is None:
                raise RuntimeError("Must provide `max_trg_len` for training.")
            src_bucket_boundaries, trg_bucket_boundaries = dataset_utils.associated_bucket_boundaries(
                dataset_utils.create_batch_bucket_boundaries(args["max_src_len"]),
                dataset_utils.create_batch_bucket_boundaries(args["max_trg_len"]))

            bucket_boundaries = {
                "feature": src_bucket_boundaries,
                "label": trg_bucket_boundaries
            }
            bucket_batch_sizes = dataset_utils.adjust_batch_size(
                args["batch_size"],
                args["batch_size_per_gpu"],
                bucket_boundaries=bucket_boundaries if args["batch_by_tokens"] else None,
                boundaries_reduce_to_length_fn=lambda x: max(tf.nest.flatten(x)),
                num_replicas_in_sync=num_replicas_in_sync)
            return dataset_utils.batch_examples_by_token(
                dataset,
                bucket_boundaries=bucket_boundaries,
                bucket_batch_sizes=bucket_batch_sizes,
                padding_values=padding_values,
                example_length_func=lambda x: {k: tf.size(v) for k, v in x.items()}
            )

    def build_metric_layer(self):
        return [SequenceTokenMetricLayer("src"), SequenceTokenMetricLayer("trg"),
                BatchCountMetricLayer("src")]

    def get_eval_metric(self, args, name="metric", ds=None):
        """ Returns a neurst.metrics.metric.Metric object for evaluation."""
        if ds is not None and hasattr(ds, "trg_lang") and ds.trg_lang is not None:
            return build_metric(args[name + ".class"], language=ds.trg_lang,
                                **args[name + ".params"])
        return build_metric(args[name + ".class"], language=self._trg_data_pipeline.meta["language"],
                            **args[name + ".params"])
