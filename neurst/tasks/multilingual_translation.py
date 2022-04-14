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
from neurst.data.data_pipelines.multilingual_text_data_pipeline import MultilingualTextDataPipeline
from neurst.layers.metric_layers.token_metric_layers import BatchCountMetricLayer, SequenceTokenMetricLayer
from neurst.metrics import build_metric
from neurst.models import build_model
from neurst.models.model_utils import deduce_text_length
from neurst.tasks import register_task
from neurst.tasks.task import Task
from neurst.training.training_utils import maximum_lower_multiple, minimal_multiple
from neurst.utils import compat
from neurst.utils.configurable import deep_merge_dict
from neurst.utils.flags_core import Flag

_TRG_LANG_TAG_POSITIONS = ["source", "target", "src", "trg"]


@register_task
class MultilingualTranslation(Task):
    """ Defines the translation task. """

    def __init__(self, args):
        """ Initializes the task.

        Args:
            args: A dict of model configurations.
        """
        super(MultilingualTranslation, self).__init__(args)
        self._multilingual_dp = MultilingualTextDataPipeline(
            vocab_path=args["vocab_path"], spm_model=args["spm_model"],
            languages=args["languages"])
        self._with_src_lang_tag = args["with_src_lang_tag"]
        self._trg_lang_tag_position = args["trg_lang_tag_position"]
        assert self._trg_lang_tag_position in _TRG_LANG_TAG_POSITIONS

    @staticmethod
    def class_or_method_args():
        this_args = super(MultilingualTranslation, MultilingualTranslation).class_or_method_args()
        this_args.extend([
            # for creating multilingual pipeline
            Flag("vocab_path", dtype=Flag.TYPE.STRING,
                 help="The path to the vocabulary file, or a list of word tokens."),
            Flag("spm_model", dtype=Flag.TYPE.STRING,
                 help="The path to the sentence piece model."),
            Flag("languages", dtype=Flag.TYPE.STRING,
                 help="A list of languages. The corresponding language tags "
                      "will automatically append to the vocabulary. "),
            # for preprocessing data
            Flag("max_src_len", dtype=Flag.TYPE.INTEGER, default=80,
                 help="The maximum source length of training data."),
            Flag("max_trg_len", dtype=Flag.TYPE.INTEGER, default=80,
                 help="The maximum target length of training data."),
            Flag("truncate_src", dtype=Flag.TYPE.BOOLEAN, default=None,
                 help="Whether to truncate source to max_src_len."),
            Flag("truncate_trg", dtype=Flag.TYPE.BOOLEAN, default=None,
                 help="Whether to truncate target to max_trg_len."),
            # for batching dataset
            Flag("batch_by_tokens", dtype=Flag.TYPE.BOOLEAN, default=None,
                 help="Whether to batch the data by word tokens."),
            Flag("with_src_lang_tag", dtype=Flag.TYPE.STRING, default=False,
                 help="Whether to append the source language tag at the beginning of the source sentence."),
            Flag("trg_lang_tag_position", dtype=Flag.TYPE.STRING, default="trg",
                 choices=_TRG_LANG_TAG_POSITIONS,
                 help="The position where the target language tag will be appended"),
        ])
        return this_args

    def get_config(self):
        return {
            "vocab_path": self._args["vocab_path"],
            "spm_model": self._args["spm_model"],
            "languages": self._args["languages"],
            "with_src_lang_tag": self._with_src_lang_tag,
            "trg_lang_tag_position": self._trg_lang_tag_position,
        }

    def inputs_signature(self, mode):
        """ Returns the input dtypes and signatures. """
        dtypes = {"feature": tf.int64, "src_lang": tf.int64, "trg_lang": tf.int64}
        signatures = {"feature": tf.TensorShape([None, None]),
                      "src_lang": tf.TensorShape([None, ]),
                      "trg_lang": tf.TensorShape([None, ])}
        if mode == compat.ModeKeys.INFER:
            return dtypes, signatures
        dtypes["label"] = tf.int64
        signatures["label"] = tf.TensorShape([None, None])
        return dtypes, signatures

    def build_model(self, args, name=None):
        """ Builds and return a keras model. """
        model = build_model(args, self._multilingual_dp.meta,
                            self._multilingual_dp.meta, name=name)
        return model

    def example_to_input(self, batch_of_data: dict, mode) -> dict:
        """ Transform the data examples to model acceptable inputs.

        Args:
            batch_of_data: A data tensor with shape [batch, ...]
            mode: The running mode.

        Returns: The input data for model.
        """
        src = batch_of_data["feature"]
        if self._trg_lang_tag_position in ["src", "source"]:
            src = tf.concat([tf.expand_dims(batch_of_data["trg_lang"], axis=1), src], axis=1)
        if self._with_src_lang_tag:
            src = tf.concat([tf.expand_dims(batch_of_data["src_lang"], axis=1), src], axis=1)

        input_dict = {"src": src,
                      "src_length": deduce_text_length(src, self._multilingual_dp.meta["pad_id"],
                                                       self._multilingual_dp.meta["padding_mode"])}
        if self._trg_lang_tag_position in ["trg", "target"]:
            target_bos = batch_of_data["trg_lang"]
        else:
            target_bos = tf.tile([tf.convert_to_tensor(
                self._multilingual_dp.meta["bos_id"], dtype=tf.int64)], [tf.shape(src)[0]])
        if mode == compat.ModeKeys.INFER:
            input_dict["trg_input"] = target_bos
        else:
            input_dict["trg"] = batch_of_data["label"]
            input_dict["trg_length"] = deduce_text_length(batch_of_data["label"],
                                                          self._multilingual_dp.meta["pad_id"],
                                                          self._multilingual_dp.meta["padding_mode"])
            input_dict["trg_input"] = tf.concat([tf.expand_dims(target_bos, axis=1),
                                                 batch_of_data["label"][:, :-1]], axis=1)
        return input_dict

    def get_data_postprocess_fn(self, data_status, **kwargs) -> callable:
        if data_status == compat.DataStatus.PROJECTED:
            return self._multilingual_dp.decode
        elif data_status == compat.DataStatus.PROCESSED:
            return self._multilingual_dp.postprocess
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

        def _process_and_truncate(text, trunc, max_len):
            if data_status != compat.DataStatus.PROJECTED:
                text = self._multilingual_dp.encode(
                    text, is_processed=(data_status == compat.DataStatus.PROCESSED))
            if mode == compat.ModeKeys.TRAIN and trunc and max_len:
                if compat.is_tf_tensor(text):
                    text = tf.cond(
                        tf.less_equal(tf.size(text), max_len), lambda: text,
                        lambda: tf.concat([text[:(max_len - 1)], text[-1:]], axis=0))
                elif len(text) > max_len:
                    text = text[:(max_len - 1)] + text[-1:]
            return text

        def _process_lang(lang):
            if not compat.is_tf_tensor(lang) and isinstance(lang, str):
                return self._multilingual_dp.meta["lang2id"][lang]
            assert isinstance(lang, int)
            return lang

        if mode == compat.ModeKeys.INFER:
            return lambda data: {
                "feature": _process_and_truncate(data["feature"], truncate_src, max_src_len),
                "src_lang": _process_lang(data["src_lang"]),
                "trg_lang": _process_lang(data["trg_lang"]), }
        return lambda data: {
            "feature": _process_and_truncate(data["feature"], truncate_src, max_src_len),
            "label": _process_and_truncate(data["label"], truncate_trg, max_trg_len),
            "src_lang": _process_lang(data["src_lang"]),
            "trg_lang": _process_lang(data["trg_lang"]),
        }

    def create_and_batch_tfds(self, ds, mode,
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
        eos = tf.constant(self._multilingual_dp.meta["eos_id"], dtype=tf.int64)
        int_zero = tf.zeros([], dtype=tf.int64)

        dataset = ds.build(map_func=self.get_data_preprocess_fn(mode, ds.status, args),
                           map_output_dtypes=self.inputs_signature(mode)[0],
                           auto_shard=(mode == compat.ModeKeys.TRAIN),
                           shuffle=(mode == compat.ModeKeys.TRAIN))

        if mode == compat.ModeKeys.INFER:
            logging.info("Creating test dataset.")
            return dataset.cache().padded_batch(
                dataset_utils.adjust_batch_size(args["batch_size"],
                                                num_replicas_in_sync=num_replicas_in_sync),
                padded_shapes={"feature": [None], "src_lang": [], "trg_lang": []},
                padding_values={"feature": eos, "src_lang": int_zero, "trg_lang": int_zero},
                drop_remainder=False)
        elif mode == compat.ModeKeys.EVAL:
            logging.info("Creating evaluation dataset.")
            return dataset.cache().padded_batch(
                dataset_utils.adjust_batch_size(args["batch_size"],
                                                num_replicas_in_sync=num_replicas_in_sync),
                padded_shapes={"feature": [None], "label": [None], "src_lang": [], "trg_lang": []},
                padding_values={"feature": eos, "label": eos,
                                "src_lang": int_zero, "trg_lang": int_zero},
                drop_remainder=False)
        else:
            logging.info("Creating training dataset.")
            dataset = dataset_utils.clean_dataset_by_length(
                dataset, {"feature": args["max_src_len"], "label": args["max_trg_len"]})
            if args["cache_dataset"]:
                dataset = dataset.cache()
            if args["shuffle_buffer"]:
                dataset = dataset.shuffle(buffer_size=args["shuffle_buffer"])
            padding_values = {"feature": eos, "label": eos,
                              "src_lang": int_zero, "trg_lang": int_zero}
            if args["max_src_len"] is None:
                raise RuntimeError("Must provide `max_src_len` for training.")
            if args["max_trg_len"] is None:
                raise RuntimeError("Must provide `max_trg_len` for training.")

            num_extra_srctokens = 0
            if self._with_src_lang_tag:
                num_extra_srctokens += 1
            if self._trg_lang_tag_position in ["src", "source"]:
                num_extra_srctokens += 1

            max_src_len = minimal_multiple(args["max_src_len"] + num_extra_srctokens, 8)
            max_trg_len = minimal_multiple(args["max_trg_len"], 8)
            batch_size = dataset_utils.adjust_batch_size(args["batch_size"], args["batch_size_per_gpu"],
                                                         num_replicas_in_sync=num_replicas_in_sync,
                                                         verbose=False)
            src_bucket_boundaries = [8 * i for i in range(1, max_src_len // 8 + 1)]
            if src_bucket_boundaries[-1] < max_src_len:
                src_bucket_boundaries.append(minimal_multiple(src_bucket_boundaries[-1] + 1, 8))
            trg_bucket_boundaries = [8 * i for i in range(1, max_trg_len // 8 + 1)]
            if trg_bucket_boundaries[-1] < max_trg_len:
                trg_bucket_boundaries.append(minimal_multiple(trg_bucket_boundaries[-1] + 1, 8))
            src_bucket_boundaries, trg_bucket_boundaries = dataset_utils.associated_bucket_boundaries(
                src_bucket_boundaries, trg_bucket_boundaries)
            src_bucket_boundaries = [x - num_extra_srctokens for x in src_bucket_boundaries]
            bucket_boundaries = {
                "feature": src_bucket_boundaries,
                "label": trg_bucket_boundaries
            }
            bucket_batch_sizes = dataset_utils.adjust_batch_size(
                batch_size,
                bucket_boundaries=bucket_boundaries if args["batch_by_tokens"] else None,
                boundaries_reduce_to_length_fn=lambda x: max(tf.nest.flatten(x)),
                num_replicas_in_sync=num_replicas_in_sync)
            if isinstance(bucket_batch_sizes, list):
                bucket_batch_sizes = [
                    int(maximum_lower_multiple(x // num_replicas_in_sync, 8) * num_replicas_in_sync)
                    for x in bucket_batch_sizes]
            else:
                bucket_batch_sizes = int(maximum_lower_multiple(
                    bucket_batch_sizes // num_replicas_in_sync, 8) * num_replicas_in_sync)
            return dataset_utils.batch_examples_by_token(
                dataset,
                bucket_boundaries=bucket_boundaries,
                bucket_batch_sizes=bucket_batch_sizes,
                padding_values=padding_values,
                example_length_func=lambda x: {"feature": tf.size(x["feature"]),
                                               "label": tf.size(x["label"])},
                extra_padded_shapes={"src_lang": [], "trg_lang": []}
            )

    def build_metric_layer(self):
        return [SequenceTokenMetricLayer("src"), SequenceTokenMetricLayer("trg"),
                BatchCountMetricLayer("src")]

    def get_eval_metric(self, args, name="metric", ds=None):
        """ Returns a neurst.metrics.metric.Metric object for evaluation."""
        if ds is None or not hasattr(ds, "trg_lang") or ds.trg_lang is None:
            logging.info("WARNING: The dataset must have `trg_lang` property, "
                         "otherwise no metric will be created.")
            return None
        return build_metric(args[name + ".class"], language=ds.trg_lang,
                            **args[name + ".params"])
