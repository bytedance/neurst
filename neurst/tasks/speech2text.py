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
from typing import Tuple

import tensorflow as tf
from absl import logging

from neurst.data import dataset_utils
from neurst.data.data_pipelines import DataPipeline, build_data_pipeline
from neurst.data.data_pipelines.text_data_pipeline import TextDataPipeline
from neurst.data.datasets import Dataset
from neurst.layers.metric_layers.token_metric_layers import (AudioFramesMetricLayer, BatchCountMetricLayer,
                                                             SequenceTokenMetricLayer)
from neurst.metrics import build_metric
from neurst.models import build_model
from neurst.models.model_utils import deduce_text_length
from neurst.tasks import register_task
from neurst.tasks.task import Task
from neurst.training.training_utils import minimal_multiple
from neurst.utils import compat
from neurst.utils.audio_lib import SpecAugment
from neurst.utils.configurable import deep_merge_dict
from neurst.utils.flags_core import Flag, ModuleFlag


def create_audio_bucket_boundaries(maxlen, minlen=128):
    if minlen is None:
        minlen = 128
    bounds = [minlen]
    base = minlen
    base_incr = int(2 ** ((math.log2(minlen) + 1) // 2))
    base_incr_mult = 1
    times = len(str(int(minlen)))
    while True:
        for _ in range(times):
            bounds.append(bounds[-1] + base)
            if bounds[-1] > maxlen:
                break
        base += base_incr * base_incr_mult
        base_incr_mult += 1
        if bounds[-1] > maxlen:
            break
    bounds[-1] = maxlen + 1
    return bounds


@register_task(["speech2text", "audio2text", "AudioToText"])
class SpeechToText(Task):
    """ Defines the audio to text task. """

    def __init__(self, args):
        """ Initializes the task.

        Args:
            args: A dict of model configurations.
        """
        super(SpeechToText, self).__init__(args)
        trg_data_pipeline_cls = args.get("transcript_data_pipeline.class", TextDataPipeline)
        trg_data_pipeline_params = args.get("transcript_data_pipeline.params", None) or {}
        self._trg_data_pipeline = build_data_pipeline(
            trg_data_pipeline_cls, **trg_data_pipeline_params)
        self._audio_feature_dim = args["audio_feature_dim"]
        self._audio_feature_channels = args["audio_feature_channels"]
        self._specaug = SpecAugment.build(args.get("specaug", None))

    def get_config(self):
        return {
            "transcript_data_pipeline.class": self._trg_data_pipeline.__class__.__name__,
            "transcript_data_pipeline.params": self._trg_data_pipeline.get_config(),
            "audio_feature_dim": self._audio_feature_dim,
            "audio_feature_channels": self._audio_feature_channels
        }

    @staticmethod
    def class_or_method_args():
        this_args = super(SpeechToText, SpeechToText).class_or_method_args()
        this_args.extend([
            ModuleFlag("transcript_data_pipeline", DataPipeline.REGISTRY_NAME,
                       default=None, help="The target side transcript data pipeline."),
            Flag("audio_feature_dim", dtype=Flag.TYPE.INTEGER, default=80,
                 help="The dimension of audio features."),
            Flag("audio_feature_channels", dtype=Flag.TYPE.INTEGER, default=1,
                 help="The number of channels of audio features."),
            Flag("max_src_len", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The maximum source length of training data (audio frames)."),
            Flag("min_src_bucket_boundary", dtype=Flag.TYPE.INTEGER, default=128,
                 help="The minimum source length of the training bucket (audio frames)."),
            Flag("max_trg_len", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The maximum target length of training data."),
            Flag("truncate_src", dtype=Flag.TYPE.BOOLEAN, default=None,
                 help="Whether to truncate source to max_src_len."),
            Flag("truncate_trg", dtype=Flag.TYPE.BOOLEAN, default=None,
                 help="Whether to truncate target to max_trg_len."),
            Flag("experimental_frame_transcript_ratio", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The ratio of the number of frames and its transcript for training batch bucket."),
            Flag("specaug", dtype=Flag.TYPE.STRING, default=None,
                 help="The arguments for spec augment, can be either predefined settings "
                      "like LB, LD, SM, SS... or a dict containing detailed arguments."),
            Flag("disable_batch_efficiency", dtype=Flag.TYPE.BOOLEAN, default=None,
                 help="Whether to disable the batch efficiency.")
        ])
        return this_args

    def inputs_signature(self, mode) -> Tuple[dict, dict]:
        """ Returns the input dtypes and signatures (from dataset). """
        dtypes = {"audio": tf.float32, "audio_length": tf.int64}
        # [batch, frames, feature_dim]
        signatures = {"audio": tf.TensorShape([None, None]),
                      "audio_length": tf.TensorShape([None, ])}
        if mode == compat.ModeKeys.INFER:
            return dtypes, signatures
        dtypes["transcript"] = tf.int64
        signatures["transcript"] = tf.TensorShape([None, None])
        return dtypes, signatures

    def build_model(self, args, name=None):
        """ Creates the model. """
        model = build_model(args, {"audio_feature_dim": self._audio_feature_dim,
                                   "audio_feature_channels": self._audio_feature_channels},
                            self._trg_data_pipeline.meta, name=name)
        return model

    def example_to_input(self, batch_of_data: dict, mode) -> dict:
        """ Transform the data examples to model acceptable inputs.

        Args:
            batch_of_data: A data tensor with shape [batch, ...]
            mode: The running mode.

        Returns: The input data for model.
        """
        batch = tf.shape(batch_of_data["audio"])[0]
        input_dict = {"src": tf.reshape(batch_of_data["audio"],
                                        [batch, -1, self._audio_feature_dim, self._audio_feature_channels]),
                      "src_length": batch_of_data["audio_length"]}

        target_bos = tf.tile(
            [tf.convert_to_tensor(self._trg_data_pipeline.meta["bos_id"], dtype=tf.int64)],
            [tf.shape(input_dict["src"])[0]])
        if mode == compat.ModeKeys.INFER:
            input_dict["trg_input"] = target_bos
        else:
            input_dict["trg"] = batch_of_data["transcript"]
            input_dict["trg_length"] = deduce_text_length(batch_of_data["transcript"],
                                                          self._trg_data_pipeline.meta["pad_id"],
                                                          self._trg_data_pipeline.meta["padding_mode"])
            input_dict["trg_input"] = tf.concat([tf.expand_dims(target_bos, axis=1),
                                                 batch_of_data["transcript"][:, :-1]], axis=1)
        return input_dict

    def get_data_postprocess_fn(self, data_status, is_src=False, **kwargs) -> callable:
        if isinstance(data_status, dict):
            data_status = data_status["transcript"]
        if data_status == compat.DataStatus.PROJECTED:
            return self._trg_data_pipeline.decode
        elif data_status == compat.DataStatus.PROCESSED:
            return self._trg_data_pipeline.postprocess
        return lambda x: x

    def get_data_preprocess_fn(self, mode, data_status, args=None) -> callable:
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
        trunc_audio = args.get("truncate_src", None)
        max_audio_len = args.get("max_src_len", None)
        trunc_trg = args.get("truncate_trg", None)
        max_trg_len = args.get("max_trg_len", None)

        if data_status["audio"] != compat.DataStatus.PROJECTED:
            raise RuntimeError("We recommend one to preprocess the audio in advance.")

        def _process_audio(audio):
            if trunc_audio and max_audio_len:
                audio = audio[:max_audio_len * self._audio_feature_dim * self._audio_feature_channels]
            if self._specaug is not None:
                audio = tf.reshape(
                    audio, [-1, self._audio_feature_dim * self._audio_feature_channels])
                audio = tf.reshape(self._specaug(audio), [-1])
            return audio

        def _process_and_truncate_text(text):
            if data_status["transcript"] == compat.DataStatus.RAW:
                if compat.is_tf_tensor(text):
                    text = text.numpy()
                text = self._trg_data_pipeline.encode(text, is_processed=False)
            else:
                assert data_status["transcript"] == compat.DataStatus.PROJECTED
            if mode == compat.ModeKeys.TRAIN and trunc_trg and max_trg_len:
                if compat.is_tf_tensor(text):
                    text = tf.cond(
                        tf.less_equal(tf.size(text), max_trg_len), lambda: text,
                        lambda: tf.concat([text[:(max_trg_len - 1)], text[-1:]], axis=0))
                else:
                    if len(text) > max_trg_len:
                        text = text[:(max_trg_len - 1)] + text[-1:]
            return text

        def data_proc(data, with_label):
            feature = _process_audio(data["audio"])
            ret = {"audio": feature,
                   "audio_length": tf.cast(
                       (tf.shape(feature)[0] if compat.is_tf_tensor(feature)
                        else feature.shape[0]) // self._audio_feature_dim // self._audio_feature_channels,
                       dtype=tf.int64)}
            if with_label:
                ret["transcript"] = tf.convert_to_tensor(
                    _process_and_truncate_text(data["transcript"]), tf.int64)
            return ret

        if mode == compat.ModeKeys.INFER:
            return lambda data: data_proc(data, False)
        return lambda data: data_proc(data, True)

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
        float_zero = tf.constant(0, dtype=tf.float32)
        int_zero = tf.constant(0, dtype=tf.int64)
        trg_pad = tf.constant(self._trg_data_pipeline.meta["pad_id"], dtype=tf.int64)

        dataset = ds.build(map_func=self.get_data_preprocess_fn(mode, ds.status, args),
                           map_output_dtypes=self.inputs_signature(mode)[0],
                           auto_shard=(mode == compat.ModeKeys.TRAIN),
                           shuffle=(mode == compat.ModeKeys.TRAIN))

        if mode == compat.ModeKeys.INFER:
            logging.info("Creating test dataset.")
            return dataset.cache().padded_batch(
                dataset_utils.adjust_batch_size(args["batch_size"],
                                                num_replicas_in_sync=num_replicas_in_sync),
                padded_shapes={"audio": [None], "audio_length": []},
                padding_values={"audio": float_zero, "audio_length": int_zero},
                drop_remainder=False)

        elif mode == compat.ModeKeys.EVAL:
            logging.info("Creating evaluation dataset.")
            return dataset.cache().padded_batch(
                dataset_utils.adjust_batch_size(args["batch_size"],
                                                num_replicas_in_sync=num_replicas_in_sync),
                padded_shapes={"audio": [None], "audio_length": [], "transcript": [None]},
                padding_values={"audio": float_zero, "audio_length": int_zero, "transcript": trg_pad},
                drop_remainder=False)
        else:
            logging.info("Creating training dataset.")
            dataset = dataset_utils.clean_dataset_by_length(
                dataset, {"audio": args["max_src_len"] * self._audio_feature_dim * self._audio_feature_channels,
                          "audio_length": -1, "transcript": args["max_trg_len"]})
            if args["cache_dataset"]:
                dataset = dataset.cache()
            if args["shuffle_buffer"]:
                dataset = dataset.shuffle(buffer_size=args["shuffle_buffer"])
            padding_values = {"audio": float_zero, "audio_length": int_zero, "transcript": trg_pad}
            if args["max_src_len"] is None:
                raise RuntimeError("`max_src_len` for SpeechToText task must be provided.")
            if args["max_trg_len"] is None:
                raise RuntimeError("`max_trg_len` for SpeechToText task must be provided.")
            max_src_len = args["max_src_len"]
            max_trg_len = minimal_multiple(args["max_trg_len"], 8)
            audio_bucket_boundaries = create_audio_bucket_boundaries(max_src_len, args["min_src_bucket_boundary"])
            audio_bucket_boundaries[-1] = minimal_multiple(audio_bucket_boundaries[-1], 8)
            batch_size = dataset_utils.adjust_batch_size(args["batch_size"], args["batch_size_per_gpu"],
                                                         num_replicas_in_sync=num_replicas_in_sync,
                                                         verbose=False)
            batch_size_per_gpu = batch_size // num_replicas_in_sync
            assert batch_size_per_gpu > max_src_len, (
                f"batch size per gpu({batch_size_per_gpu} must be greater than "
                f"`max_src_len`={max_src_len}")
            if args["disable_batch_efficiency"]:
                bucket_batch_sizes = [int(batch_size_per_gpu // bound
                                          * num_replicas_in_sync) for bound in audio_bucket_boundaries]
            else:
                bucket_batch_sizes = [int(minimal_multiple(batch_size_per_gpu // bound, 8)
                                          * num_replicas_in_sync) for bound in audio_bucket_boundaries]
            frame_transcript_ratio = args["experimental_frame_transcript_ratio"]
            if frame_transcript_ratio is None:
                logging.info("WARNING: we recommend one to pre-scan the dataset and estimate the ratio: "
                             "frame length / transcript length.")
            else:
                trans_bucket_boundaries = [
                    int(bound / (frame_transcript_ratio + i * (
                        max_src_len / max_trg_len - frame_transcript_ratio) / len(audio_bucket_boundaries)))
                    for i, bound in enumerate(audio_bucket_boundaries)]
                trans_bucket_boundaries = [minimal_multiple(min(i, max_trg_len), 8) for i in
                                           trans_bucket_boundaries]
                num_buckets = len(trans_bucket_boundaries)
                true_trans_bucket_boundaries = []
                num_input_shapes = 0
                for idx, (batc, bound, tbound) in enumerate(zip(bucket_batch_sizes, audio_bucket_boundaries,
                                                                trans_bucket_boundaries)):
                    max_trans_len = [tbound,
                                     trans_bucket_boundaries[min(idx + 1, len(bucket_batch_sizes) - 1)]]
                    num_input_shapes += len(set(max_trans_len))
                    true_trans_bucket_boundaries.append(max_trans_len)
                logging.info(f"There are {num_input_shapes} input shapes to be compiled:")
                for idx, (batc, bound, tbound) in enumerate(zip(bucket_batch_sizes, audio_bucket_boundaries,
                                                                true_trans_bucket_boundaries)):
                    logging.info(f"   - batch={batc}, maximum-frames={bound}, "
                                 f"maximum-transcript-length={set(tbound)}")
                true_trans_bucket_boundaries = tf.constant(true_trans_bucket_boundaries, dtype=tf.int32)
                true_audio_bucket_boundaries = tf.transpose(tf.constant([audio_bucket_boundaries] * 2, dtype=tf.int32))

            bucket_batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)
            audio_bucket_boundaries = tf.constant(audio_bucket_boundaries, dtype=tf.int32)

            def example_to_bucket_id(examples):
                """Return int64 bucket id for this example, calculated based on length."""
                if frame_transcript_ratio is None:
                    conditions_c = tf.less_equal(tf.cast(examples["audio_length"], tf.int32),
                                                 audio_bucket_boundaries)
                    return tf.reduce_min(tf.where(conditions_c))
                conditions_c = tf.logical_and(
                    tf.less_equal(tf.cast(examples["audio_length"], tf.int32), true_audio_bucket_boundaries),
                    tf.less_equal(tf.size(examples["transcript"]), true_trans_bucket_boundaries))
                minimum_match = tf.where(conditions_c)[0]
                return minimum_match[0] * num_buckets + minimum_match[1]

            def window_size_fn(bucket_id):
                """Return number of examples to be grouped when given a bucket id."""
                if frame_transcript_ratio is None:
                    return bucket_batch_sizes[bucket_id]
                return bucket_batch_sizes[bucket_id // num_buckets]

            def batching_fn(bucket_id, grouped_dataset):
                """Batch and add padding to a dataset of elements with similar lengths."""
                bucket_batch_size = window_size_fn(bucket_id)

                # Batch the dataset and add padding so that all input sequences in the
                # examples have the same length, and all target sequences have the same
                # lengths as well. Resulting lengths of inputs and targets can differ.
                return grouped_dataset.padded_batch(
                    bucket_batch_size,
                    padded_shapes={
                        "audio": ([(audio_bucket_boundaries[bucket_id] if frame_transcript_ratio is None
                                    else audio_bucket_boundaries[bucket_id // num_buckets])
                                   * self._audio_feature_dim * self._audio_feature_channels]),
                        "audio_length": [],
                        "transcript": ([None] if frame_transcript_ratio is None
                                       else [
                            true_trans_bucket_boundaries[bucket_id // num_buckets][bucket_id % num_buckets]])
                    },
                    padding_values=padding_values, drop_remainder=True)

            return dataset.apply(tf.data.experimental.group_by_window(
                key_func=example_to_bucket_id,
                reduce_func=batching_fn,
                window_size=None,
                window_size_func=window_size_fn))

    def build_metric_layer(self):
        return [AudioFramesMetricLayer("src"), SequenceTokenMetricLayer("trg"),
                BatchCountMetricLayer("src")]

    def get_eval_metric(self, args, name="metric", ds=None):
        """ Returns a neurst.metrics.metric.Metric object for evaluation."""
        if ds is not None and hasattr(ds, "trg_lang") and ds.trg_lang is not None:
            return build_metric(args[name + ".class"], language=ds.trg_lang,
                                **args[name + ".params"])
        return build_metric(args[name + ".class"], language=self._trg_data_pipeline.meta["language"],
                            **args[name + ".params"])


@register_task
class MultiTaskSpeechTranslation(Task):

    def __init__(self, args):
        """ Initializes with configuration. """
        super(MultiTaskSpeechTranslation, self).__init__(args)
        transcript_dp_cls = args.get("transcript_data_pipeline.class", TextDataPipeline)
        transcript_dp_params = args.get("transcript_data_pipeline.params", None) or {}
        self._transcript_data_pipeline = build_data_pipeline(
            transcript_dp_cls, **transcript_dp_params)
        translation_dp_cls = args.get("translation_data_pipeline.class", TextDataPipeline)
        translation_dp_params = args.get("translation_data_pipeline.params", None) or {}
        self._translation_data_pipeline = build_data_pipeline(
            translation_dp_cls, **translation_dp_params)

    def get_config(self):
        return {
            "transcript_data_pipeline.class": self._transcript_data_pipeline.__class__.__name__,
            "transcript_data_pipeline.params": self._transcript_data_pipeline.get_config(),
            "translation_data_pipeline.class": self._translation_data_pipeline.__class__.__name__,
            "translation_data_pipeline.params": self._translation_data_pipeline.get_config(),
        }

    @staticmethod
    def class_or_method_args():
        """ Returns a list of args for flag definition. """
        this_args = super(SpeechToText, SpeechToText).class_or_method_args()
        this_args.extend([
            ModuleFlag("transcript_data_pipeline", DataPipeline.REGISTRY_NAME,
                       default=TextDataPipeline.__name__,
                       help="The data pipeline for ASR transcription."),
            ModuleFlag("translation_data_pipeline", DataPipeline.REGISTRY_NAME,
                       default=TextDataPipeline.__name__,
                       help="The data pipeline for translation."),
        ])
        return this_args

    def inputs_signature(self, mode) -> Tuple[dict, dict]:
        """ Returns the input dtypes and signatures (from dataset). """
        dtypes = {"audio": tf.float32, "audio_length": tf.int64}
        # [batch, frames, feature_dim]
        signatures = {"audio": tf.TensorShape([None, None]),
                      "audio_length": tf.TensorShape([None, ])}
        if mode == compat.ModeKeys.INFER:
            return dtypes, signatures
        dtypes["transcript"] = tf.int64
        signatures["transcript"] = tf.TensorShape([None, None])
        dtypes["translation"] = tf.int64
        signatures["translation"] = tf.TensorShape([None, None])
        return dtypes, signatures

    def example_to_input(self, batch_of_data: dict, mode) -> dict:
        """ Converts a batch of data into the model readable structure. """
        raise NotImplementedError

    def get_data_preprocess_fn(self, mode, data_status, args=None) -> callable:
        """ Returns a callable function that preprocess the data sample
            according to this task. """
        _ = args
        if mode != compat.ModeKeys.TRAIN:
            raise NotImplementedError

        # if data_status["audio"] != compat.DataStatus.PROJECTED:
        #     raise RuntimeError("We recommend one to preprocess the audio in advance.")

        def _process_text(text, status, dp):
            if status == compat.DataStatus.RAW:
                text = dp.encode(text, is_processed=False)
            return text

        def _process(example):
            return {"audio": example["audio"],
                    "transcript": _process_text(example["transcript"], data_status["transcript"],
                                                self._transcript_data_pipeline),
                    "translation": _process_text(example["translation"], data_status["translation"],
                                                 self._translation_data_pipeline)
                    }

        return _process

    def get_data_postprocess_fn(self, mode) -> callable:
        """ Returns a callable function that postprocess the data sample
            according to this task. """
        raise NotImplementedError

    def create_and_batch_tfds(self,
                              ds: Dataset,
                              mode,
                              args=None,
                              num_replicas_in_sync=1) -> tf.data.Dataset:
        """ Batch dataset. """
        raise NotImplementedError

    def build_model(self, args, name=None):
        """Build a new model instance."""
        raise NotImplementedError
