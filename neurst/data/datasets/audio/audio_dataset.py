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
import contextlib
import os
import tarfile
import time
import zipfile
from abc import ABCMeta, abstractmethod

import six
import tensorflow as tf
from absl import logging

from neurst.data.audio import FeatureExtractor, build_feature_extractor
from neurst.data.dataset_utils import glob_tfrecords, load_tfrecords
from neurst.data.datasets import Dataset, TFRecordDataset, register_dataset
from neurst.data.datasets.text_gen_dataset import TextGenDataset
from neurst.utils.compat import DataStatus
from neurst.utils.flags_core import Flag, ModuleFlag
from neurst.utils.misc import to_numpy_or_python_type

try:
    import soundfile
except (ImportError, OSError):
    pass


@six.add_metaclass(ABCMeta)
class RawAudioDataset(Dataset):
    """ Read from publicly available audio dataset.

    The raw dataset should yield element with following format:
        {
            "audio": ......,
            "transcript": ......,
            "translation": ......,
        }
    """

    def __init__(self, args):
        super(RawAudioDataset, self).__init__()
        self._input_tarball = args["input_tarball"]
        self._transcripts = None
        self._translations = None
        self._feature_extractor = build_feature_extractor(args)
        try:
            import soundfile
            _ = soundfile
        except ImportError:
            raise ImportError("Please install soundfile with: pip3 install soundfile")
        except OSError:
            raise OSError("sndfile library not found. Please install with: "
                          "apt-get install libsndfile1")

    @staticmethod
    def class_or_method_args():
        return [
            Flag("input_tarball", dtype=Flag.TYPE.STRING, default=None,
                 help="The original tarball."),
            ModuleFlag(FeatureExtractor.REGISTRY_NAME, default=None,
                       help="The audio feature extractor.")
        ]

    @contextlib.contextmanager
    def open_tarball(self, mode="tar"):
        if not tf.io.gfile.exists(self._input_tarball):
            raise ValueError(f"Input tarball not found: {self._input_tarball}")
        mode = mode.lower()
        if mode == "tar":
            fp = tf.io.gfile.GFile(self._input_tarball, "rb")
            tar = tarfile.open(fileobj=fp, mode="r:*")
            yield tar
            tar.close()
            fp.close()
        elif mode == "zip":
            local_path = self._input_tarball
            if self._input_tarball.startswith("hdfs://"):
                local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          os.path.basename(self._input_tarball))
                if not os.path.exists(local_path):
                    if os.path.exists(local_path + ".incomplete"):
                        while not os.path.exists(local_path):
                            time.sleep(20)
                    else:
                        logging.info(f"Copying ZIP file from HDFS to local disk {local_path}...")
                        tf.io.gfile.copy(self._input_tarball, local_path + ".incomplete", overwrite=True)
                        tf.io.gfile.rename(local_path + ".incomplete", local_path, overwrite=True)
            fp = zipfile.ZipFile(local_path, "r")
            yield fp
            fp.close()
        else:
            raise NotImplementedError(f"Unsupported type of input tarball: {self._input_tarball}.")

    def extract_audio_feature(self, sig=None, rate=None,
                              file=None, fileobj=None,
                              mode="wav"):
        """ Reads the audio data and extracts audio features.

        Args:
            signal: A 1-D or 2-D numpy array of either integer or float data-type, the audio data.
            rate: int, the sample rate.
            file: str, the file to read from.
            fileobj: file-like object, the file to read from.
            mode: The format of the audio file.

        Returns:

        """
        if sig is None:
            assert (file is None) ^ (fileobj is None), (
                "Only one of `file` and `fileobj` should be provided.")
            mode = mode.lower()
            if mode in ["flac", "wav"]:
                sig, rate = soundfile.read(file or fileobj, dtype='float32')
            else:
                raise NotImplementedError
        if self._feature_extractor is None:
            return sig
        return self._feature_extractor(sig, rate)

    @property
    def transcripts(self):
        if self._transcripts is None:
            if self._translations is None:
                self.load_transcripts()
        return self._transcripts

    @property
    def translations(self):
        if self._translations is None:
            if self._transcripts is None:
                self.load_transcripts()
        return self._translations

    @property
    def num_samples(self):
        if self.transcripts is not None:
            return len(self.transcripts)
        elif self.translations is not None:
            return len(self.translations)
        raise ValueError("Fail to get the number of samples.")

    @property
    @abstractmethod
    def status(self):
        raise NotImplementedError

    @abstractmethod
    def load_transcripts(self):
        """ Loads transcripts (and translations if exists). """
        raise NotImplementedError

    @abstractmethod
    def build_iterator(self, map_func=None, shard_id=0, total_shards=1):
        """ Returns the iterator of the dataset.

        Args:
            map_func: A function mapping a dataset element to another dataset element.
            shard_id: Generator yields on the `shard_id`-th shard of the whole dataset.
            total_shards: The number of total shards.
        """
        raise NotImplementedError


@register_dataset("audio_tfrecord")
class AudioTFRecordDataset(TFRecordDataset, TextGenDataset):
    """ The TF Record Dataset for audio to text.
    The element spec must be
        {
            'audio': ...,
            'transcript': ...,
         }
    """

    def __init__(self, args):
        """ Initializes the dataset. """
        super(AudioTFRecordDataset, self).__init__(args)
        self._feature_key = args["feature_key"]
        self._transcript_key = args["transcript_key"]
        for x in tf.data.TFRecordDataset(glob_tfrecords(self._data_path)).take(1):
            example = tf.train.Example()
            example.ParseFromString(x.numpy())
            if len(example.features.feature[self._feature_key].float_list.value) > 0:
                self._audio_is_extracted = True
            elif len(example.features.feature[self._feature_key].int64_list.value) > 0:
                self._audio_is_extracted = False
            else:
                raise ValueError
            if len(example.features.feature[self._transcript_key].bytes_list.value) > 0:
                self._transcript_is_projected = False
            elif len(example.features.feature[self._transcript_key].int64_list.value) > 0:
                self._transcript_is_projected = True
            else:
                raise ValueError
        if not hasattr(self, "_audio_is_extracted"):
            raise ValueError(f"Fail to read {self._data_path}")

    @staticmethod
    def class_or_method_args():
        this_args = TFRecordDataset.class_or_method_args()
        this_args.extend([
            Flag("feature_key", dtype=Flag.TYPE.STRING, default="audio",
                 help="The key of the audio features in the TF Record."),
            Flag("transcript_key", dtype=Flag.TYPE.STRING, default="transcript",
                 help="The key of the audio transcript/translation in the TF Record."),
        ])
        return this_args

    @property
    def status(self):
        return {
            "audio": (DataStatus.PROJECTED if self._audio_is_extracted
                      else Dataset.RAW),  # audio
            "transcript": (DataStatus.PROJECTED if self._transcript_is_projected
                           else DataStatus.RAW),  # transcript
        }

    @property
    def fields(self):
        return {
            self._feature_key: (tf.io.VarLenFeature(tf.float32) if self._audio_is_extracted
                                else tf.io.VarLenFeature(tf.int64)),
            self._transcript_key: (tf.io.VarLenFeature(tf.int64) if self._transcript_is_projected
                                   else tf.io.VarLenFeature(tf.string)),
        }

    @property
    def targets(self):
        """ Returns a list of targets. """
        if self._targets is None:
            assert not self._transcript_is_projected
            gen = self.build_iterator(map_func=lambda x: x["transcript"])
            self._targets = [x for x in gen()]
        return self._targets

    def build(self, auto_shard=False, map_func=None, map_output_dtypes=None,
              shuffle=True) -> tf.data.Dataset:

        try:
            return load_tfrecords(
                self._data_path, shuffle=self._shuffle_dataset and shuffle,
                deterministic=(not shuffle),
                map_func=lambda x: x if map_func is None else map_func(x),
                auto_shard=auto_shard, name_to_features=self.fields,
                feature_name_mapping={self._feature_key: "audio", self._transcript_key: "transcript"})
        except AttributeError:

            logging.info("Call Dataset.from_generator for AudioTFRecord")

            def gen():
                for data in load_tfrecords(self._data_path, shuffle=self._shuffle_dataset and shuffle,
                                           deterministic=(not shuffle),
                                           auto_shard=auto_shard, name_to_features=self.fields,
                                           feature_name_mapping={self._feature_key: "audio",
                                                                 self._transcript_key: "transcript"}):
                    data = to_numpy_or_python_type(data, bytes_as_str=True)
                    if map_func is not None:
                        data = map_func(data)
                    yield data

            return tf.data.Dataset.from_generator(gen, output_types=map_output_dtypes)

    def build_iterator(self, map_func=None, shard_id=0, total_shards=1):
        """ Returns the iterator of the dataset. """

        def gen():
            ds = load_tfrecords(self._data_path, shuffle=False, auto_shard=False,
                                name_to_features=self.fields,
                                sharding_index=shard_id, num_shards=total_shards,
                                feature_name_mapping={self._feature_key: "audio",
                                                      self._transcript_key: "transcript"})
            for x in ds:
                data = to_numpy_or_python_type(x, bytes_as_str=True)
                if map_func is not None:
                    data = map_func(data)
                yield data

        return gen


@register_dataset("audio_triple_tfrecord")
class AudioTripleTFRecordDataset(TFRecordDataset, TextGenDataset):
    """ The TF Record Dataset for audio to text.
    The element spec must be
        {
            'audio': ...,
            'transcript': ...,
            'translation': ...
         }
    """

    def __init__(self, args):
        """ Initializes the dataset. """
        super(AudioTripleTFRecordDataset, self).__init__(args)
        self._feature_key = args["feature_key"]
        self._transcript_key = args["transcript_key"]
        self._translation_key = args["translation_key"]
        for x in tf.data.TFRecordDataset(glob_tfrecords(self._data_path)).take(1):
            example = tf.train.Example()
            example.ParseFromString(x.numpy())
            if len(example.features.feature[self._feature_key].float_list.value) > 0:
                self._audio_is_extracted = True
            elif len(example.features.feature[self._feature_key].int64_list.value) > 0:
                self._audio_is_extracted = False
            else:
                raise ValueError
            if len(example.features.feature[self._transcript_key].bytes_list.value) > 0:
                self._transcript_is_projected = False
            elif len(example.features.feature[self._transcript_key].int64_list.value) > 0:
                self._transcript_is_projected = True
            else:
                raise ValueError
            if len(example.features.feature[self._translation_key].bytes_list.value) > 0:
                self._translation_is_projected = False
            elif len(example.features.feature[self._translation_key].int64_list.value) > 0:
                self._translation_is_projected = True
            else:
                raise ValueError
        if not hasattr(self, "_audio_is_extracted"):
            raise ValueError(f"Fail to read {self._data_path}")
        self._transcripts = None

    @staticmethod
    def class_or_method_args():
        this_args = TFRecordDataset.class_or_method_args()
        this_args.extend([
            Flag("feature_key", dtype=Flag.TYPE.STRING, default="audio",
                 help="The key of the audio features in the TF Record."),
            Flag("transcript_key", dtype=Flag.TYPE.STRING, default="transcript",
                 help="The key of the audio transcript in the TF Record."),
            Flag("translation_key", dtype=Flag.TYPE.STRING, default="translation",
                 help="The key of the audio translation in the TF Record."),
        ])
        return this_args

    @property
    def status(self):
        return {
            "audio": (DataStatus.PROJECTED if self._audio_is_extracted
                      else Dataset.RAW),  # audio
            "transcript": (DataStatus.PROJECTED if self._transcript_is_projected
                           else DataStatus.RAW),  # transcript
            "translation": (DataStatus.PROJECTED if self._translation_is_projected
                            else DataStatus.RAW),  # transcript
        }

    @property
    def fields(self):
        return {
            self._feature_key: (tf.io.VarLenFeature(tf.float32) if self._audio_is_extracted
                                else tf.io.VarLenFeature(tf.int64)),
            self._transcript_key: (tf.io.VarLenFeature(tf.int64) if self._transcript_is_projected
                                   else tf.io.VarLenFeature(tf.string)),
            self._translation_key: (tf.io.VarLenFeature(tf.int64) if self._translation_is_projected
                                    else tf.io.VarLenFeature(tf.string)),
        }

    @property
    def targets(self):
        """ Returns a list of targets. """
        if self._targets is None:
            assert not self._translation_is_projected
            gen = self.build_iterator(map_func=lambda x: x["translation"])
            self._targets = [x for x in gen()]
        return self._targets

    @property
    def transcripts(self):
        """ Returns a list of transcripts. """
        if self._transcripts is None:
            assert not self._transcript_is_projected
            gen = self.build_iterator(map_func=lambda x: x["transcript"])
            self._transcripts = [x for x in gen()]
        return self._transcripts

    def build(self, auto_shard=False, map_func=None, map_output_dtypes=None,
              shuffle=True) -> tf.data.Dataset:

        try:
            return load_tfrecords(
                self._data_path, shuffle=self._shuffle_dataset and shuffle,
                deterministic=(not shuffle),
                map_func=lambda x: x if map_func is None else map_func(x),
                auto_shard=auto_shard, name_to_features=self.fields,
                feature_name_mapping={self._feature_key: "audio", self._transcript_key: "transcript",
                                      self._translation_key: "translation"})
        except AttributeError:

            logging.info("Call Dataset.from_generator for AudioTripleTFRecordDataset")

            def gen():
                for data in load_tfrecords(self._data_path, shuffle=self._shuffle_dataset and shuffle,
                                           deterministic=(not shuffle),
                                           auto_shard=auto_shard, name_to_features=self.fields,
                                           feature_name_mapping={self._feature_key: "audio",
                                                                 self._transcript_key: "transcript",
                                                                 self._translation_key: "translation"}):
                    data = to_numpy_or_python_type(data, bytes_as_str=True)
                    if map_func is not None:
                        data = map_func(data)
                    yield data

            return tf.data.Dataset.from_generator(gen, output_types=map_output_dtypes)

    def build_iterator(self, map_func=None, shard_id=0, total_shards=1):
        """ Returns the iterator of the dataset. """

        def gen():
            ds = load_tfrecords(self._data_path, shuffle=False, auto_shard=False,
                                name_to_features=self.fields,
                                sharding_index=shard_id, num_shards=total_shards,
                                feature_name_mapping={self._feature_key: "audio",
                                                      self._transcript_key: "transcript",
                                                      self._translation_key: "translation"})
            for x in ds:
                data = to_numpy_or_python_type(x, bytes_as_str=True)
                if map_func is not None:
                    data = map_func(data)
                yield data

        return gen
