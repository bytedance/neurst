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
import io
import os
import re

import tensorflow as tf
import yaml
from absl import logging

from neurst.data.datasets import register_dataset
from neurst.data.datasets.audio.audio_dataset import RawAudioDataset
from neurst.utils.compat import DataStatus
from neurst.utils.flags_core import Flag

try:
    import soundfile
except (ImportError, OSError):
    pass


@register_dataset
class MuSTC(RawAudioDataset):
    """
    MuST-C(Gangi et al.,2019) is a multilingual speech translation corpus from English
    to 8 languages: Dutch (NL), French (FR), German (DE), Italian (IT), Portuguese (PT),
    Romanian (RO), Russian (RU) and Spanish (ES). MuST-C comprises at least 385 hours of
    audio recordings from English TED talks with their manual transcriptions and
    translations at sentence level for training.

    Homepage: https://ict.fbk.eu/must-c/
    """
    EXTRACTION_CHOICES = ["train", "dev", "tst", "test", "tst-HE", "tst-COMMON"]
    TARGET_LANGUAGES = ["nl", "fr", "de", "it", "pt", "ro", "ru", "es"]

    def __init__(self, args):
        super(MuSTC, self).__init__(args)
        self._extraction = args["extraction"]
        if self._extraction not in MuSTC.EXTRACTION_CHOICES:
            raise ValueError("`extraction` for MuST-C dataset must be "
                             "one of {}".format(", ".join(MuSTC.EXTRACTION_CHOICES)))
        if self._extraction in ["tst", "test"]:
            self._extraction = "tst-COMMON"
        self._transc_transla_dict = None

    @staticmethod
    def class_or_method_args():
        this_args = super(MuSTC, MuSTC).class_or_method_args()
        this_args.append(
            Flag("extraction", dtype=Flag.TYPE.STRING, default=None,
                 choices=MuSTC.EXTRACTION_CHOICES,
                 help="The dataset portion to be extracted, e.g. train, dev, test (tst-COMMON)."))
        return this_args

    @property
    def status(self):
        return {
            "audio": DataStatus.RAW,
            "transcript": DataStatus.RAW,
            "translation": DataStatus.RAW
        }

    def load_transcripts(self):
        """ Loads transcripts and translations. """
        if self._transc_transla_dict is not None:
            return
        logging.info(f"Loading transcriptions from tarball: {self._input_tarball}")
        srcs = None
        trgs = None
        segments = None
        if tf.io.gfile.isdir(self._input_tarball):
            path = os.path.join(self._input_tarball, f"data/{self._extraction}/txt")
            for filename in tf.io.gfile.listdir(path):
                with tf.io.gfile.GFile(os.path.join(path, filename), "rb") as fp:
                    if filename.endswith(".yaml"):
                        segments = yaml.load(fp.read().decode("utf-8"), Loader=yaml.FullLoader)
                    elif filename.endswith(".en"):
                        srcs = fp.read().decode("utf-8").strip().split("\n")
                    elif filename[-2:] in MuSTC.TARGET_LANGUAGES:
                        trgs = fp.read().decode("utf-8").strip().split("\n")
                    if srcs is not None and trgs is not None and segments is not None:
                        break
        else:
            with self.open_tarball("tar") as tar:
                for tarinfo in tar:
                    if not tarinfo.isreg():
                        continue
                    if tarinfo.name.endswith(f"{self._extraction}.yaml"):
                        f = tar.extractfile(tarinfo)
                        segments = yaml.load(f.read().decode("utf-8"), Loader=yaml.FullLoader)
                        f.close()
                    elif tarinfo.name.endswith(f"{self._extraction}.en"):
                        f = tar.extractfile(tarinfo)
                        srcs = f.read().decode("utf-8").strip().split("\n")
                        f.close()
                    elif re.match(r"^.*/{}.({})$".format(self._extraction, "|".join(MuSTC.TARGET_LANGUAGES)),
                                  tarinfo.name):
                        f = tar.extractfile(tarinfo)
                        trgs = f.read().decode("utf-8").strip().split("\n")
                        f.close()
                    if srcs is not None and trgs is not None and segments is not None:
                        break
        assert len(srcs) == len(trgs) == len(segments)
        self._transc_transla_dict = {}
        self._transcripts = []
        self._translations = []
        n = 0
        for src, trg, seg in zip(srcs, trgs, segments):
            src = src.strip()
            trg = trg.strip()
            if not src or not trg:
                continue
            if seg["wav"] not in self._transc_transla_dict:
                self._transc_transla_dict[seg["wav"]] = []
            self._transc_transla_dict[seg["wav"]].append([float(seg["offset"]),
                                                          float(seg["duration"]), src, trg])
            n += 1
            self._transcripts.append(src)
            self._translations.append(trg)
        logging.info("Total %d utterances (%d skipped).", n, len(srcs) - n)

    def build_iterator(self, map_func=None, shard_id=0, total_shards=1):
        """ Returns the iterator of the dataset.

        Args:
            map_func: A function mapping a dataset element to another dataset element.
        """
        if total_shards > 1:
            total_samples = self.num_samples
            samples_per_part = total_samples // total_shards
            range_begin = samples_per_part * shard_id
            if shard_id == total_shards - 1:
                range_end = total_samples + 1
                logging.info(f"Iterate on dataset from {range_begin} "
                             f"to the end (total {total_samples}).")
            else:
                range_end = range_begin + samples_per_part
                logging.info(f"Iterate on dataset from {range_begin} "
                             f"to {range_end} (total {total_samples}).")

        def gen():
            if self._transc_transla_dict is None:
                self.load_transcripts()
            n = 0
            former_wavkey = None
            hit_end = False
            ori_audio = None
            sample_rate = None
            num_processed = 0
            if tf.io.gfile.isdir(self._input_tarball):
                for wavname in self._transc_transla_dict:
                    for offset, duration, transcript, transla in self._transc_transla_dict[wavname]:
                        n += 1
                        if total_shards > 1:
                            if n < range_begin:
                                continue
                            if n >= range_end:
                                hit_end = True
                                break
                        if wavname != former_wavkey:
                            former_wavkey = wavname
                            ori_audio, sample_rate = soundfile.read(
                                os.path.join(self._input_tarball, f"data/{self._extraction}/wav/{wavname}"),
                                dtype='int16')
                        start = int(offset * sample_rate)
                        end = int((offset + duration) * sample_rate) + 1
                        data_sample = {
                            "audio": self.extract_audio_feature(
                                sig=ori_audio[start:end], rate=sample_rate),
                            "transcript": transcript,
                            "translation": transla,
                        }
                        if map_func is None:
                            yield data_sample
                        else:
                            yield map_func(data_sample)
                    if hit_end:
                        break

            else:
                with self.open_tarball("tar") as tar:
                    for tarinfo in tar:
                        if num_processed == len(self._transc_transla_dict):
                            break
                        if not tarinfo.isreg():
                            continue
                        if self._extraction not in tarinfo.name:
                            continue
                        if not tarinfo.name.endswith(".wav"):
                            continue
                        wavname = tarinfo.name.split("/")[-1]
                        if wavname not in self._transc_transla_dict:
                            continue
                        num_processed += 1
                        for offset, duration, transcript, transla in self._transc_transla_dict[wavname]:
                            n += 1
                            if total_shards > 1:
                                if n < range_begin:
                                    continue
                                if n >= range_end:
                                    hit_end = True
                                    break
                            if wavname != former_wavkey:
                                former_wavkey = wavname
                                f = tar.extractfile(tarinfo)
                                b = io.BytesIO(f.read())
                                f.close()
                                ori_audio, sample_rate = soundfile.read(b, dtype='int16')
                                b.close()
                            start = int(offset * sample_rate)
                            end = int((offset + duration) * sample_rate) + 1
                            data_sample = {
                                "audio": self.extract_audio_feature(
                                    sig=ori_audio[start:end], rate=sample_rate),
                                "transcript": transcript,
                                "translation": transla,
                            }
                            if map_func is None:
                                yield data_sample
                            else:
                                yield map_func(data_sample)
                        if hit_end:
                            break

        return gen
