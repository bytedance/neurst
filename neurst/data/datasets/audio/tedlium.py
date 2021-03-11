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
import os
import time

import tensorflow as tf
from absl import logging

from neurst.data.datasets import register_dataset
from neurst.data.datasets.audio.audio_dataset import RawAudioDataset
from neurst.utils.flags_core import Flag


@register_dataset("TedLIUM")
class TedLium(RawAudioDataset):
    """
    TED-LIUM is an ASR dataset. TED-LIUM-release-3 contains 2351 audio talks in NIST sphere format (SPH),
    452 hours of audio and 2351 aligned automatic transcripts in STM format. All talks and text
    are property of TED Conferences LLC.

    Paper:  TED-LIUM 3: twice as much data and corpus repartition for experiments on speaker adaptation
    Homepage: https://www.openslr.org/51/
    """
    EXTRACTION_CHOICES = ["dev", "test", "train"]
    VERSIONS = ["release-1", "release-2", "releaser-3"]

    def __init__(self, args):
        super(TedLium, self).__init__(args)
        self._extraction = args["extraction"]
        if self._extraction not in TedLium.EXTRACTION_CHOICES:
            raise ValueError("`extraction` for TED_LIUM dataset must be "
                             "one of {}".format(", ".join(TedLium.EXTRACTION_CHOICES)))
        self._transcripts_dict = None

    @staticmethod
    def class_or_method_args():
        this_args = super(TedLium, TedLium).class_or_method_args()
        this_args.append(
            Flag("extraction", dtype=Flag.TYPE.STRING, default=None,
                 choices=TedLium.EXTRACTION_CHOICES,
                 help="The dataset portion to be extracted, e.g. train, dev, test."))
        return this_args

    def load_transcripts(self):
        """ Loads transcripts and translations.
        xxx/train.tsv: client_id \t path \t sentence \t up_votes \t down_votes \t age \t gender \t accent
        """
        if self._transcripts_dict is not None:
            return
        logging.info(f"Loading transcriptions from tarball: {self._input_tarball}")
        self._transcripts_dict = {}
        self._transcripts = []

        def add(line):
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            tokens = line.strip().split()
            name = tokens[0]
            start = float(tokens[3])
            end = float(tokens[4])
            txt = (" ".join(tokens[6:])).strip()
            if (txt == "ignore_time_segment_in_scoring" or "<unk>" in txt
                or not self._validate(txt)):
                return
            if name not in self._transcripts_dict:
                self._transcripts_dict[name] = []
            self._transcripts_dict[name].append([start, end, txt])
            self._transcripts.append(txt)

        if tf.io.gfile.isdir(self._input_tarball):
            input_dir = os.path.join(self._input_tarball, f"legacy/{self._extraction}/stm")
            for stm_file in tf.io.gfile.glob(os.path.join(input_dir, "*.stm")):
                with tf.io.gfile.GFile(stm_file, "r") as fp:
                    for line in fp:
                        add(line)
        else:
            with self.open_tarball("tar") as tar:
                for tarinfo in tar:
                    if not tarinfo.isreg():
                        continue
                    if f"legacy/{self._extraction}/stm" not in tarinfo.name:
                        continue
                    if not tarinfo.name.endswith(".stm"):
                        continue
                    f = tar.extractfile(tarinfo)
                    for line in f:
                        add(line)
                    f.close()
        logging.info("Total %d utterances.", len(self._transcripts))

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
            if self._transcripts_dict is None:
                self.load_transcripts()

            n = 0
            hit_end = False
            tmp_sph_file = os.path.join(os.path.dirname(__file__), f"_tmp{time.time()}.sph")
            tmp_wav_file = os.path.join(os.path.dirname(__file__), f"_tmp{time.time()}.wav")

            from pydub import AudioSegment

            if tf.io.gfile.isdir(self._input_tarball):
                input_dir = os.path.join(self._input_tarball, f"legacy/{self._extraction}/sph")
                for sph_file in tf.io.gfile.glob(os.path.join(input_dir, "*.sph")):
                    audio_prefix = sph_file.split("/")[-1].split(".")[0]
                    if audio_prefix not in self._transcripts_dict:
                        continue

                    # write to local tmp_sph_file from hdfs
                    with tf.io.gfile.GFile(tmp_sph_file, "wb") as fw:
                        f = tf.io.gfile.GFile(sph_file, "rb")
                        fw.write(f.read())
                        f.close()

                    sph = AudioSegment.from_file(tmp_sph_file, "nistsphere")
                    for start, end, transcript in self._transcripts_dict[audio_prefix]:
                        n += 1
                        if total_shards > 1:
                            if n < range_begin:
                                continue
                            if n >= range_end:
                                hit_end = True
                                break
                        sph[int(start * 1000): int(end * 1000) + 1].set_frame_rate(16000).export(
                            tmp_wav_file, format="wav")
                        audio = self.extract_audio_feature(file=tmp_wav_file, mode="wav")
                        if audio is None:
                            logging.info("Detected 1 nan/inf audio feature. SKIP...")
                            continue
                        data_sample = self._pack_example_as_dict(audio=audio,
                                                                 transcript=transcript, src_lang="en")
                        if map_func is None:
                            yield data_sample
                        else:
                            yield map_func(data_sample)
                    if hit_end:
                        break

            else:
                with self.open_tarball("tar") as tar:
                    for tarinfo in tar:
                        if not tarinfo.isreg():
                            continue
                        if f"legacy/{self._extraction}/sph" not in tarinfo.name:
                            continue
                        if not tarinfo.name.endswith(".sph"):
                            continue
                        audio_prefix = tarinfo.name.split("/")[-1].split(".")[0]
                        if audio_prefix not in self._transcripts_dict:
                            continue
                        # write to local tmp_sph_file from hdfs
                        with tf.io.gfile.GFile(tmp_sph_file, "wb") as fw:
                            f = tar.extractfile(tarinfo)
                            fw.write(f.read())
                            f.close()
                        sph = AudioSegment.from_file(tmp_sph_file, "nistsphere")

                        for start, end, transcript in self._transcripts_dict[audio_prefix]:
                            n += 1
                            if total_shards > 1:
                                if n < range_begin:
                                    continue
                                if n >= range_end:
                                    hit_end = True
                                    break
                            sph[int(start * 1000): int(end * 1000) + 1].set_frame_rate(16000).export(
                                tmp_wav_file, format="wav")
                            audio = self.extract_audio_feature(file=tmp_wav_file, mode="wav")
                            if audio is None:
                                logging.info("Detected 1 nan/inf audio feature. SKIP...")
                                continue
                            data_sample = self._pack_example_as_dict(audio=audio,
                                                                     transcript=transcript, src_lang="en")
                            if map_func is None:
                                yield data_sample
                            else:
                                yield map_func(data_sample)
                        if hit_end:
                            break

        return gen
