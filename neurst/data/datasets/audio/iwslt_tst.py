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
import yaml
from absl import logging

from neurst.data.datasets import register_dataset
from neurst.data.datasets.audio.audio_dataset import RawAudioDataset


@register_dataset
class IWSLTTest(RawAudioDataset):
    """ The IWSLT evaluation campaign. """

    def __init__(self, args):
        super(IWSLTTest, self).__init__(args)
        self._transc_transla_dict = None

    @staticmethod
    def class_or_method_args():
        this_args = super(IWSLTTest, IWSLTTest).class_or_method_args()
        return this_args

    def load_transcripts(self):
        if self._transc_transla_dict is not None:
            return
        logging.info(f"Loading transcriptions and translations from tarball: {self._input_tarball}")
        segments = None
        if self._input_tarball.endswith(".tgz") or self._input_tarball.endswith(".tar.gz"):
            with self.open_tarball("tar") as tar:
                for tarinfo in tar:
                    if tarinfo.name.endswith(".en-de.yaml"):
                        f = tar.extractfile(tarinfo)
                        segments = []
                        for seg in f.readlines():
                            if isinstance(seg, bytes):
                                seg = seg.decode("utf-8")
                            seg = seg.strip()
                            if seg == "":
                                continue
                            if seg.startswith("-"):
                                segments.append(seg)
                            else:
                                segments[-1] += seg
                        f.close()
                        segments = yaml.load("\n".join(segments), Loader=yaml.FullLoader)
                        break
        elif tf.io.gfile.isdir(self._input_tarball):
            for filename in tf.io.gfile.glob(os.path.join(self._input_tarball, "*")):
                if filename.endswith(".en-de.yaml"):
                    segments = []
                    with tf.io.gfile.GFile(filename) as f:
                        for seg in f.readlines():
                            if isinstance(seg, bytes):
                                seg = seg.decode("utf-8")
                            seg = seg.strip()
                            if seg == "":
                                continue
                            if seg.startswith("-"):
                                segments.append(seg)
                            else:
                                segments[-1] += seg
                    segments = yaml.load("\n".join(segments), Loader=yaml.FullLoader)
                    break
        else:
            raise ValueError(f"Unknown tarball type: {self._input_tarball}")
        assert segments, "Fail to load segmentation file."
        total = 0
        self._transc_transla_dict = dict()
        self._wav_order = []
        for mapp in segments:
            wavname = mapp["wav"].split("/")[-1]
            if wavname not in self._transc_transla_dict:
                self._transc_transla_dict[wavname] = []
                self._wav_order.append(wavname)
            self._transc_transla_dict[wavname].append(
                {
                    "duration": float(mapp["duration"]),
                    "offset": float(mapp["offset"]),
                }
            )
            total += 1
        logging.info("Total %d segments.", total)

    def build_iterator(self, map_func=None, shard_id=0, total_shards=1):
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

        from pydub import AudioSegment

        def file_iterator():
            if self._input_tarball.endswith(".tgz") or self._input_tarball.endswith(".tar.gz"):
                for wavname in self._wav_order:
                    with self.open_tarball("tar") as tar:
                        for tarinfo in tar:
                            if tarinfo.name.endswith(wavname):
                                f = tar.extractfile(tarinfo)
                                audio_segment = AudioSegment.from_file(f, "wav")
                                yield tarinfo.name.strip().split("/")[-1], audio_segment
                                f.close()
                            break
            elif tf.io.gfile.isdir(self._input_tarball):
                for wavname in self._wav_order:
                    yield wavname, AudioSegment.from_file(os.path.join(self._input_tarball, f"wavs/{wavname}"), "wav")
            else:
                raise ValueError(f"Unknown tarball type: {self._input_tarball}")

        def gen():
            n = 0
            hit_end = False
            if self._transc_transla_dict is None:
                self.load_transcripts()
            for wavname, audio_segment in file_iterator():
                for sample in self._transc_transla_dict[wavname]:
                    n += 1
                    if total_shards > 1:
                        if n < range_begin:
                            continue
                        if n >= range_end:
                            hit_end = True
                            break
                    tmp_wav_file = os.path.join(os.path.dirname(__file__), f"_tmp{time.time()}.wav")
                    audio_segment[int(sample["offset"] * 1000):
                                  int((sample["offset"] + sample["duration"]) * 1000) + 1].set_frame_rate(
                        16000).export(tmp_wav_file, format="wav")
                    data_sample = self._pack_example_as_dict(
                        audio=self.extract_audio_feature(file=tmp_wav_file, mode="wav"),
                        src_lang=self.LANGUAGES.EN, trg_lang=self.LANGUAGES.DE,
                        wav_name=wavname, offset=[sample["offset"]])
                    tf.io.gfile.remove(tmp_wav_file)
                    if map_func is None:
                        yield data_sample
                    else:
                        yield map_func(data_sample)
                if hit_end:
                    break

        return gen
