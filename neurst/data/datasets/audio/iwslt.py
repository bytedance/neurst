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
import re
import time

import tensorflow as tf
import yaml
from absl import logging

from neurst.data.datasets import register_dataset
from neurst.data.datasets.audio.audio_dataset import RawAudioDataset


def _extract_from_xml(lines):
    texts = []
    for x in lines:
        if isinstance(x, bytes):
            x = x.decode("utf-8")
        x = x.strip()
        if x.startswith("<seg"):
            x = x.replace("</seg>", "")
            x = re.sub('<seg id="[0-9]*">', "", x)
            texts.append(x)
    return texts


@register_dataset
class IWSLT(RawAudioDataset):
    """ The IWSLT evaluation campaign. """

    def __init__(self, args):
        super(IWSLT, self).__init__(args)
        self._transc_transla_dict = None

    def load_transcripts(self):
        if self._transc_transla_dict is not None:
            return
        logging.info(f"Loading transcriptions and translations from tarball: {self._input_tarball}")
        ens = None
        des = None
        segments = None
        if self._input_tarball.endswith(".zip"):  # the tarball for training
            with self.open_tarball("zip") as fp:
                for n in fp.namelist():
                    if n.endswith("train.en"):
                        ens = fp.read(n).decode("utf-8").strip().split("\n")
                    elif n.endswith("train.de"):
                        des = fp.read(n).decode("utf-8").strip().split("\n")
                    elif n.endswith("train.yaml"):
                        segments = yaml.load(fp.read(n).decode("utf-8"), Loader=yaml.FullLoader)
                    if segments is not None and ens is not None and des is not None:
                        break
        elif self._input_tarball.endswith(".tgz") or self._input_tarball.endswith(".tar.gz"):
            with self.open_tarball("tar") as tar:
                for tarinfo in tar:
                    if tarinfo.name.endswith(".en.xml"):
                        f = tar.extractfile(tarinfo)
                        ens = _extract_from_xml(f.readlines())
                        f.close()
                    elif tarinfo.name.endswith(".de.xml"):
                        f = tar.extractfile(tarinfo)
                        des = _extract_from_xml(f.readlines())
                        f.close()
                    elif tarinfo.name.endswith(".en-de.yaml"):
                        f = tar.extractfile(tarinfo)
                        segments = yaml.load(f.read().decode("utf-8"), Loader=yaml.FullLoader)
                        f.close()
                    if segments is not None and ens is not None and des is not None:
                        break
        else:
            raise ValueError(f"Unknown tarball type: {self._input_tarball}")
        if ens is None:
            ens = [None] * len(segments)
        if des is None:
            des = [None] * len(segments)
        self._transcripts = []
        self._translations = []
        n = 0
        total = 0
        self._transc_transla_dict = dict()
        for mapp, en, de in zip(segments, ens, des):
            if en is None and de is None:
                self._transcripts.append(en)
                self._translations.append(de)
            elif self._validate(en) and self._validate(de):
                en = en.strip()
                de = de.strip()
                self._transcripts.append(en)
                self._translations.append(de)
            else:
                n += 1
                continue
            total += 1
            wavname = mapp["wav"].split("/")[-1]
            if wavname not in self._transc_transla_dict:
                self._transc_transla_dict[wavname] = []
            self._transc_transla_dict[wavname].append(
                {
                    "duration": float(mapp["duration"]),
                    "offset": float(mapp["offset"]),
                    "transcript": en,
                    "translation": de
                }
            )
        logging.info("Total %d utterances (%d skipped).", total, n)

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
            if self._input_tarball.endswith(".zip"):  # the tarball for training
                with self.open_tarball("zip") as fp:
                    for n in fp.namelist():
                        if n.endswith(".wav"):
                            tmp_wavname = f"ram://_tmpwav{time.time()}.wav"
                            with tf.io.gfile.GFile(tmp_wavname, "wb") as fw:
                                fw.write(fp.read(n))
                            with tf.io.gfile.GFile(tmp_wavname, "rb") as f:
                                audio_segment = AudioSegment.from_file(f, "wav")
                            yield n.strip().split("/")[-1], audio_segment
            elif self._input_tarball.endswith(".tgz") or self._input_tarball.endswith(".tar.gz"):
                with self.open_tarball("tar") as tar:
                    for tarinfo in tar:
                        if tarinfo.name.endswith(".wav"):
                            f = tar.extractfile(tarinfo)
                            audio_segment = AudioSegment.from_file(f, "wav")
                            yield tarinfo.name.strip().split("/")[-1], audio_segment
                            f.close()

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
                        transcript=sample["transcript"], translation=sample["translation"],
                        src_lang=self.LANGUAGES.EN, trg_lang=self.LANGUAGES.DE)
                    tf.io.gfile.remove(tmp_wav_file)
                    if map_func is None:
                        yield data_sample
                    else:
                        yield map_func(data_sample)
                if hit_end:
                    break

        return gen
