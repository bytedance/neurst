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

import tensorflow as tf
from absl import logging

from neurst.data.datasets import register_dataset
from neurst.data.datasets.audio.audio_dataset import RawAudioDataset


@register_dataset("Librispeech")
class LibriSpeech(RawAudioDataset):
    """
    LibriSpeech is a corpus of approximately 1000 hours of read English speech.
        Homepage: http://www.openslr.org/12
        The raw dataset contains 7 files:
            - train-clean-100.tar.gz
            - train-clean-360.tar.gz
            - train-other-500.tar.gz
            - dev-clean.tar.gz
            - dev-other.tar.gz
            - test-clean.tar.gz
            - test-other.tar.gz
    """

    def __init__(self, args):
        super(LibriSpeech, self).__init__(args)
        self._transcripts_dict = None

    def load_transcripts(self):
        """ Loads transcripts (and translations if exists). """
        if self._transcripts_dict is not None:
            return
        logging.info(f"Loading transcriptions from tarball: {self._input_tarball}")
        n = 0
        trans = {}
        level1_cnt = 0
        level2_cnt = 0
        excluded_count = 0
        self._transcripts = []

        def read(f, filename):
            # The file LibriSpeech/dev-clean/3170/137482/3170-137482.trans.txt
            # will contain lines such as:
            # 3170-137482-0000 WITH AN EDUCATION WHICH OUGHT TO ...
            # 3170-137482-0001 I WAS COMPELLED BY POVERTY ...
            key = filename.strip(".trans.txt")
            path0, path1 = key.split("/")[-1].split("-")
            this_dict = {}
            bad_count = 0
            for line in f.readlines():
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                tid, txt = line.strip("\n").split(" ", 1)
                if self._validate(txt):
                    txt = txt.lower()
                    this_dict[tid] = txt
                    self._transcripts.append(txt)
                else:
                    bad_count += 1
                    this_dict[tid] = ""
            logging.info("[%s] = %d utterances.", key, len(this_dict))
            if path0 not in trans:
                trans[path0] = dict()
            trans[path0][path1] = this_dict
            return path0, len(this_dict)

        if tf.io.gfile.isdir(self._input_tarball):
            level0_pathwildcard = os.path.join(self._input_tarball, "*")
            level1_paths = tf.io.gfile.glob(os.path.join(level0_pathwildcard, "*"))
            level0_cnt = len(tf.io.gfile.glob(level0_pathwildcard))
            level1_cnt = len(level1_paths)
            for path in level1_paths:
                for filename in tf.io.gfile.glob(os.path.join(path, "*")):
                    n += 1
                    if n % 10000 == 0:
                        logging.info("Scanned %d entries...", n)
                    if not filename.endswith(".trans.txt"):
                        continue
                    with tf.io.gfile.GFile(filename, "r") as f:
                        _, trans_size = read(f, filename)
                    level2_cnt += trans_size
        else:
            level0 = set()
            with self.open_tarball("tar") as tar:
                for tarinfo in tar:
                    if not tarinfo.isreg():
                        continue
                    n += 1
                    if n % 10000 == 0:
                        logging.info("Scanned %d entries...", n)
                    if not tarinfo.name.endswith(".trans.txt"):
                        continue
                    f = tar.extractfile(tarinfo)
                    path0, trans_size = read(f, tarinfo.name)
                    f.close()
                    level1_cnt += 1
                    level0.add(path0)
                    level2_cnt += trans_size
            level0_cnt = len(level0)
        logging.info("Total %d directories, %d sub-directories, %d utterances, %d matched excluded file",
                     level0_cnt, level1_cnt, level2_cnt, excluded_count)
        # {'2277': {'149896': {'2277-149896-0000': "HE WAS IN A FEVERED STATE OF MIND OWING TO THE', ...}, ...}
        self._transcripts_dict = trans

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

            def get_sample(file=None, fileobj=None):
                audio = self.extract_audio_feature(fileobj=fileobj, file=file, mode="flac")
                if audio is None:
                    logging.info("Detected 1 nan/inf audio feature. SKIP...")
                    return None

                data_sample = self._pack_example_as_dict(audio=audio, transcript=this_trans,
                                                         src_lang=self.LANGUAGES.EN)
                if map_func is None:
                    return data_sample
                else:
                    return map_func(data_sample)

            if tf.io.gfile.isdir(self._input_tarball):
                level0_pathwildcard = os.path.join(self._input_tarball, "*")
                level1_paths = tf.io.gfile.glob(os.path.join(level0_pathwildcard, "*"))

                for path in level1_paths:
                    for filename in tf.io.gfile.glob(os.path.join(path, "*")):
                        if not filename.endswith(".flac"):
                            continue
                        uttid = re.sub(".*/(.+)\\.flac", "\\1", filename)
                        path0, path1, _ = uttid.strip().split("-")
                        this_trans = self._transcripts_dict[path0][path1][uttid]
                        if this_trans.strip() == "":
                            continue
                        n += 1
                        if total_shards > 1:
                            if n < range_begin:
                                continue
                            if n >= range_end:
                                break
                        sample = get_sample(file=filename)
                        if sample is None:
                            continue
                        yield sample
            else:
                with self.open_tarball("tar") as tar:
                    n = 0
                    for tarinfo in tar:
                        if not tarinfo.isreg():
                            continue
                        if not tarinfo.name.endswith(".flac"):
                            continue

                        uttid = re.sub(".*/(.+)\\.flac", "\\1", tarinfo.name)
                        path0, path1, _ = uttid.strip().split("-")
                        this_trans = self._transcripts_dict[path0][path1][uttid]
                        if this_trans.strip() == "":
                            continue
                        n += 1
                        if total_shards > 1:
                            if n < range_begin:
                                continue
                            if n >= range_end:
                                break
                        f = tar.extractfile(tarinfo)
                        sample = get_sample(fileobj=f)
                        f.close()
                        if sample is None:
                            continue
                        yield sample

        return gen
