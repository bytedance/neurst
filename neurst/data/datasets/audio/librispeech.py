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
import re

import tensorflow as tf
from absl import logging

from neurst.data.datasets import register_dataset
from neurst.data.datasets.audio.audio_dataset import RawAudioDataset
from neurst.utils.compat import DataStatus
from neurst.utils.flags_core import Flag


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
        self._excluded_file = args["excluded_file"]
        self._excluded_list = None
        if self._excluded_file is not None:
            if not tf.io.gfile.exists(self._excluded_file):
                raise ValueError(f"`excluded_file` not found: {self._excluded_file}")
            with tf.io.gfile.GFile(self._excluded_file) as fp:
                self._excluded_list = [x.strip().lower() for x in fp]

        self._transcripts_dict = None

    @staticmethod
    def class_or_method_args():
        this_args = super(LibriSpeech, LibriSpeech).class_or_method_args()
        this_args.append(
            Flag("excluded_file", dtype=Flag.TYPE.STRING, default=None,
                 help="A file containing transcriptions "
                      "that would be removed in the LibriSpeech corpus."))
        return this_args

    @property
    def status(self):
        return {
            "audio": DataStatus.RAW,
            "transcript": DataStatus.RAW
        }

    def load_transcripts(self):
        """ Loads transcripts (and translations if exists). """
        if self._transcripts_dict is not None:
            return
        logging.info(f"Loading transcriptions from tarball: {self._input_tarball}")
        n = 0
        trans = {}
        level0 = set()
        level1_cnt = 0
        level2_cnt = 0
        excluded_count = 0
        excluded_str = ""
        if self._excluded_list is not None:
            excluded_str = " ".join(self._excluded_list)
        self._transcripts = []
        with self.open_tarball("tar") as tar:
            for tarinfo in tar:
                if not tarinfo.isreg():
                    continue
                n += 1
                if n % 10000 == 0:
                    logging.info("Scanned %d entries...", n)
                if not tarinfo.name.endswith(".trans.txt"):
                    continue
                level1_cnt += 1
                # The file LibriSpeech/dev-clean/3170/137482/3170-137482.trans.txt
                # will contain lines such as:
                # 3170-137482-0000 WITH AN EDUCATION WHICH OUGHT TO ...
                # 3170-137482-0001 I WAS COMPELLED BY POVERTY ...
                key = tarinfo.name.strip(".trans.txt")
                path0, path1 = key.split("/")[-1].split("-")
                level0.add(path0)
                f = tar.extractfile(tarinfo)
                this_dict = {}
                for line in f.readlines():
                    tid, txt = line.decode("utf-8").strip("\n").split(" ", 1)
                    txt_tokens = txt.split()
                    if txt in excluded_str:
                        excluded_count += 1
                        this_dict[tid] = ""
                    elif len(txt_tokens) > 10 and (
                        " ".join(txt_tokens[:len(txt_tokens) // 2]) in excluded_str
                        or " ".join(txt_tokens[len(txt_tokens) // 2:]) in excluded_str):
                        excluded_count += 1
                        this_dict[tid] = ""
                    else:
                        txt = txt.lower()
                        this_dict[tid] = txt
                        self._transcripts.append(txt)
                logging.info("[%s] = %d utterances.", key, len(this_dict))
                level2_cnt += len(this_dict)
                if path0 not in trans:
                    trans[path0] = dict()
                trans[path0][path1] = this_dict
                f.close()
        logging.info("Total %d directories, %d sub-directories, %d utterances, %d matched excluded file",
                     len(level0), level1_cnt, level2_cnt, excluded_count)
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
                    audio = self.extract_audio_feature(fileobj=f, mode="flac")
                    f.close()
                    data_sample = {
                        "audio": audio,
                        "transcript": this_trans
                    }
                    if map_func is None:
                        yield data_sample
                    else:
                        yield map_func(data_sample)

        return gen
