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
import csv
import os
from io import StringIO

import tensorflow as tf
from absl import logging

from neurst.data.datasets import register_dataset
from neurst.data.datasets.audio.audio_dataset import RawAudioDataset
from neurst.utils.flags_core import Flag


def extract_last_two_number(fname):
    # common_voice_en_19594267.mp3
    num_str = fname.split(".")[0]
    last = num_str[-1]
    last_two = num_str[-2]
    return last, last_two


def get_transcription(fname, trans_included):
    fname = fname.strip().split("/")[-1]
    last, last_two = extract_last_two_number(fname)
    if last in trans_included:
        if last_two in trans_included[last]:
            if fname in trans_included[last][last_two]:
                return trans_included[last][last_two][fname]
    return None


@register_dataset("CommonVoice")
class CommonVoice(RawAudioDataset):
    """
    Common Voice is an ASR dataset. Each entry in the dataset consists of a unique MP3
    and corresponding text file. The dataset currently consists of 5,671 validated
    hours in 54 languages. There are 5 versions for the corpus.

    Homepage: https://commonvoice.mozilla.org/en/datasets
    Github: https://github.com/common-voice/cv-dataset
    """
    EXTRACTION_CHOICES = ["dev", "other", "test", "train", "validated", "invalidated"]

    def __init__(self, args):
        super(CommonVoice, self).__init__(args)

        self._extraction = args["extraction"]
        if self._extraction not in CommonVoice.EXTRACTION_CHOICES:
            raise ValueError("`extraction` for CommonVoice dataset must be "
                             "one of {}".format(", ".join(CommonVoice.EXTRACTION_CHOICES)))
        self._transcripts_dict = None
        if self._input_tarball.endswith(".tar"):
            self._language = self._input_tarball.split("/")[-1].split(".")[0].split('-')[0]
        elif tf.io.gfile.isdir(self._input_tarball):
            self._language = args["language"]
            assert self._language, "language must be provided."
        else:
            raise ValueError(f"Unknown type of input: {self._input_tarball}")

    @staticmethod
    def class_or_method_args():
        this_args = super(CommonVoice, CommonVoice).class_or_method_args()
        this_args.extend([
            Flag("extraction", dtype=Flag.TYPE.STRING, default=None,
                 choices=CommonVoice.EXTRACTION_CHOICES,
                 help="The dataset portion to be extracted, i.e. train, dev, test, other, validated."),
            Flag("language", dtype=Flag.TYPE.STRING, default=None,
                 help="the language portion to be extracted, e.g. en, zh-CN. Must be provided "
                      "if the input is a directory.")])
        return this_args

    def load_transcripts(self):
        """ Loads transcripts and translations.
        xxx/train.tsv: client_id \t path \t sentence \t up_votes \t down_votes \t age \t gender \t accent
        """
        if self._transcripts_dict is not None:
            return
        logging.info(f"Loading transcriptions from tarball: {self._input_tarball}")
        n = 0
        skipped = 0
        self._transcripts_dict = {}
        self._transcripts = []
        csv.register_dialect("tsv", delimiter="\t", quoting=csv.QUOTE_ALL)

        def add(_fname, _txt):
            if not self._validate(txt):
                return False
            last, last_two = extract_last_two_number(_fname)
            self._transcripts.append(txt)
            if last not in self._transcripts_dict:
                self._transcripts_dict[last] = dict()
            if last_two not in self._transcripts_dict[last]:
                self._transcripts_dict[last][last_two] = dict()
            self._transcripts_dict[last][last_two][fname] = txt
            return True

        if tf.io.gfile.isdir(self._input_tarball):
            filename = os.path.join(self._input_tarball, f"{self._language}/{self._extraction}.tsv")
            with tf.io.gfile.GFile(filename) as file:
                fp = csv.reader(file, "tsv")
                for idx, l in enumerate(fp):
                    if idx == 0:
                        continue
                    # common_voice_en_699711.mp3
                    fname = l[1]
                    txt = l[2].strip()
                    if add(fname, txt):
                        n += 1
                    else:
                        skipped += 1
        else:
            with self.open_tarball("tar") as tar:
                for tarinfo in tar:
                    if tarinfo.name.endswith(f"{self._extraction}.tsv"):
                        f = tar.extractfile(tarinfo)
                        str_io = StringIO(f.read().decode("utf-8"))
                        f.close()
                        fp = csv.reader(str_io, "tsv")
                        for idx, l in enumerate(fp):
                            if idx == 0:
                                continue
                            # common_voice_en_699711.mp3
                            fname = l[1]
                            txt = l[2].strip()
                            if add(fname, txt):
                                n += 1
                            else:
                                skipped += 1
                        break
        logging.info("Total {} utterances, {} skipped.".format(
            len(self._transcripts), skipped))

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
            if tf.io.gfile.isdir(self._input_tarball):
                for last in self._transcripts_dict:
                    for last_two in self._transcripts_dict[last]:
                        for fname, txt in self._transcripts_dict[last][last_two].items():
                            n += 1
                            if total_shards > 1:
                                if n < range_begin:
                                    continue
                                if n >= range_end:
                                    break
                            audio = self.extract_audio_feature(
                                file=os.path.join(self._input_tarball, f"{self._language}/clips/{fname}"),
                                mode="mp3")
                            if audio is None:
                                logging.info("Detected 1 nan/inf audio feature. SKIP...")
                                continue
                            data_sample = self._pack_example_as_dict(
                                audio=audio, transcript=txt, src_lang=self._language.split("-")[0])
                            if map_func is None:
                                yield data_sample
                            else:
                                yield map_func(data_sample)
            else:
                with self.open_tarball("tar") as tar:
                    for tarinfo in tar:
                        if not tarinfo.isreg():
                            continue
                        if not tarinfo.name.endswith(".mp3"):
                            continue
                        this_trans = get_transcription(tarinfo.name, self._transcripts_dict)
                        if not this_trans:
                            continue
                        n += 1
                        if total_shards > 1:
                            if n < range_begin:
                                continue
                            if n >= range_end:
                                break
                        f = tar.extractfile(tarinfo)
                        audio = self.extract_audio_feature(fileobj=f, mode="mp3")
                        f.close()
                        if audio is None:
                            logging.info("Detected 1 nan/inf audio feature. SKIP...")
                            continue
                        data_sample = self._pack_example_as_dict(
                            audio=audio, transcript=this_trans, src_lang=self._language)
                        if map_func is None:
                            yield data_sample
                        else:
                            yield map_func(data_sample)

        return gen
