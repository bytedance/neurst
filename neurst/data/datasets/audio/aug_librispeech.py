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

from absl import logging

from neurst.data.datasets import register_dataset
from neurst.data.datasets.audio.audio_dataset import RawAudioDataset


@register_dataset("AugmentedLibriSpeech")
class AugLibriSpeech(RawAudioDataset):
    """
    Augmented LibriSpeech is a small EN->FR dataset which
    was originally started from the LibriSpeech corpus.
    The English utterances were automatically aligned to
    the e-books in French and 236 hours of English speech
    aligned to French translations at utterance level were
    finally extracted. It has been widely used in previous studies.
    As such, we use the clean 100-hour portion plus the augmented
    machine translation from Google Translate as the training data.

    Homepage: https://github.com/alicank/Translation-Augmented-LibriSpeech-Corpus
    The raw dataset contains 3 files:
        - train_100h.zip
        - dev.zip
        - test.zip
    """

    def __init__(self, args):
        super(AugLibriSpeech, self).__init__(args)
        self._transc_transla_list = None

    def load_transcripts(self):
        """ Loads transcripts and translations. """
        if self._transc_transla_list is not None:
            return
        logging.info(f"Loading transcriptions from tarball: {self._input_tarball}")
        with self.open_tarball("zip") as fp:
            for n in fp.namelist():
                if n.endswith("alignments.meta"):
                    mapping = fp.read(n).decode("utf-8")
                    mapping_id = [line.strip().split()[4]
                                  for line in mapping.split("\n")[1:] if line.strip()]
                elif n.endswith(".en"):
                    transcript = fp.read(n).decode("utf-8")
                    transcript = transcript.strip().split("\n")
                elif n.endswith(".fr") and "gtrans" not in n:
                    translation = fp.read(n).decode("utf-8")
                    translation = translation.strip().split("\n")
                elif "gtrans" in n:
                    g_translation = fp.read(n).decode("utf-8")
                    g_translation = g_translation.strip().split("\n")

        trans = [(idx, tc.strip().lower(), ts.strip(), gts.strip()) for idx, tc, ts, gts
                 in zip(mapping_id, transcript, translation, g_translation)
                 if self._validate(tc) and self._validate(ts) and self._validate(gts)]
        # [uttid, transcript, translation, google-translation]
        self._transc_transla_list = trans
        self._transcripts = [x[1] for x in trans]
        self._translations = [x[2] for x in trans]
        if "train" in os.path.basename(self._input_tarball):
            self._transcripts.extend([x[1] for x in trans])
            self._translations.extend(x[3] for x in trans)
        logging.info("Total {} utterances.".format(len(self._transcripts)))

    def build_iterator(self, map_func=None, shard_id=0, total_shards=1):
        """ Returns the iterator of the dataset.

        Args:
            map_func: A function mapping a dataset element to another dataset element.
        """

        def gen():
            if self._transc_transla_list is None:
                self.load_transcripts()
            uttid2wavname = dict()
            with self.open_tarball("zip") as fp:
                for name in fp.namelist():
                    if name.endswith(".wav"):
                        uttid2wavname[name.split("/")[-1].split(".")[0]] = name
                n = 0
                for uttid, transcript, transla, g_transla in self._transc_transla_list:
                    iter_list = [transla]
                    if "train" in os.path.basename(self._input_tarball):
                        iter_list.append(g_transla)
                    for translation in iter_list:
                        n += 1
                        if total_shards > 1:
                            if n % total_shards != shard_id:
                                continue
                        binary_data = fp.read(uttid2wavname[uttid])
                        b = io.BytesIO(binary_data)
                        audio = self.extract_audio_feature(fileobj=b, mode="wav")
                        b.close()
                        if audio is None:
                            logging.info("Detected 1 nan/inf audio feature. SKIP...")
                            continue
                        data_sample = self._pack_example_as_dict(
                            audio=audio, transcript=transcript, translation=translation,
                            src_lang=self.LANGUAGES.EN, trg_lang=self.LANGUAGES.FR)
                        if map_func is None:
                            yield data_sample
                        else:
                            yield map_func(data_sample)

        return gen
