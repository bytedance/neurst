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
import numpy
from absl import app, logging

import neurst.utils.flags_core as flags_core
from neurst.data.audio import FeatureExtractor, build_feature_extractor
from neurst.data.audio.float_identity import FloatIdentity
from neurst.data.datasets import Dataset, build_dataset
from neurst.data.datasets.audio.audio_dataset import AudioTripleTFRecordDataset

FLAG_LIST = [
    flags_core.ModuleFlag(Dataset.REGISTRY_NAME, help="The audio TFRecord dataset.",
                          default=AudioTripleTFRecordDataset.__name__),
    flags_core.ModuleFlag(FeatureExtractor.REGISTRY_NAME,
                          help="The feature extractor already applied on the audio.",
                          default=FloatIdentity.__name__),
]

_DISPLAY_PERCENTS = [0.1, 0.3, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999]


class BigCounter(object):

    def __init__(self, base=1000):
        self._base = base
        self._values = dict()
        self.min_value = 1000000
        self.max_value = 0
        self.total = 0

    def count(self, item):
        k1 = item // self._base
        if k1 not in self._values:
            self._values[k1] = dict()
        k2 = item % self._base
        if k2 in self._values[k1]:
            self._values[k1][k2] += 1
        else:
            self._values[k1][k2] = 1
        if item < self.min_value:
            self.min_value = item
        if item > self.max_value:
            self.max_value = item
        self.total += 1

    def __getitem__(self, item):
        k1 = item // self._base
        if k1 not in self._values:
            return None
        k2 = item % self._base
        if k2 not in self._values[k1]:
            return None
        return self._values[k1][k2]


def get_element_size(element):
    if isinstance(element[0], bytes):
        return len(element[0].decode("utf-8").strip().split())
    return len(element)


def freq_percent(counter):
    cnt = 0.
    current_idx = 0
    margin_values = []
    for val in range(counter.min_value, counter.max_value + 1):
        this_cnt = counter[val]
        if this_cnt:
            cnt += this_cnt
            percent = cnt / counter.total
            while (current_idx < len(_DISPLAY_PERCENTS)
                   and percent > _DISPLAY_PERCENTS[current_idx]):
                margin_values.append(val)
                current_idx += 1
        if current_idx == len(_DISPLAY_PERCENTS):
            break
    return margin_values


def main(dataset, feature_extractor=None):
    audio_counter = BigCounter()
    transcript_counter = BigCounter()
    translation_counter = BigCounter()
    has_translation = False
    total_seconds = 0.
    for idx, sample in enumerate(dataset.build().as_numpy_iterator()):
        if idx == 0:
            if "translation" in sample:
                has_translation = True
        audio_length = get_element_size(sample["audio"])
        audio_length //= feature_extractor.feature_dim
        total_seconds += feature_extractor.seconds(
            numpy.reshape(sample["audio"], [-1, feature_extractor.feature_dim]))
        audio_counter.count(audio_length)
        transcript_counter.count(get_element_size(sample["transcript"]))
        if has_translation:
            translation_counter.count(get_element_size(sample["translation"]))
    logging.info("Total %d samples, %.2f hours", audio_counter.total, total_seconds / 3600.)
    logging.info(f"Max audio size: {audio_counter.max_value}, min audio size: {audio_counter.min_value}")
    logging.info("Audio feature size distribution: ")
    for percent, margin in zip(_DISPLAY_PERCENTS, freq_percent(audio_counter)):
        logging.info(f"  {percent} samples <= {margin}")
    logging.info("")
    logging.info("Transcript distribution: ")
    for percent, margin in zip(_DISPLAY_PERCENTS, freq_percent(transcript_counter)):
        logging.info(f"  {percent} samples <= {margin}")
    if has_translation:
        logging.info("")
        logging.info("Translation distribution: ")
        for percent, margin in zip(_DISPLAY_PERCENTS, freq_percent(translation_counter)):
            logging.info(f"  {percent} samples <= {margin}")


def _main(_):
    # define and parse program flags
    arg_parser = flags_core.define_flags(FLAG_LIST, with_config_file=True)
    args, remaining_argv = flags_core.intelligent_parse_flags(FLAG_LIST, arg_parser)
    flags_core.verbose_flags(FLAG_LIST, args, remaining_argv)
    dataset = build_dataset(args)
    feature_extractor = build_feature_extractor(args)
    if dataset is None:
        raise ValueError("dataset must be provided.")
    main(dataset, feature_extractor)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app.run(_main, argv=["pseudo.py"])
