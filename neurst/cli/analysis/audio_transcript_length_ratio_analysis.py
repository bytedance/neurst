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
from absl import app, logging

import neurst.utils.flags_core as flags_core
from neurst.data.audio import FeatureExtractor, build_feature_extractor
from neurst.data.audio.float_identity import FloatIdentity
from neurst.data.datasets import Dataset, build_dataset
from neurst.data.datasets.audio.audio_dataset import AudioTFRecordDataset

FLAG_LIST = [
    flags_core.ModuleFlag(Dataset.REGISTRY_NAME, help="The audio TFRecord dataset.",
                          default=AudioTFRecordDataset.__name__),
    flags_core.ModuleFlag(FeatureExtractor.REGISTRY_NAME,
                          help="The feature extractor already applied on the audio.",
                          default=FloatIdentity.__name__),
]

_DISPLAY_PERCENTS = [0.1, 0.3, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999]


def get_element_size(element):
    if isinstance(element[0], bytes):
        return len(element[0].decode("utf-8").strip().split())
    return len(element)


def freq_percent(counter, min_val, max_val, total):
    cnt = 0.
    current_idx = 0
    margin_values = []
    for val in range(int(min_val * 10), int(max_val * 10) + 1):
        this_cnt = counter.get(round(val / 10., 1), None)
        if this_cnt:
            cnt += this_cnt
            percent = cnt / total
            while (current_idx < len(_DISPLAY_PERCENTS)
                   and percent > _DISPLAY_PERCENTS[current_idx]):
                margin_values.append(val)
                current_idx += 1
        if current_idx == len(_DISPLAY_PERCENTS):
            break
    return margin_values


def main(dataset, feature_extractor=None):
    audio_counter = dict()
    max_val = 0
    min_val = 99999
    total = 0
    for idx, sample in enumerate(dataset.build().as_numpy_iterator()):
        total += 1
        audio_length = get_element_size(sample["audio"])
        audio_length //= feature_extractor.feature_dim
        transcript_length = get_element_size(sample["transcript"])
        ratio = round(audio_length * 1. / transcript_length, 1)
        if ratio not in audio_counter:
            audio_counter[ratio] = 0
        audio_counter[ratio] += 1
        if ratio > max_val:
            max_val = ratio
        elif ratio < min_val:
            min_val = ratio
    logging.info(f"Max audio transcript ratio: {max_val}, min ratio: {min_val}")
    logging.info("Audio feature size distribution: ")
    for percent, margin in zip(_DISPLAY_PERCENTS, freq_percent(audio_counter, min_val, max_val, total)):
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
