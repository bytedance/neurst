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
import tensorflow as tf
from absl import app, logging

import neurst.utils.flags_core as flags_core
from neurst.data.datasets import Dataset, build_dataset
from neurst.data.datasets.audio.audio_dataset import RawAudioDataset

FLAG_LIST = [
    flags_core.Flag("output_transcript_file", dtype=flags_core.Flag.TYPE.STRING,
                    required=True, help="The path to save transcriptions."),
    flags_core.Flag("output_translation_file", dtype=flags_core.Flag.TYPE.STRING,
                    default=None, help="The path to save transcriptions."),
    flags_core.ModuleFlag(Dataset.REGISTRY_NAME, help="The raw dataset."),
]


def main(dataset, output_transcript_file, output_translation_file=None):
    assert isinstance(dataset, RawAudioDataset)
    transcripts = dataset.transcripts
    translations = dataset.translations
    assert transcripts, "Fail to extract transcripts."
    with tf.io.gfile.GFile(output_transcript_file, "w") as fw:
        fw.write("\n".join(transcripts) + "\n")
    if translations and output_translation_file:
        with tf.io.gfile.GFile(output_translation_file, "w") as fw:
            fw.write("\n".join(translations) + "\n")


def _main(_):
    # define and parse program flags
    arg_parser = flags_core.define_flags(FLAG_LIST, with_config_file=True)
    args, remaining_argv = flags_core.intelligent_parse_flags(FLAG_LIST, arg_parser)
    flags_core.verbose_flags(FLAG_LIST, args, remaining_argv)
    dataset = build_dataset(args)
    if dataset is None:
        raise ValueError("dataset must be provided.")
    main(dataset=dataset,
         output_transcript_file=args["output_transcript_file"],
         output_translation_file=args["output_translation_file"])


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app.run(_main, argv=["pseudo.py"])
