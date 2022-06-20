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
from neurst.data.data_pipelines.data_pipeline import lowercase_and_remove_punctuations
from neurst.data.text import Tokenizer, build_tokenizer

FLAG_LIST = [
    flags_core.Flag("input", dtype=flags_core.Flag.TYPE.STRING, default=None,
                    help="The path to the input text file."),
    flags_core.Flag("output", dtype=flags_core.Flag.TYPE.STRING, default=None,
                    help="The path to the output text file."),
    flags_core.Flag("lowercase", dtype=flags_core.Flag.TYPE.BOOLEAN, default=None,
                    help="Whether to lowercase."),
    flags_core.Flag("remove_punctuation", dtype=flags_core.Flag.TYPE.BOOLEAN, default=None,
                    help="Whether to remove the punctuations."),
    flags_core.Flag("language", dtype=flags_core.Flag.TYPE.BOOLEAN, default="en",
                    help="The text language."),
    flags_core.ModuleFlag(Tokenizer.REGISTRY_NAME, help="The tokenizer."),
]


def _main(_):
    arg_parser = flags_core.define_flags(FLAG_LIST, with_config_file=False)
    args, remaining_argv = flags_core.intelligent_parse_flags(FLAG_LIST, arg_parser)
    flags_core.verbose_flags(FLAG_LIST, args, remaining_argv)

    tokenizer = build_tokenizer(args)
    with tf.io.gfile.GFile(args["input"]) as fp:
        with tf.io.gfile.GFile(args["output"], "w") as fw:
            for line in fp:
                line = lowercase_and_remove_punctuations(args["language"], line.strip(),
                                                         args["lowercase"], args["remove_punctuation"])
                fw.write(tokenizer.tokenize(line, return_str=True) + "\n")
                if tokenizer is None:
                    fw.write(line + "\n")
                else:
                    fw.write(tokenizer.tokenize(line, return_str=True) + "\n")


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app.run(_main, argv=["pseudo.py"])
