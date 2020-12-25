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
from neurst.utils.converters import Converter, build_converter

FLAG_LIST = [
    flags_core.Flag("from", dtype=flags_core.Flag.TYPE.STRING, default=None,
                    required=True, help="The path to pretrained model directory "
                                        "or a key indicating the publicly available model name."),
    flags_core.Flag("to", dtype=flags_core.Flag.TYPE.STRING, default=None,
                    required=True, help="The path to save the converted checkpoint."),
    flags_core.Flag("model_name", dtype=flags_core.Flag.TYPE.STRING, default=None,
                    required=True, help="The name of pretrained model, e.g. google_bert."),
]


def convert(converter: Converter, from_path, to_path):
    assert converter is not None
    assert from_path
    assert to_path
    converter.convert(from_path, to_path)


def _main(_):
    arg_parser = flags_core.define_flags(FLAG_LIST, with_config_file=False)
    args, remaining_argv = flags_core.parse_flags(FLAG_LIST, arg_parser)
    flags_core.verbose_flags(FLAG_LIST, args, remaining_argv)
    converter = build_converter(args["model_name"])
    convert(converter, args["from"], args["to"])


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app.run(_main, argv=["pseudo.py"])
