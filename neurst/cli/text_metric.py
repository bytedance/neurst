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
from neurst.metrics import Metric, build_metric
from neurst.utils.misc import flatten_string_list

FLAG_LIST = [
    flags_core.Flag("hypo_file", dtype=flags_core.Flag.TYPE.STRING, default=None,
                    help="The path to hypothesis file."),
    flags_core.Flag("ref_file", dtype=flags_core.Flag.TYPE.STRING, default=None, multiple=True,
                    help="The path to reference file. "),
    flags_core.ModuleFlag(Metric.REGISTRY_NAME, help="The metric for evaluation."),
]


def evaluate(metric, hypo_file, ref_file):
    assert metric is not None
    assert hypo_file
    assert ref_file
    with tf.io.gfile.GFile(hypo_file) as fp:
        hypo = [line.strip() for line in fp]

    ref_list = []
    for one_ref_file in flatten_string_list(ref_file):
        with tf.io.gfile.GFile(one_ref_file) as fp:
            ref = [line.strip() for line in fp]
            ref_list.append(ref)

    metric_result = (metric(hypo, ref_list) if len(ref_list) > 1
                     else metric(hypo, ref_list[0]))
    for k, v in metric_result.items():
        logging.info("Evaluation result: %s=%.2f", k, v)


def _main(_):
    arg_parser = flags_core.define_flags(FLAG_LIST, with_config_file=False)
    args, remaining_argv = flags_core.intelligent_parse_flags(FLAG_LIST, arg_parser)
    flags_core.verbose_flags(FLAG_LIST, args, remaining_argv)
    metric = build_metric(args)
    evaluate(metric, args["hypo_file"], args["ref_file"])


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app.run(_main, argv=["pseudo.py"])
