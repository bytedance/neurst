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
from neurst.data.datasets.audio.audio_dataset import AudioTripleTFRecordDataset
from neurst.data.datasets.parallel_text_dataset import ParallelTextDataset
from neurst.exps import build_exp
from neurst.exps.sequence_generator import SequenceGenerator
from neurst.layers.search import SequenceSearch
from neurst.metrics.metric import Metric
from neurst.tasks import build_task
from neurst.training import training_utils
from neurst.utils.configurable import ModelConfigs

FLAG_LIST = [
    flags_core.Flag("distribution_strategy", dtype=flags_core.Flag.TYPE.STRING, default="mirrored",
                    help="The distribution strategy."),
    flags_core.Flag("dtype", dtype=flags_core.Flag.TYPE.STRING, default="float16",
                    help="The computation type of the whole model."),
    flags_core.Flag("enable_check_numerics", dtype=flags_core.Flag.TYPE.BOOLEAN, default=None,
                    help="Whether to open the tf.debugging.enable_check_numerics. "
                         "Note that this may lower down the training speed."),
    flags_core.Flag("asr_model_dir", dtype=flags_core.Flag.TYPE.STRING,
                    help="The path to the ASR model checkpoint."),
    flags_core.Flag("mt_model_dir", dtype=flags_core.Flag.TYPE.STRING,
                    help="The path to the MT model checkpoint."),
    flags_core.Flag("asr_output_file", dtype=flags_core.Flag.TYPE.STRING,
                    help="The path to save ASR hypothesis."),
    flags_core.Flag("mt_output_file", dtype=flags_core.Flag.TYPE.STRING,
                    help="The path to save MT hypothesis."),
    flags_core.Flag("batch_size", dtype=flags_core.Flag.TYPE.INTEGER, default=32,
                    help="The batch size for inference."),
    flags_core.ModuleFlag(Dataset.REGISTRY_NAME, default=AudioTripleTFRecordDataset.__name__,
                          help="The audio dataset."),
    flags_core.ModuleFlag("asr_" + SequenceSearch.REGISTRY_NAME,
                          module_name=SequenceSearch.REGISTRY_NAME,
                          default="beam_search", help="The search method for ASR."),
    flags_core.ModuleFlag("mt_" + SequenceSearch.REGISTRY_NAME,
                          module_name=SequenceSearch.REGISTRY_NAME,
                          default="beam_search", help="The search method for MT."),
    flags_core.ModuleFlag("asr_" + Metric.REGISTRY_NAME,
                          module_name=Metric.REGISTRY_NAME,
                          default="wer", help="The metric to evaluate ASR output."),
    flags_core.ModuleFlag("mt_" + Metric.REGISTRY_NAME,
                          module_name=Metric.REGISTRY_NAME,
                          default="bleu", help="The metric to evaluate MT output."),
]


def _build_task_model(strategy, model_dir, batch_size):
    with training_utils.get_strategy_scope(strategy):
        model_configs = ModelConfigs.load(model_dir)
        task = build_task(model_configs, batch_size=batch_size)
        model = task.build_model(model_configs)
        return task, model


def _main(_):
    # define and parse program flags
    arg_parser = flags_core.define_flags(FLAG_LIST)
    args, remaining_argv = flags_core.parse_flags(FLAG_LIST, arg_parser)
    flags_core.verbose_flags(FLAG_LIST, args, remaining_argv)
    strategy = training_utils.handle_distribution_strategy(args["distribution_strategy"])
    training_utils.startup_env(dtype=args["dtype"], enable_xla=False,
                               enable_check_numerics=args["enable_check_numerics"])

    asr_task, asr_model = _build_task_model(strategy, args["asr_model_dir"],
                                            batch_size=args["batch_size"])
    mt_task, mt_model = _build_task_model(strategy, args["mt_model_dir"],
                                          batch_size=args["batch_size"])
    audio_dataset = build_dataset(args)
    # ========= ASR ==========
    asr_output_file = args["asr_output_file"]
    if asr_output_file is None:
        asr_output_file = "ram://asr_output_file"
    logging.info("Creating ASR generator.")
    with training_utils.get_strategy_scope(strategy):
        asr_generator = build_exp(
            {"class": SequenceGenerator,
             "params": {
                 "output_file": asr_output_file,
                 "search_method.class": args["asr_search_method.class"],
                 "search_method.params": args["asr_search_method.params"],
             }},
            strategy=strategy,
            model=asr_model,
            task=asr_task,
            model_dir=args["asr_model_dir"],
            custom_dataset=audio_dataset)
    asr_generator.run()
    if hasattr(audio_dataset, "transcripts") and audio_dataset.transcripts is not None:
        asr_metric = asr_task.get_eval_metric(args, "asr_metric")
        with tf.io.gfile.GFile(asr_output_file, "r") as fp:
            metric_result = asr_metric([line.strip() for line in fp],
                                       audio_dataset.transcripts)
        logging.info("Evaluation Result of ASR:")
        for k, v in metric_result.items():
            logging.info("   %s=%.2f", k, v)

    logging.info("Creating MT generator.")
    mt_reference_file = "ram://mt_reference_file"
    with tf.io.gfile.GFile(mt_reference_file, "w") as fw:
        for x in audio_dataset.targets:
            fw.write(x.strip() + "\n")

    with training_utils.get_strategy_scope(strategy):
        mt_generator = build_exp(
            {"class": SequenceGenerator,
             "params": {
                 "output_file": args["mt_output_file"],
                 "search_method.class": args["mt_search_method.class"],
                 "search_method.params": args["mt_search_method.params"],
                 "metric.class": args["mt_metric.class"],
                 "metric.params": args["mt_metric.params"]
             }},
            strategy=strategy,
            model=mt_model,
            task=mt_task,
            model_dir=args["mt_model_dir"],
            custom_dataset=build_dataset({
                "class": ParallelTextDataset,
                "params": {
                    "src_file": asr_output_file,
                    "trg_file": mt_reference_file
                }}))
    mt_generator.run()


def cli_main():
    logging.set_verbosity(logging.INFO)
    app.run(_main, argv=["pseudo.py"])


if __name__ == "__main__":
    cli_main()
