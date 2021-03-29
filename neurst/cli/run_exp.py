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
from neurst.data.datasets import Dataset, build_dataset
from neurst.exps import BaseExperiment, build_exp
from neurst.layers.quantization import QuantLayer
from neurst.models import BaseModel
from neurst.tasks import Task, build_task
from neurst.training import training_utils
from neurst.utils.configurable import ModelConfigs, deep_merge_dict, load_from_config_path, yaml_load_checking
from neurst.utils.hparams_sets import get_hyper_parameters
from neurst.utils.misc import flatten_string_list

FLAG_LIST = [
    flags_core.Flag("distribution_strategy", dtype=flags_core.Flag.TYPE.STRING, default="mirrored",
                    help="The distribution strategy."),
    flags_core.Flag("dtype", dtype=flags_core.Flag.TYPE.STRING, default="float16",
                    help="The computation type of the whole model."),
    flags_core.Flag("enable_check_numerics", dtype=flags_core.Flag.TYPE.BOOLEAN, default=None,
                    help="Whether to open the tf.debugging.enable_check_numerics. "
                         "Note that this may lower down the training speed."),
    flags_core.Flag("enable_xla", dtype=flags_core.Flag.TYPE.BOOLEAN, default=None,
                    help="Whether to enable XLA for training."),
    flags_core.Flag("hparams_set", dtype=flags_core.Flag.TYPE.STRING,
                    help="A string indicating a set of pre-defined hyper-parameters, "
                         "e.g. transformer_base, transformer_big or transformer_768_16e_3d."),
    flags_core.Flag("model_dir", dtype=flags_core.Flag.TYPE.STRING,
                    help="The path to the checkpoint for saving and loading."),
    flags_core.Flag("enable_quant", dtype=flags_core.Flag.TYPE.BOOLEAN, default=False,
                    help="Whether to enable quantization for finetuning."),
    flags_core.Flag("quant_params", dtype=flags_core.Flag.TYPE.STRING,
                    help="A dict of parameters for quantization."),
    flags_core.ModuleFlag(BaseExperiment.REGISTRY_NAME, help="The program."),
    flags_core.ModuleFlag(Task.REGISTRY_NAME, help="The binding task."),
    flags_core.ModuleFlag(BaseModel.REGISTRY_NAME, help="The model."),
    flags_core.ModuleFlag(Dataset.REGISTRY_NAME, help="The dataset."),
]


def _pre_load_args(args):
    cfg_file_args = yaml_load_checking(load_from_config_path(
        flatten_string_list(getattr(args, flags_core.DEFAULT_CONFIG_FLAG.name))))
    model_dirs = flatten_string_list(args.model_dir or cfg_file_args.get("model_dir", None))
    hparams_set = args.hparams_set
    if hparams_set is None:
        hparams_set = cfg_file_args.get("hparams_set", None)
    predefined_parameters = get_hyper_parameters(hparams_set)
    formatted_parameters = {}
    if "model.class" in predefined_parameters:
        formatted_parameters["model.class"] = predefined_parameters.pop("model.class")
    if "model" in predefined_parameters:
        formatted_parameters["model"] = predefined_parameters.pop("model")
    if "model.params" in predefined_parameters:
        formatted_parameters["model.params"] = predefined_parameters.pop("model.params")
    if len(predefined_parameters) > 0:
        formatted_parameters["entry.params"] = predefined_parameters

    try:
        model_cfgs = ModelConfigs.load(model_dirs[0])
        return deep_merge_dict(deep_merge_dict(
            model_cfgs, formatted_parameters), cfg_file_args)
    except Exception:
        return deep_merge_dict(formatted_parameters, cfg_file_args)


def run_experiment(args, remaining_argv):
    strategy = training_utils.handle_distribution_strategy(args["distribution_strategy"])
    flags_core.verbose_flags(FLAG_LIST, args, remaining_argv)
    training_utils.startup_env(
        dtype=args["dtype"],
        enable_check_numerics=args["enable_check_numerics"],
        enable_xla=args["enable_xla"])

    # initialize parameters for quantization.
    if args.get("quant_params", None) is None:
        args["quant_params"] = {}
    QuantLayer.global_init(args["enable_quant"], **args["quant_params"])

    # create exps: trainer, evaluator or ...
    with training_utils.get_strategy_scope(strategy):
        task = build_task(args)
        custom_dataset = build_dataset(args)
        try:
            model = task.build_model(args)
            training_utils.validate_unique_varname(model.weights)
        except AttributeError:
            model = None
        entry = build_exp(args,
                          strategy=strategy,
                          model=model,
                          task=task,
                          model_dir=args["model_dir"],
                          custom_dataset=custom_dataset)
    entry.run()


def _main(_):
    # define and parse program flags
    arg_parser = flags_core.define_flags(FLAG_LIST)
    args, remaining_argv = flags_core.intelligent_parse_flags(FLAG_LIST, arg_parser, _pre_load_args)
    args, remaining_argv = flags_core.extend_define_and_parse(
        BaseExperiment.REGISTRY_NAME, args, remaining_argv)
    if args["entry.class"] is None:
        raise ValueError("Must provide entry/entry.class.")
    run_experiment(args, remaining_argv)


def cli_main():
    logging.set_verbosity(logging.INFO)
    app.run(_main, argv=["pseudo.py"])


if __name__ == "__main__":
    cli_main()
