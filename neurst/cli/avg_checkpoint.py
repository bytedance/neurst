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
import os

import tensorflow as tf
from absl import app, logging

import neurst.utils.flags_core as flags_core
from neurst.utils.compat import wrapper_var_name
from neurst.utils.configurable import ModelConfigs
from neurst.utils.flags_core import Flag
from neurst.utils.misc import flatten_string_list

FLAG_LIST = [
    flags_core.Flag("checkpoints", dtype=Flag.TYPE.STRING, default=None, multiple=True,
                    help="A list or comma-separated string of checkpoints to be averaged. "
                         "The averaged checkpoint will be saved to `output_path`."),
    flags_core.Flag("output_path", dtype=flags_core.Flag.TYPE.STRING,
                    help="The path to the averaged checkpoint."),
]


def checkpoint_exists(path):
    return (tf.io.gfile.exists(path) or tf.io.gfile.exists(path + ".meta")
            or tf.io.gfile.exists(path + ".index"))


def checkpoint_list_checking(path_list):
    if path_list:
        new_path_list = []
        for path in path_list:
            if checkpoint_exists(path):
                new_path_list.append(path)
        return new_path_list
    return []


def average_checkpoints(checkpoints, output_path):
    assert checkpoints
    # Get the checkpoints list from flags and run some basic checks.
    checkpoints = flatten_string_list(checkpoints)
    checkpoints = [c for c in checkpoints if c]
    if not checkpoints:
        raise ValueError("No checkpoints provided for averaging.")
    model_config_yml_path = None
    for c in checkpoints:
        if model_config_yml_path:
            break
        if tf.io.gfile.exists(os.path.join(c, ModelConfigs.MODEL_CONFIG_YAML_FILENAME)):
            model_config_yml_path = os.path.join(c, ModelConfigs.MODEL_CONFIG_YAML_FILENAME)
    all_checkpoint_paths = []
    for c in checkpoints:
        if tf.io.gfile.isdir(c):
            checkpoint_states = tf.train.get_checkpoint_state(c)
            all_checkpoint_paths.extend(checkpoint_list_checking(checkpoint_states.all_model_checkpoint_paths))
        else:
            all_checkpoint_paths.append(c)
    var_values = {}
    var_cnts = {}
    var_name_shape_list = tf.train.list_variables(all_checkpoint_paths[0])
    for ckpt in all_checkpoint_paths:
        logging.info("loading from {}".format(ckpt))
        for var_name, _ in var_name_shape_list:
            if var_name.startswith("_") or var_name.startswith("save_counter"):
                logging.info("ignore {}...".format(var_name))
                continue
            var = tf.train.load_variable(ckpt, var_name)
            fine_name = wrapper_var_name(var_name)
            if fine_name in var_values:
                var_cnts[fine_name] += 1.
                var_values[fine_name] = var * 1. / var_cnts[fine_name] + var_values[fine_name] * (
                    var_cnts[fine_name] - 1.) / var_cnts[fine_name]
            else:
                var_cnts[fine_name] = 1.
                var_values[fine_name] = var
    tf_vars = dict()
    logging.info("Averaged variables: ")
    for var_name in var_values.keys():
        fine_name = wrapper_var_name(var_name)
        assert var_cnts[fine_name] == len(all_checkpoint_paths)
        logging.info(fine_name)
        tf_vars[fine_name] = tf.Variable(
            initial_value=var_values[fine_name],
            trainable=True,
            name=fine_name,
            dtype=str(var_values[fine_name].dtype))
    ckpt_saver = tf.train.Checkpoint(**tf_vars)
    ckpt_saver.save(os.path.join(output_path, "ckpt"))
    tf.io.gfile.copy(model_config_yml_path,
                     os.path.join(output_path, ModelConfigs.MODEL_CONFIG_YAML_FILENAME),
                     overwrite=True)


def _main(_):
    arg_parser = flags_core.define_flags(FLAG_LIST, with_config_file=False)
    args, remaining_argv = flags_core.intelligent_parse_flags(FLAG_LIST, arg_parser)
    flags_core.verbose_flags(FLAG_LIST, args, remaining_argv)
    average_checkpoints(
        checkpoints=flatten_string_list(args["checkpoints"]),
        output_path=args["output_path"])


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app.run(_main, argv=["pseudo.py"])
