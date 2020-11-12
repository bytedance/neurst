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
import re
import sys

import tensorflow as tf

from neurst.models.model_utils import _summary_model_variables
from neurst.utils.compat import wrapper_var_name


def cli_main():
    structured = False
    if "--structured" in sys.argv:
        structured = True
        sys.argv.remove("--structured")
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and (sys.argv[1] in ["help", "--help", "-h"])):
        print("Usage: ")
        print("    >> python3 -m neurst.cli.inspect_checkpoint modeldir_or_checkpoint (--structured)")
        print("           List all variables and their shapes.")
        print()
        print("    >> python3 -m neurst.cli.inspect_checkpoint model_dir/checkpoint regular_expr")
        print("           List the variables and their shapes if the name matches the `regular_expr`.")
        print()
        print("    >> python3 -m neurst.cli.inspect_checkpoint model_dir/checkpoint var_name")
        print("           Print the variable tensor.")
        exit()

    model_dir = sys.argv[1]
    var_name = None
    if len(sys.argv) == 3:
        var_name = sys.argv[2]
    latest_ckpt_path = tf.train.latest_checkpoint(model_dir)
    if not latest_ckpt_path:
        latest_ckpt_path = model_dir
    try:
        varname_shape_list = tf.train.list_variables(latest_ckpt_path)
    except (tf.errors.NotFoundError, ValueError, tf.errors.DataLossError):
        print(f"ERROR: fail to load checkpoint from {model_dir}")
        exit()
    clean_varname2ckpt_varname = {
        wrapper_var_name(varname): (varname, shape)
        for varname, shape in varname_shape_list}
    specify_varname = False
    if var_name is not None:
        clean_varname2ckpt_varname = {
            wrapper_var_name(varname): (varname, shape)
            for varname, shape in varname_shape_list}
        if var_name in clean_varname2ckpt_varname:
            specify_varname = True
            print(f"Variable name: {var_name}")
            print(f"Tensor Shape: {str(clean_varname2ckpt_varname[var_name][1])}")
            print("Tensor Value: ")
            print(tf.train.load_variable(latest_ckpt_path, clean_varname2ckpt_varname[var_name][0]))
    if not specify_varname:
        if not structured:
            if var_name is None:
                print("\tvariable name \t shape")
            else:
                print(f"\tvariable name ({var_name}) \t shape")
        print_varname_shape_list = []
        for clean_varname, (varname, shape) in clean_varname2ckpt_varname.items():
            if varname in ["_CHECKPOINTABLE_OBJECT_GRAPH"]:
                continue
            if var_name is None or re.search(var_name, clean_varname):
                print_varname_shape_list.append((clean_varname, shape))
        if structured:
            _summary_model_variables(print_varname_shape_list, print)
        else:
            for clean_varname, shape in print_varname_shape_list:
                print(clean_varname + "\t" + str(shape))


if __name__ == "__main__":
    cli_main()
