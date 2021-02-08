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
import json
import sys

from neurst.utils.flags_core import ModuleFlag
from neurst.utils.hparams_sets import get_hyper_parameters
from neurst.utils.registry import REGISTRIES


def cli_main():
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and (sys.argv[1] in ["help", "--help", "-h"])):
        print("Usage: ")
        print("    >> python3 -m neurst.cli.view_registry registry_name")
        print("           Show registered classes and their aliases.")
        print()
        print("    >> python3 -m neurst.cli.view_registry registry_name class_name")
        print("           Show detailed parameters of the class.")
        print()
        print("All registry names: ")
        for k in REGISTRIES["tf"]:
            print(f"    - {k}")
        exit()

    registry_name = sys.argv[1].lower()
    if registry_name not in REGISTRIES["tf"]:
        print(f"Unknown registry name: {registry_name}")
    elif len(sys.argv) == 2:
        print(f"All registered {registry_name}(s): ")
        clsname2alias = {}
        for name, cls in REGISTRIES["tf"][registry_name].items():
            clsname = cls.__name__
            if clsname not in clsname2alias:
                clsname2alias[clsname] = []
            clsname2alias[clsname].append(name)
        if registry_name == "hparams_set":
            for k in clsname2alias:
                print(f"    - {k}")
        else:
            print("    |  Class  |  Aliases  |")
            for k, v in clsname2alias.items():
                print("    |  {}  |  {}  |".format(k, ", ".join(v)))
    elif len(sys.argv) == 3:
        detail_name = sys.argv[2]
        if registry_name == "hparams_set":
            hparams = get_hyper_parameters(detail_name)
            if len(hparams) == 0:
                print(f"Unknown hparams_set: {detail_name}")
            else:
                print(f"Pre-defined hyperparameters set of `{detail_name}`: ")
                print(json.dumps(get_hyper_parameters(detail_name), indent=4))
        elif detail_name not in REGISTRIES["tf"][registry_name]:
            print(f"Unknown class: {detail_name} under `{registry_name}`")
        else:
            if hasattr(REGISTRIES["tf"][registry_name][detail_name], "class_or_method_args"):
                flags = []
                module_flags = []
                for f in REGISTRIES["tf"][registry_name][detail_name].class_or_method_args():
                    if isinstance(f, ModuleFlag):
                        module_flags.append(f)
                    else:
                        flags.append(f)
                if len(flags) > 0:
                    print(f"Flags for {detail_name}:")
                    print("    |  flag  |  type  |  default  |  help  |")
                    for f in flags:
                        print(f"    |  {f.name}  |  {str(f.dtype)}  |  {f.default}  |  {f.help}  |")
                if len(module_flags) > 0:
                    print(f"Dependent modules for {detail_name}: ")
                    print("    |  name  |  module  |  help  |")
                    for f in module_flags:
                        print(f"    |  {f.name}  |  {f.module_name}  |  {f.help}  |")

            else:
                print(f"No flags defined for `{detail_name}` ({registry_name})")


if __name__ == "__main__":
    cli_main()
