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
from absl import logging

from neurst.utils.registry import REGISTRIES


def register_hparams_set(name, backend="tf"):
    registry_name = "hparams_set"
    if registry_name not in REGISTRIES[backend]:
        REGISTRIES[backend][registry_name] = {}

    def register_x_fn(fn_, short_name=None):
        names = set()
        if short_name:
            for n in short_name:
                names.add(n.lower())
        names.add(fn_.__name__)
        for n in names:
            if n in REGISTRIES[backend][registry_name]:
                if REGISTRIES[backend][registry_name][n] != fn_:
                    raise ValueError('Cannot register duplicate {} (under {})'.format(n, registry_name))
            else:
                REGISTRIES[backend][registry_name][n] = fn_

    if isinstance(name, str):
        return lambda fn: register_x_fn(fn, [name])
    elif isinstance(name, list):
        return lambda c: register_x_fn(c, name)
    else:
        raise ValueError("Not supported type: {}".format(type(name)))


def get_hyper_parameters(name, backend="tf"):
    registry_name = "hparams_set"
    if name is None:
        return {}
    if registry_name in REGISTRIES[backend] and name in REGISTRIES[backend][registry_name]:
        logging.info("matched the pre-defined hyper-parameters set: {}".format(name))
        return REGISTRIES[backend][registry_name][name]()
    for m, mc in REGISTRIES[backend]["model"].items():
        if hasattr(mc, "build_model_args_by_name"):
            p = mc.build_model_args_by_name(name)
            if p is not None:
                return p
    return {}
