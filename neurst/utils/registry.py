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

from absl import logging

from neurst.utils.configurable import deep_merge_dict

REGISTRIES = {}
REGISTRIED_CLS2ALIAS = {}


def setup_registry(registry_name, base_class=None, create_fn=None,
                   verbose_creation=False, backend="tf"):
    if backend not in REGISTRIES:
        REGISTRIES[backend] = {}
        REGISTRIED_CLS2ALIAS[backend] = {}

    if registry_name not in REGISTRIES[backend]:
        REGISTRIES[backend][registry_name] = {}
        REGISTRIED_CLS2ALIAS[backend][registry_name] = {}

    from neurst.utils.flags_core import Flag, ModuleFlag

    def _verbose_creation(cls_, args, *extra_args, **kwargs):
        if not verbose_creation:
            return
        logging.info("Creating {}: {}".format(registry_name, cls_))
        if args is not None and len(args) > 0:
            logging.info("  ({}) arguments: ".format(registry_name))
            for k, v in args.items():
                if isinstance(v, dict):
                    logging.info("    {}:".format(k))
                    for kk, vv in v.items():
                        if isinstance(vv, list) and len(vv) > 10:
                            logging.info("      {}: {}".format(kk, vv[:10] + ["......"]))
                        else:
                            logging.info("      {}: {}".format(kk, vv))
                else:
                    logging.info("    {}: {}".format(k, v))
        if len(extra_args) > 0:
            logging.info("  ({}) extra args: ".format(registry_name))
            for x in extra_args:
                logging.info("    - {}".format(x))
        if len(kwargs) > 0:
            logging.info("  ({}) extra k-v args: ".format(registry_name))
            for k, v in kwargs.items():
                logging.info("    {}: {}".format(k, v))

    def build_x(args, *extra_args, **kwargs):
        params_ = {}
        if isinstance(args, dict):
            cls_ = (args.get("class", None)
                    or args.get(f"{registry_name}.class", None)
                    or args.get(f"{registry_name}", None))
            params_ = (args.get("params", None)
                       or args.get("{}.params".format(registry_name), {})) or {}
        else:
            cls_ = args
        if cls_ is None:
            return None
        if isinstance(cls_, str):
            if cls_.lower() == "none":
                return None
            if cls_ not in REGISTRIES[backend][registry_name]:
                raise ValueError("Not registered class name: {}.".format(cls_))
            cls_ = REGISTRIES[backend][registry_name][cls_]
            builder = cls_
        elif callable(cls_):
            builder = cls_
        else:
            raise ValueError("Not supported type: {} for builder.".format(type(cls_)))
        if create_fn is not None:
            assert hasattr(builder, create_fn), "{} has no {} for creation.".format(cls_, create_fn)
            builder = getattr(builder, create_fn)
        if kwargs is None:
            kwargs = {}
        assert isinstance(params_, dict), f"Not supported type: {type(params_)} for params"
        if hasattr(cls_, "class_or_method_args"):
            for f in cls_.class_or_method_args():
                if isinstance(f, Flag):
                    if f.name in kwargs:
                        params_[f.name] = kwargs.pop(f.name)
                    elif f.name not in params_:
                        params_[f.name] = f.default
                elif isinstance(f, ModuleFlag) and f.cls_key not in params_:
                    params_[f.cls_key] = f.default
                if isinstance(f, ModuleFlag) and f.params_key not in params_:
                    params_[f.params_key] = {}
            _verbose_creation(cls_, params_, *extra_args, **kwargs)
            return builder(params_, *extra_args, **kwargs)
        params_ = deep_merge_dict(params_, kwargs, merge_only_exist=False)
        _verbose_creation(cls_, {}, *extra_args, **params_)
        return builder(*extra_args, **params_)

    def register_x(name):

        def register_x_cls(cls_, short_name=None):
            if base_class is not None and not issubclass(cls_, base_class):
                raise ValueError('{} must extend {}'.format(cls_.__name__, base_class.__name__))
            names = set()
            if short_name:
                for n in short_name:
                    names.add(n)
            names.add(cls_.__name__)
            names.add(cls_.__name__.lower())
            names.add("_".join(re.sub("([A-Z])", r' \1', cls_.__name__).lower().strip().split()))
            for n in names:
                if n in REGISTRIES[backend][registry_name]:
                    if REGISTRIES[backend][registry_name][n] != cls_:
                        raise ValueError('Cannot register duplicate {} (under {})'.format(n, registry_name))
                else:
                    REGISTRIES[backend][registry_name][n] = cls_
            REGISTRIED_CLS2ALIAS[backend][registry_name][cls_.__name__] = names
            return cls_

        if isinstance(name, str):
            return lambda c: register_x_cls(c, [name])
        elif callable(name):
            return register_x_cls(cls_=name)
        elif isinstance(name, list):
            return lambda c: register_x_cls(c, name)
        else:
            raise ValueError("Not supported type: {}".format(type(name)))

    return build_x, register_x


def get_registered_class(cls_, registry_name, backend="tf"):
    if cls_ is None:
        return None
    if isinstance(cls_, str):
        if cls_.lower() == "none":
            return None
        if cls_ not in REGISTRIES[backend][registry_name]:
            raise ValueError("Not registered class name: {}.".format(cls_))
        return REGISTRIES[backend][registry_name][cls_]
    elif callable(cls_):
        return cls_
    return None
