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
import argparse
import copy
import importlib
import json
import os
import traceback
from collections import namedtuple

import tensorflow as tf
from absl import logging

from neurst.utils.configurable import deep_merge_dict, load_from_config_path, yaml_load_checking
from neurst.utils.misc import flatten_string_list
from neurst.utils.registry import REGISTRIES

_DEFINED_FLAGS = dict()


class Flag(object):
    TYPE = namedtuple(
        "FLAG_ARG_TYPES", "INTEGER BOOLEAN FLOAT STRING")(int, bool, float, str)

    UNIQ_SET = []

    def __init__(self,
                 name,
                 dtype,
                 required=False,
                 choices=None,
                 help="",
                 default=None,
                 multiple=False,
                 alias=None):
        """ The flags

        Args:
            name: The flag name.
            dtype: The type.
            choices: A list of acceptable values.
            default: The default value.
            help: The help text.
            multiple: Whether the flag accepts multiple arguments.
            alias: The alias name for this flag.
        """
        if name in ["class", "params"]:
            raise ValueError("Invalid flag name: {}".format(name))
        if "-" in name:
            raise ValueError("Flag name with '-' is not supported.")
        self._name = name.strip()
        self._dtype = dtype
        self._default = default
        self._help = help
        self._choices = choices
        self._multiple = multiple
        self._required = required
        self._alias = alias

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype

    @property
    def default(self):
        return self._default

    @property
    def multiple(self):
        return self._multiple

    @property
    def help(self):
        return self._help

    @property
    def alias(self):
        return self._alias

    @property
    def choices(self):
        return self._choices

    def define(self, arg_parser: argparse.ArgumentParser, default_is_none=True):
        """ Adds argument to the parser.

        Args:
            arg_parser: An ArgumentParser object.
            default_is_none: Whether the default value is None

        Returns: The parser.
        """
        try:
            idx = self.UNIQ_SET.index(self)
        except ValueError:
            self.UNIQ_SET.append(self)
        else:
            raise ValueError("Defines duplicate flag: {}, while {} is already exists.".format(
                str(self), str(self.UNIQ_SET[idx])))

        flag_names = ["--" + self.name]
        if self.alias is not None:
            flag_names.append("--" + self.alias)
        kwargs = {"type": self.dtype, "dest": self.name, "help": self.help}
        if self.dtype is bool:
            kwargs["action"] = "store_true"
            kwargs["default"] = None
            kwargs.pop("type")
        if self.multiple:
            kwargs["nargs"] = "+"
        if self.choices:
            kwargs["choices"] = self.choices
        if self.default and not default_is_none:
            kwargs["default"] = self.default
        if self._required:
            kwargs["required"] = True
        arg_parser.add_argument(*flag_names, **kwargs)
        if self.name in _DEFINED_FLAGS:
            raise ValueError(f"Defined duplicate arg key: {self.name}")
        _DEFINED_FLAGS[self.name] = self
        return arg_parser


class ModuleFlag(object):
    def __init__(self, name, module_name=None, default=None, help=""):
        """ Initializes the module flag. """
        self._name = name
        self._module_name = module_name or name
        self._help = help
        self._default = default

    @property
    def help(self):
        return self._help

    @property
    def default(self):
        return self._default

    @property
    def name(self):
        return self._name

    @property
    def module_name(self):
        return self._module_name

    @property
    def cls_key(self):
        return self.name + ".class"

    @property
    def params_key(self):
        return self.name + ".params"

    def define(self, arg_parser: argparse.ArgumentParser, backend="tf"):
        """ Adds argument to the parser.

        Args:
            arg_parser: An ArgumentParser object.
            backend: The DL backend.

        Returns: The parser.
        """
        _DEFINED_FLAGS[self.name] = self
        Flag(name=self.cls_key, dtype=Flag.TYPE.STRING, alias=self.name,
             choices=list(REGISTRIES[backend][self.module_name].keys()), default=self.default,
             help=f"The class name of {self.module_name} for '{self.help}'").define(
            arg_parser, default_is_none=False)
        Flag(name=self.params_key, dtype=Flag.TYPE.STRING, default="{}",
             help=f"The json/yaml-like parameter string for {self.module_name}").define(
            arg_parser, default_is_none=False)
        return arg_parser


COMMON_DATA_ARGS = [
    Flag("shuffle_buffer", dtype=Flag.TYPE.INTEGER, default=0,
         help="The buffer size for dataset shuffle."),
    Flag("batch_size", dtype=Flag.TYPE.INTEGER, default=None,
         help="The number of samples per update."),
    Flag("batch_size_per_gpu", dtype=Flag.TYPE.INTEGER, default=None,
         help="The per-GPU batch size, that takes precedence of `batch_size`."),
    Flag("cache_dataset", dtype=Flag.TYPE.BOOLEAN,
         help="Whether to cache the training data in memory.")]

DEFAULT_CONFIG_FLAG = Flag(name="config_paths", dtype=Flag.TYPE.STRING, multiple=True,
                           help="Path to a json/yaml configuration files defining FLAG values. "
                                "Multiple files can be separated by commas. Files are merged recursively. "
                                "Setting a key in these files is equivalent to "
                                "setting the FLAG value with the same name.")

EXTRA_IMPORT_LIB = Flag(name="include", dtype=Flag.TYPE.STRING, multiple=True,
                        help="The extra python path to be included and imported.")


def add_extra_includes():
    arg_parser = argparse.ArgumentParser()
    EXTRA_IMPORT_LIB.define(arg_parser)
    parsed, _ = arg_parser.parse_known_args()
    include = parsed.include
    if include is None:
        return
    for path in include:
        if not os.path.isdir(path):
            try:
                importlib.import_module(path)
                logging.info(f"[INFO] import user package {path}")
            except (RuntimeError, ImportError, tf.errors.OpError) as e:
                logging.info(traceback.format_exc(e))
                logging.info(f"WARNING: fail to import {path}")
            continue
        for file in os.listdir(path):
            if not file.startswith('_') and not file.startswith('.') and file.endswith('.py'):
                module_name = file[:file.find('.py')] if file.endswith('.py') else file
                src_file = os.path.join(path, file)
                with tf.io.gfile.GFile(src_file) as fp:
                    should_skip = True
                    for line in fp:
                        if line.strip().startswith("@register"):
                            should_skip = False
                            break
                    if should_skip:
                        logging.info(f"[INFO] skip {src_file}")
                        continue
                trg_file = os.path.join(os.path.dirname(__file__), "userdef/" + file)
                tf.io.gfile.copy(src_file, trg_file, overwrite=True)
                try:
                    importlib.import_module("neurst.utils.userdef." + module_name)
                    logging.info(f"[INFO] import user-defined {src_file}")
                except (RuntimeError, ImportError, tf.errors.OpError) as e:
                    logging.info(traceback.format_exc(e))
                    logging.info(f"WARNING: fail to import {src_file}")


def define_flags(flag_list: list, arg_parser=None, with_config_file=True) -> argparse.ArgumentParser:
    """ Defines the root module name.

    Args:
        flag_list: A list of flags.
        arg_parser: The argument parser.
        with_config_file: Whether to define `config_paths` as default.

    Returns: The argument parser.
    """
    add_extra_includes()
    if arg_parser is None:
        arg_parser = argparse.ArgumentParser()
    if with_config_file:
        DEFAULT_CONFIG_FLAG.define(arg_parser)
    for f in flag_list:
        f.define(arg_parser)
    return arg_parser


def get_argparser(module_name, cls_name, backend="tf") -> argparse.ArgumentParser:
    """ Returns the argument parser for the class.

    Args:
        module_name: The registered module name.
        cls_name: The class name (or alias).
        backend: The DL backend.

    Returns: An argument parser that parses the class args.
    """
    arg_parser = argparse.ArgumentParser()
    if hasattr(REGISTRIES[backend][module_name][cls_name], "class_or_method_args"):
        for f in REGISTRIES[backend][module_name][cls_name].class_or_method_args():
            f.define(arg_parser)
    return arg_parser


def _flatten_args(flag_list, from_args, backend="tf"):
    args = copy.deepcopy(from_args)
    flattened_args = {}
    for f in flag_list:
        if isinstance(f, Flag) and f.name in args:
            flattened_args[f.name] = args.pop(f.name)
    for f in flag_list:
        if isinstance(f, ModuleFlag):
            if f.cls_key in args:
                flattened_args[f.cls_key] = args.pop(f.cls_key)
                args.pop(f.name, None)
            elif f.name in args:
                flattened_args[f.cls_key] = args.pop(f.name)
            if f.cls_key in flattened_args and flattened_args[f.cls_key] and args.get(f.params_key, None):
                if hasattr(REGISTRIES[backend][f.module_name][flattened_args[f.cls_key]], "class_or_method_args"):
                    for ff in REGISTRIES[backend][f.module_name][flattened_args[f.cls_key]].class_or_method_args():
                        if isinstance(ff, Flag):
                            if ff.name in args[f.params_key] and ff.name not in flattened_args:
                                flattened_args[ff.name] = args[f.params_key].pop(ff.name)
                        else:
                            if ff.cls_key in args:
                                flattened_args[ff.cls_key] = args.pop(ff.cls_key)
                                args.pop(ff.name, None)
                            elif ff.name in args:
                                flattened_args[ff.cls_key] = args.pop(ff.name)
                            elif ff.cls_key in args[f.params_key]:
                                flattened_args[ff.cls_key] = args[f.params_key].pop(ff.cls_key)
                            elif ff.name in args[f.params_key]:
                                flattened_args[ff.cls_key] = args[f.params_key].pop(ff.name)
                            if ff.params_key in args[f.params_key]:
                                flattened_args[ff.params_key] = deep_merge_dict(
                                    args[f.params_key][ff.params_key], flattened_args.get(ff.params_key, {}))
                else:
                    flattened_args[f.params_key] = args.pop(f.params_key)
                args.pop(f.params_key, None)
    return deep_merge_dict(flattened_args, args)


def _args_preload_from_config_files(args):
    cfg_file_args = yaml_load_checking(load_from_config_path(
        flatten_string_list(getattr(args, DEFAULT_CONFIG_FLAG.name, None))))
    return cfg_file_args


def parse_flags(flag_list, arg_parser: argparse.ArgumentParser,
                args_preload_func=_args_preload_from_config_files):
    """ Parses flags from argument parser.

    Args:
        flag_list: A list of flags.
        arg_parser: The program argument parser.
        args_preload_func: A callable function for pre-loading arguments, maybe from
            config file, hyper parameter set.
    """
    program_parsed_args, remaining_argv = arg_parser.parse_known_args()
    cfg_file_args = {}
    if args_preload_func is not None:
        cfg_file_args = args_preload_func(program_parsed_args)
    program_parsed_args = yaml_load_checking(program_parsed_args.__dict__)
    top_program_parsed_args = {}
    for f in flag_list:
        flag_key = f.name
        if isinstance(f, ModuleFlag):
            flag_key = f.cls_key
            top_program_parsed_args[f.params_key] = {}
            if program_parsed_args.get(f.params_key, None) is not None:
                top_program_parsed_args[f.params_key] = program_parsed_args[f.params_key]
            if f.params_key in cfg_file_args:
                top_program_parsed_args[f.params_key] = deep_merge_dict(
                    cfg_file_args[f.params_key], top_program_parsed_args[f.params_key])
        if program_parsed_args.get(flag_key, None) is not None:
            top_program_parsed_args[flag_key] = program_parsed_args[flag_key]
        elif flag_key in cfg_file_args:
            top_program_parsed_args[flag_key] = cfg_file_args[flag_key]
        else:
            top_program_parsed_args[flag_key] = f.default

    return top_program_parsed_args, remaining_argv


def intelligent_parse_flags(flag_list, arg_parser: argparse.ArgumentParser,
                            args_preload_func=_args_preload_from_config_files,
                            backend="tf"):
    """ Parses flags from argument parser.

    Args:
        flag_list: A list of flags.
        arg_parser: The program argument parser.
        args_preload_func: A callable function for pre-loading arguments, maybe from
            config file, hyper parameter set.
        backend: The DL backend.
    """
    program_parsed_args, remaining_argv = arg_parser.parse_known_args()
    cfg_file_args = {}
    if args_preload_func is not None:
        cfg_file_args = args_preload_func(program_parsed_args)
    top_program_parsed_args = _flatten_args(flag_list,
                                            yaml_load_checking(program_parsed_args.__dict__))
    for f in flag_list:
        if isinstance(f, ModuleFlag):
            if f.cls_key in top_program_parsed_args and top_program_parsed_args[f.cls_key]:
                cfg_file_args[f.cls_key] = top_program_parsed_args[f.cls_key]
    cfg_file_args = _flatten_args(flag_list, cfg_file_args)
    for f in flag_list:
        if isinstance(f, Flag):
            if top_program_parsed_args[f.name] is None:
                top_program_parsed_args[f.name] = cfg_file_args.get(f.name, None)
            cfg_file_args.pop(f.name, None)
        else:
            submodule_cls = (top_program_parsed_args.get(f.cls_key, None)
                             or cfg_file_args.get(f.cls_key, None))
            cfg_file_args.pop(f.cls_key, None)
            if submodule_cls is None:
                continue
            top_program_parsed_args[f.cls_key] = submodule_cls
            if top_program_parsed_args.get(f.params_key, None) is None:
                top_program_parsed_args[f.params_key] = {}
            module_arg_parser = get_argparser(f.module_name, submodule_cls)
            module_parsed_args, remaining_argv = module_arg_parser.parse_known_args(remaining_argv)
            module_parsed_args = yaml_load_checking(module_parsed_args.__dict__)

            if hasattr(REGISTRIES[backend][f.module_name][submodule_cls], "class_or_method_args"):
                key_cfg_file_args = _flatten_args(
                    REGISTRIES[backend][f.module_name][submodule_cls].class_or_method_args(), cfg_file_args)
                for inner_f in REGISTRIES[backend][f.module_name][submodule_cls].class_or_method_args():
                    flag_key = inner_f.name
                    if isinstance(inner_f, ModuleFlag):
                        flag_key = inner_f.cls_key
                        cfg_file_args.pop(flag_key, None)
                    if module_parsed_args[flag_key] is not None:
                        top_program_parsed_args[f.params_key][flag_key] = module_parsed_args[flag_key]
                        top_program_parsed_args.pop(flag_key, None)
                        key_cfg_file_args.pop(flag_key, None)
                        cfg_file_args.pop(flag_key, None)
                    elif flag_key in top_program_parsed_args:
                        top_program_parsed_args[f.params_key][flag_key] = top_program_parsed_args.pop(flag_key)
                        key_cfg_file_args.pop(flag_key, None)
                        cfg_file_args.pop(flag_key, None)
                    elif flag_key in key_cfg_file_args:
                        top_program_parsed_args[f.params_key][flag_key] = key_cfg_file_args.pop(flag_key)
                        cfg_file_args.pop(flag_key, None)

                    if isinstance(inner_f, ModuleFlag):
                        top_program_parsed_args[f.params_key][inner_f.params_key] = deep_merge_dict(
                            cfg_file_args.pop(inner_f.params_key, {}) or {},
                            deep_merge_dict(top_program_parsed_args[f.params_key].pop(inner_f.params_key, {}) or {},
                                            deep_merge_dict(top_program_parsed_args.pop(inner_f.params_key, {}) or {},
                                                            module_parsed_args.pop(inner_f.params_key, {}) or {})))
    top_program_parsed_args = deep_merge_dict(cfg_file_args, top_program_parsed_args)
    for f in flag_list:
        if isinstance(f, Flag):
            if f.name not in top_program_parsed_args or top_program_parsed_args[f.name] is None:
                top_program_parsed_args[f.name] = f.default
    return top_program_parsed_args, remaining_argv


def extend_define_and_parse(flag_name, args, remaining_argv, backend="tf"):
    f = _DEFINED_FLAGS.get(flag_name, None)
    if f is None or not isinstance(f, ModuleFlag):
        return args
    if not hasattr(REGISTRIES[backend][f.module_name][args[f.cls_key]], "class_or_method_args"):
        return args
    arg_parser = argparse.ArgumentParser()
    for ff in REGISTRIES[backend][f.module_name][args[f.cls_key]].class_or_method_args():
        if isinstance(ff, ModuleFlag):
            if args[f.params_key].get(ff.cls_key, None):
                this_cls = REGISTRIES[backend][ff.module_name][args[f.params_key][ff.cls_key]]
                if hasattr(this_cls, "class_or_method_args"):
                    for fff in this_cls.class_or_method_args():
                        fff.define(arg_parser)
    parsed_args, remaining_argv = arg_parser.parse_known_args(remaining_argv)
    parsed_args = yaml_load_checking(parsed_args.__dict__)
    for ff in REGISTRIES[backend][f.module_name][args[f.cls_key]].class_or_method_args():
        if isinstance(ff, ModuleFlag):
            if args[f.params_key].get(ff.cls_key, None):
                this_cls = REGISTRIES[backend][ff.module_name][args[f.params_key][ff.cls_key]]
                if hasattr(this_cls, "class_or_method_args"):
                    if args[f.params_key].get(ff.params_key, None) is None:
                        args[f.params_key][ff.params_key] = {}
                    for fff in this_cls.class_or_method_args():
                        flag_key = fff.name
                        if isinstance(fff, ModuleFlag):
                            flag_key = fff.cls_key
                        if parsed_args[flag_key] is not None:
                            args[f.params_key][ff.params_key][flag_key] = parsed_args[flag_key]
                            args.pop(flag_key, None)
                            args.pop(fff.name, None)
                        elif flag_key in args:
                            args[f.params_key][ff.params_key][flag_key] = args.pop(flag_key)
                            args.pop(fff.name, None)
                        elif fff.name in args:
                            args[f.params_key][ff.params_key][flag_key] = args.pop(fff.name)
                        elif fff.name in args[f.params_key][ff.params_key]:
                            if flag_name not in args[f.params_key][ff.params_key]:
                                args[f.params_key][ff.params_key][flag_key] = args[f.params_key][ff.params_key].pop(
                                    fff.name)
                        if isinstance(fff, ModuleFlag):
                            args[f.params_key][ff.params_key][fff.params_key] = deep_merge_dict(
                                args[f.params_key][ff.params_key].get(fff.params_key, {}) or {},
                                deep_merge_dict(args.get(fff.params_key, {}) or {},
                                                parsed_args.get(fff.params_key, {}) or {}))

    return args, remaining_argv


def _handle_too_long_verbosity(indent, key, val, default, help_txt):
    text = indent + key + ": "
    if isinstance(val, list) and len(val) > 10:
        text += str(val[:10] + ["....."])
    elif isinstance(val, dict):
        new_val = {}
        for k, v in val.items():
            if isinstance(v, list) and len(v) > 10:
                new_val[k] = v[:10] + ["......"]
            else:
                new_val[k] = v
        text += json.dumps(new_val)
    else:
        text += f"{val}"
    text += "     # "
    if default is not None:
        text += f"(default: {default}) "
    text += help_txt
    logging.info(text)


def verbose_flags(flag_list, args, remaining_argv, backend="tf"):
    logging.info("==========================================================================")
    logging.info("Parsed all matched flags: ")
    verbose_args = copy.deepcopy(args)
    for f in flag_list:
        if isinstance(f, Flag):
            if f.name in verbose_args:
                logging.info(f" {f.name}: {verbose_args.pop(f.name)}     # (default: {f.default}) {f.help}")
        else:
            if f.cls_key in verbose_args:
                logging.info(f" {f.cls_key}: {verbose_args[f.cls_key]}")
            if f.params_key in verbose_args:
                if (verbose_args.get(f.cls_key, None) and hasattr(
                    REGISTRIES[backend][f.module_name][verbose_args[f.cls_key]], "class_or_method_args")):
                    logging.info(f" {f.params_key}:")
                    for ff in REGISTRIES[backend][f.module_name][verbose_args[f.cls_key]].class_or_method_args():
                        if isinstance(ff, Flag):
                            if ff.name in verbose_args[f.params_key]:
                                _handle_too_long_verbosity("   ", ff.name, verbose_args[f.params_key][ff.name],
                                                           ff.default, ff.help)
                        else:
                            if ff.cls_key in verbose_args[f.params_key]:
                                logging.info(f"   {ff.cls_key}: {verbose_args[f.params_key][ff.cls_key]}")
                            if ff.params_key in verbose_args[f.params_key]:
                                _handle_too_long_verbosity("   ", ff.params_key,
                                                           verbose_args[f.params_key][ff.params_key],
                                                           None, ff.help)
                else:
                    logging.info(f" {f.params_key}: {json.dumps(verbose_args[f.params_key])}")
            verbose_args.pop(f.cls_key, None)
            verbose_args.pop(f.params_key, None)
    if len(verbose_args) > 0:
        logging.info("")
        logging.info("Other flags:")
        for k, v in verbose_args.items():
            logging.info(f" {k}: {str(v)}")
    if len(remaining_argv) > 0:
        logging.info("")
        logging.info("Remaining unparsed flags: ")
        text = None
        for arg in remaining_argv:
            if arg.startswith("-"):
                if text and len(text) > 0:
                    logging.info("  {}".format(" ".join(text)))
                text = []
            if text is None:
                continue
            text.append(arg)
        if text and len(text) > 0:
            logging.info("  {}".format(" ".join(text)))
    logging.info("==========================================================================")
