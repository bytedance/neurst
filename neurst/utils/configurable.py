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
import copy
import os

import tensorflow as tf
import yaml
from absl import logging

from neurst.utils.misc import temp_download


def copy_dict_list(d):
    """ Repeat a python build-in dict. """
    if type(d) in [int, float, bool, str]:
        return d
    try:
        new_d = {}
        for k, v in d.items():
            new_d[k] = copy_dict_list(v)
        return new_d
    except AttributeError:
        try:
            new_d = []
            for v in d:
                new_d.append(v)
            return new_d
        except TypeError:
            return d


def print_params(title, params, with_none=True, indent=0):
    """ Prints parameters.

    Args:
        title: A string.
        params: A dict.
    """
    if indent > 0:
        indent = "".join([" "] * indent)
    else:
        indent = ""

    def _params_to_stringlist(params, prefix):
        """ Convert a dictionary/list of parameters to a
            formatted string for logging purpose.

        Args:
            params: A dictionary/list of parameters.
            prefix: A string.

        Returns: A format string.

        Raises:
            ValueError: if unknown type of `params`.
        """
        param_list = []
        if isinstance(params, dict):
            for key, val in params.items():
                if not with_none and val is None:
                    continue
                param_list.append(prefix + key + ": ")
                if isinstance(val, dict) or isinstance(val, list):
                    param_list.extend(_params_to_stringlist(val, prefix + "  "))
                else:
                    param_list[-1] += str(val)
        elif isinstance(params, list):
            prefix += "  "
            for item in params:
                if not isinstance(item, dict):
                    newprefix = copy.deepcopy(prefix[:-2])
                    newprefix += "- "
                    param_list.append(newprefix + str(item))
                    continue
                for idx, (key, val) in enumerate(item.items()):
                    if idx == 0:
                        newprefix = copy.deepcopy(prefix[:-2])
                        newprefix += "- "
                        param_list.append(newprefix + key + ": ")
                    else:
                        param_list.append(prefix + key + ": ")
                    if isinstance(val, dict):
                        param_list.extend(_params_to_stringlist(val, prefix + "  "))
                    else:
                        param_list[-1] += str(val)
        else:
            raise ValueError("Unrecognized type of params: {}".format(str(params)))
        return param_list

    logging.info(title)
    if params is None:
        return
    for info in _params_to_stringlist(params, "  " + indent):
        logging.info(info)


def extract_constructor_params(locals_of_this_fn,
                               keep_out_list=None,
                               verbose=True,
                               verbose_title=None):
    """ Removes some elements in `locals()` called in the beginning of constructor
        and does logging.

    Args:
        locals_of_this_fn: The `locals()` of a function.
        keep_out_list: A list of keepout keys.
        verbose: A bool, whether to do logging.
        verbose_title: A string, the logging title.

    Returns:
        A dict.
    """
    if keep_out_list is None:
        keep_out_list = []
    assert isinstance(keep_out_list, list), (
        "`keep_out_list` must be a list of strings or none.")
    params = dict()
    for k, v in locals_of_this_fn.items():
        if k in ["self", "_"] or k.startswith("__") or k in keep_out_list:
            continue
        elif k == "kwargs" and isinstance(v, dict):
            for kk, vv in v.items():
                params[kk] = vv
        else:
            params[k] = v
    if verbose:
        vb = params.pop("verbose", None)
        if vb is None or vb:
            if not verbose_title:

                try:
                    name = params.pop("name", None)
                    verbose_title = locals_of_this_fn["self"].__class__.__name__
                    if name:
                        verbose_title = name + "(class: {})".format(
                            verbose_title)
                    verbose_title = "Parameters for " + verbose_title
                except KeyError:
                    verbose_title = "Parameters for "
            print_params(title=verbose_title, params=params)
            if "name" in locals_of_this_fn:
                params["name"] = name
        if "verbose" in locals_of_this_fn:
            params["verbose"] = vb
    return params


def parse_params(params, default_params):
    """Parses parameter values to the types defined by the default parameters.
    Default parameters are used for missing values.

    Args:
        params: A dict.
        default_params: A dict to provide parameter structure and missing values.

    Returns: A updated dict.
    """
    # Cast parameters to correct types
    if params is None:
        params = {}
    result = copy.deepcopy(default_params)
    for key, value in params.items():
        # If param is unknown, drop it to stay compatible with past versions
        if key not in default_params:
            raise ValueError("{} is not a valid model parameter".format(key))
        # Param is a dictionary
        if isinstance(value, dict):
            default_dict = default_params[key]
            if not isinstance(default_dict, dict):
                raise ValueError("{} should not be a dictionary".format(key))
            if default_dict:
                value = parse_params(value, default_dict)
            else:
                # If the default is an empty dict we do not typecheck it
                # and assume it's done downstream
                pass
        if value is None:
            continue
        if default_params[key] is None:
            result[key] = value
        else:
            result[key] = type(default_params[key])(value)
    return result


def load_from_config_path(config_paths):
    """ Loads configurations from files of yaml format.

    Args:
        config_paths: A string (each file name is seperated by ",") or
          a list of strings (file names).

    Returns: A dictionary of model configurations, parsed from config files.
    """
    if config_paths is None:
        return {}
    if isinstance(config_paths, str):
        config_paths = config_paths.strip().split(",")
    assert isinstance(config_paths, list) or isinstance(config_paths, tuple)
    model_configs = dict()
    for config_path in config_paths:
        config_path = config_path.strip()
        if not config_path:
            continue
        if config_path.startswith("http"):
            config_path = temp_download(url=config_path)
        if not tf.io.gfile.exists(config_path):
            raise OSError("config file does not exist: {config_path}".format(
                config_path=config_path))
        logging.info("loading configurations from {config_path}".format(
            config_path=config_path))
        with tf.io.gfile.GFile(config_path, "r") as config_file:
            config_flags = yaml.load(config_file, Loader=yaml.FullLoader)
            model_configs = deep_merge_dict(model_configs, config_flags)
    return model_configs


def deep_merge_dict(dict_x, dict_y, path=None, local_overwrite=True,
                    merge_only_exist=False, raise_exception=True):
    """ Recursively merges dict_y into dict_x.

    Args:
        dict_x: A dict.
        dict_y: A dict.
        path:
        local_overwrite: Whether to overwrite the `dict_x`.
        merge_only_exist: A bool, only keys in dict_x will be overwritten if True
            otherwise all key-value pairs in dict_y will be written into dict_x.
        raise_exception: A bool, whether to raise KerError exception when keys in
            dict_y but not in dict_x when merge_only_exist=True,

    Returns:
        An updated dict of dict_x

    Raises:
        KeyError: keys in dict_y but not in dict_x when merge_only_exist=True
            and raise_exception=True.
    """
    if path is None:
        path = []
    if not local_overwrite:
        dict_x = copy.deepcopy(dict_x)
    for key in dict_y:
        if dict_y[key] is None:
            if key not in dict_x:
                dict_x[key] = None
            continue
        if key in dict_x:
            if isinstance(dict_x[key], dict) and isinstance(dict_y[key], dict):
                deep_merge_dict(dict_x[key], dict_y[key], path + [str(key)],
                                local_overwrite=local_overwrite,
                                merge_only_exist=merge_only_exist)
            elif dict_x[key] == dict_y[key]:
                pass  # same leaf value
            else:
                dict_x[key] = dict_y[key]
        else:
            if merge_only_exist:
                if raise_exception:
                    raise KeyError("Key {} not in dict_x.".format(key))
            else:
                dict_x[key] = dict_y[key]
    return dict_x


class ModelConfigs:
    """ A class for dumping and loading model configurations. """

    MODEL_CONFIG_YAML_FILENAME = "model_configs.yml"

    @staticmethod
    def dump(model_config, output_dir, **kwaux):
        """ Dumps model configurations.

        Args:
            model_config: A dict.
            output_dir: A string, the output directory.
            kwaux: Arbitrary extra parameters.
        """
        cfg = copy.deepcopy(model_config)
        cfg = deep_merge_dict(cfg, kwaux, merge_only_exist=False)
        model_config_filename = os.path.join(output_dir, ModelConfigs.MODEL_CONFIG_YAML_FILENAME)
        if not tf.io.gfile.exists(output_dir):
            tf.io.gfile.makedirs(output_dir)
        logging.info("Saving model configurations to directory: {}".format(output_dir))
        with tf.io.gfile.GFile(model_config_filename, "w") as file:
            yaml.dump(cfg, file, default_flow_style=False)

    @staticmethod
    def load(model_dir):
        """ Loads model configurations.

        Args:
            model_dir: A string, the directory.

        Returns: A dict.
        """
        if not tf.io.gfile.isdir(model_dir):
            model_dir = os.path.dirname(model_dir)
        model_config_filename = os.path.join(model_dir, ModelConfigs.MODEL_CONFIG_YAML_FILENAME)
        if not tf.io.gfile.exists(model_config_filename):
            raise FileNotFoundError("Fail to find model config file: {}".format(model_config_filename))
        with tf.io.gfile.GFile(model_config_filename, "r") as file:
            model_configs = yaml.load(file, Loader=yaml.FullLoader)
        logging.info("Loading models configs from {model_config_filename}".format(
            model_config_filename=model_config_filename))
        return model_configs


def yaml_load_checking(args):
    for k in args:
        if args[k] and isinstance(args[k], str):
            try:
                v = yaml.load(args[k], Loader=yaml.FullLoader)
                args[k] = v
            except yaml.YAMLError:
                pass
    return args
