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
import functools
import os
import re
import time
import traceback

import numpy
import tensorflow as tf
from absl import logging

from neurst.utils import compat
from neurst.utils.converters import Converter, build_converter


def remove_checkpoint_by_prefix(dirname, prefix):
    if prefix is None:
        return
    prefix = os.path.join(dirname, prefix)
    datas = tf.io.gfile.glob(prefix + ".data-?????-of-?????")
    for f in datas + [prefix + ".index"]:
        try:
            tf.io.gfile.remove(f)
        except tf.errors.OpError:
            logging.info(traceback.format_exc())


def restore_custom_checkpoint(checkpoint, checkpoint_path, model):
    """ Restore checkpoint from checkpoint_path.

    Args:
        checkpoint: A tf.train.Checkpoint.
        checkpoint_path: A string indicating the checkpoint path.
        model: A keras model.

    Returns: The checkpoint path if successfully restored or None otherwise.
    """
    traced_vars = model.weights
    pre_vars = dict(
        [(v.name.split(":")[0], v.numpy()) for v in traced_vars])
    try:
        checkpoint.restore(checkpoint_path).expect_partial()
    except (tf.errors.OpError, ValueError) as e:
        logging.info(f"ERROR: An exception occurs when trying to restore from `{checkpoint_path}`. ")
        logging.info("ERROR:Please check the setting of checkpoint restoration.")
        raise e

    logging.info('Restoring checkpoint from {latest_ckpt}'.format(
        latest_ckpt=checkpoint_path))
    after_vars = dict(
        [(v.name.split(":")[0], v.numpy()) for v in traced_vars])
    restored_var_names = []
    unrestored_var_names = []
    for v_name, v in pre_vars.items():
        try:
            if numpy.sqrt(numpy.sum((v - after_vars[v_name]) ** 2)) < 1e-6:
                unrestored_var_names.append(v_name)
            else:
                restored_var_names.append(v_name)
        except TypeError:
            logging.info(f"Ignore non-numeric variable: {v_name}")
    if len(unrestored_var_names) == 0:
        logging.info("All variables matched with checkpoint: {}".format(checkpoint_path))
    elif len(restored_var_names) == 0:
        logging.info("No variables matched with checkpoint: {}".format(checkpoint_path))
        if model is not None:
            logging.info("Trying `keras_model.load_weights()`")
            try:
                model.load_weights(checkpoint_path)
            except (ImportError, ValueError, tf.errors.OpError, AssertionError):
                logging.info("Fail to call model.load_weights.")
                logging.info(traceback.format_exc())
                return None
    else:
        for v_name in restored_var_names:
            logging.info("Restored {}".format(v_name))
        for v_name in unrestored_var_names:
            logging.info("Unrestored {}".format(v_name))
    return checkpoint_path


class _CustomSaver(object):
    """ Custom checkpoint manager for saving checkpoints. """

    def __init__(self, directory, checkpoint, max_to_keep=8):
        """ Initializes the checkpoint manager.

        Args:
            directory: The path to a directory in which to write checkpoints.
            checkpoint: A checkpoint
            max_to_keep: The maximum checkpoint numbers to keep.

        Raises:
            ValueError: Neither `traced_vars` nor `model` is provided.
        """
        self._directory = directory
        if not tf.io.gfile.exists(self._directory):
            try:
                tf.io.gfile.makedirs(self._directory)
            except tf.errors.OpError:
                pass
        self._checkpoint = checkpoint
        self._max_to_keep = max_to_keep
        # a list of tuple: (checkpoint name, timestamp)
        self._all_model_checkpoints = []

    @property
    def checkpoint(self):
        return self._checkpoint

    @property
    def directory(self):
        return self._directory

    def _update_checkpoint_meta(self):
        """Updates the checkpoint file under each model dir. """
        while len(self._all_model_checkpoints) > self._max_to_keep:
            prefix, _ = self._all_model_checkpoints.pop(0)
            remove_checkpoint_by_prefix(dirname=self.directory, prefix=prefix)
        meta_data_str = "model_checkpoint_path: \"{}\"\n".format(self._all_model_checkpoints[-1][0])
        for path, _ in self._all_model_checkpoints:
            meta_data_str += "all_model_checkpoint_paths: \"{}\"\n".format(path)
        for _, timestamp in self._all_model_checkpoints:
            meta_data_str += "all_model_checkpoint_timestamps: {}\n".format(str(timestamp))
        with tf.io.gfile.GFile(os.path.join(self.directory, "checkpoint.incomplete"), "w") as fw:
            fw.write(meta_data_str)
        tf.io.gfile.rename(os.path.join(self.directory, "checkpoint.incomplete"),
                           os.path.join(self.directory, "checkpoint"),
                           overwrite=True)

    def save(self, prefix):
        output = self._checkpoint.write(os.path.join(self.directory, prefix))
        return output


class NameBasedCheckpointManager(_CustomSaver):
    """ The name-based checkpoint manager for saving and restoring variables. """

    def __init__(self, model, directory, max_to_keep=8, checkpoint_name="ckpt"):
        """ Initializes a custom checkpoint manager.

        Args:
            model: A tf.keras.Model.
            directory: The path to a directory in which to write checkpoints.
            max_to_keep: The maximum checkpoint numbers to keep.
            checkpoint_name: The name of each checkpoint.
        """
        self._model = model
        super(NameBasedCheckpointManager, self).__init__(
            directory=directory, checkpoint=tf.train.Checkpoint(
                **dict([(x.name.split(":")[0], x) for x in model.weights])),
            max_to_keep=max_to_keep)
        self._checkpoint_name = checkpoint_name
        logging.info("Creates checkpoint manager for directory: {}".format(directory))

    def restore(self, restore_path=None):
        """ Restores checkpoint from `save_path` or self._directory by default. """
        if restore_path is None:
            restore_path = self.directory
        latest_ckpt = tf.train.latest_checkpoint(restore_path)
        if latest_ckpt is None:
            latest_ckpt = restore_path
        if latest_ckpt:
            return restore_custom_checkpoint(self.checkpoint, latest_ckpt, self._model)

    def save(self, checkpoint_number):
        prefix = "{}-{}".format(self._checkpoint_name, checkpoint_number)
        output = super(NameBasedCheckpointManager, self).save(prefix)
        self._all_model_checkpoints.append((prefix, time.time()))
        self._update_checkpoint_meta()
        return output


class KeepBestCheckpointSaver(_CustomSaver):
    """ Custom Checkpoint manager for saving and restoring variables. """

    def __init__(self, model, directory, metric, max_to_keep=8, checkpoint_name="ckpt"):
        """ Initializes a custom checkpoint manager.

        Args:
            model: A keras model.
            directory: The path to a directory in which to write checkpoints.
            metric: A metric object.
            max_to_keep: The maximum checkpoint numbers to keep.
            checkpoint_name: The name of each checkpoint.
        """
        if directory is None:
            directory = compat.get_saver_or_default().directory
            if not directory.endswith("/"):
                directory += "/"
            directory += "best"
        super(KeepBestCheckpointSaver, self).__init__(
            checkpoint=tf.train.Checkpoint(**dict([(x.name.split(":")[0], x) for x in model.weights])),
            directory=directory, max_to_keep=max_to_keep)
        self._metric = metric
        self._checkpoint_name = checkpoint_name
        logging.info("Creates custom keep-best checkpoint manager for directory: {}".format(directory))

    def save(self, checkpoint_number, metric_value):
        """ Saves a checkpoint and updates meta values if `metric_value` is better.

        Args:
            checkpoint_number: The current step.
            metric_value: The metric result.

        Returns:
            The path to the checkpoint if it is saved, otherwise None.

        """
        # whether to save or not
        if (0 <= len(self._all_model_checkpoints) < self._max_to_keep
            or self._metric.greater_or_eq(metric_value, self._all_model_checkpoints[0][1])):
            saved_prefix = "{}-{}-{}".format(self._checkpoint_name, checkpoint_number,
                                             ("%.2f" % self._metric.get_value(metric_value)))
            path = super(KeepBestCheckpointSaver, self).save(saved_prefix)
            self._all_model_checkpoints.append((saved_prefix, self._metric.get_value(metric_value)))
            self._all_model_checkpoints = sorted(
                self._all_model_checkpoints,
                key=functools.cmp_to_key(lambda x, y: (-int(self._metric.greater_or_eq(y[1], x[1])
                                                            and y[1] != x[1]))))
            self._update_checkpoint_meta()
            return path

        return None


class AverageCheckpointSaver(_CustomSaver):
    """ Custom Checkpoint manager for saving averaged variables. """

    def __init__(self, model, directory, metric, max_to_keep=8, checkpoint_name="ckpt"):
        """ Initializes a custom checkpoint manager.

        Args:
            model: A keras model.
            directory: The path to a directory in which to write checkpoints.
            metric: A metric object.
            max_to_keep: The maximum checkpoint numbers to keep.
            checkpoint_name: The name of each checkpoint.
        """
        if directory is None:
            directory = compat.get_saver_or_default().directory
            if not directory.endswith("/"):
                directory += "/"
            directory += "best_avg"
        self._checkpoint_name = checkpoint_name
        self._traced_vars = dict([(x.name.split(":")[0], x) for x in model.weights])
        self._traced_var_names = list(self._traced_vars.keys())
        self._traced_var_numpys = []
        self._metric = metric

        v_numpys = dict([(n, v.numpy()) for n, v in self._traced_vars.items()])
        with tf.distribute.OneDeviceStrategy(device="/cpu:0").scope():
            self._avg_traced_vars = dict([(n, tf.Variable(v, dtype=v.dtype, name=n + "_avg"))
                                          for n, v in v_numpys.items()])
        super(AverageCheckpointSaver, self).__init__(
            directory=directory, max_to_keep=max_to_keep,
            checkpoint=tf.train.Checkpoint(**self._avg_traced_vars))
        logging.info("Create checkpoint manager for averaged checkpoint "
                     "of the latest {} checkpoints to dir: {}".format(self._max_to_keep, self.directory))

    def _average_checkpoint(self):
        """ Averages the checkpoints. """
        for idx, name in enumerate(self._traced_var_names):
            self._avg_traced_vars[name] = self._avg_traced_vars[name].assign(
                numpy.average([var_numpys[idx] for var_numpys in self._traced_var_numpys], axis=0))

    def save(self, checkpoint_number, metric_value):
        """ Saves a checkpoint and updates meta values if `metric_value` is better.

        Args:
            checkpoint_number: The current step.
            metric_value: The metric result.

        Returns:
            The path to the checkpoint if it is saved, otherwise None.

        """
        # keep the latest checkpoints
        self._traced_var_numpys.append(
            [self._traced_vars[x].numpy() for x in self._traced_var_names])
        if len(self._traced_var_numpys) > self._max_to_keep:
            self._traced_var_numpys.pop(0)
        if (0 <= len(self._all_model_checkpoints) < self._max_to_keep
            or self._metric.greater_or_eq(metric_value, self._all_model_checkpoints[0][1])):
            # Averages the checkpoints.
            for idx, name in enumerate(self._traced_var_names):
                self._avg_traced_vars[name] = self._avg_traced_vars[name].assign(
                    numpy.average([var_numpys[idx] for var_numpys in self._traced_var_numpys], axis=0))
            saved_prefix = "{}-{}-{}".format(self._checkpoint_name, checkpoint_number,
                                             ("%.2f" % self._metric.get_value(metric_value)))
            path = super(AverageCheckpointSaver, self).save(saved_prefix)
            self._all_model_checkpoints.append((saved_prefix, self._metric.get_value(metric_value)))
            self._all_model_checkpoints = sorted(
                self._all_model_checkpoints,
                key=functools.cmp_to_key(lambda x, y: (-int(self._metric.greater_or_eq(y[1], x[1])
                                                            and y[1] != x[1]))))
            self._update_checkpoint_meta()
            return path

        return None


def checkpoint_scope_name(checkpoint_path):
    """ Lists checkpoint variables and extract top scope name.

    Args:
        checkpoint_path: A string, the checkpoint path.

    Returns: A string or None.
    """
    try:
        var_names = [compat.wrapper_var_name(x[0]) for x in tf.train.list_variables(checkpoint_path)]
    except ValueError:
        var_names = [compat.wrapper_var_name(x[0])
                     for x in tf.train.list_variables(os.path.dirname(checkpoint_path))]
    prefixs = set()
    for n in var_names:
        n_tokens = n.strip().split("/")
        if len(n_tokens) > 1:
            prefixs.add(n_tokens[0])
    if len(prefixs) > 1:
        logging.info("WARNING: more than one scope names({}) extracted from {}. "
                     "Be careful to this behavior, "
                     "which may lead to unknown issues.".format(prefixs, checkpoint_path))
    return prefixs.pop()


def restore_checkpoint_if_possible(model, model_dir, var_name_pattern=None):
    """ Restores checkpoint from `model_dir` if exists. """
    latest_ckpt_path = tf.train.latest_checkpoint(model_dir)
    if not latest_ckpt_path:
        latest_ckpt_path = model_dir
        try:
            tf.train.list_variables(latest_ckpt_path)
        except (tf.errors.NotFoundError, ValueError, tf.errors.DataLossError):
            return None

    ckpt_scope_name = checkpoint_scope_name(latest_ckpt_path)
    vars = model.weights
    if var_name_pattern is None:
        checkpoint = tf.train.Checkpoint(
            **dict([(ckpt_scope_name + x.name.split(":")[0][x.name.index("/"):], x) for x in vars]))
    else:
        logging.info("Variables only match the {} will be restored.".format(var_name_pattern))
        checkpoint = tf.train.Checkpoint(
            **dict([(ckpt_scope_name + x.name.split(":")[0][x.name.index("/"):], x) for x in vars
                    if re.search(var_name_pattern, x.name) is not None]))
    return restore_custom_checkpoint(checkpoint, latest_ckpt_path, model)


def restore_checkpoint_if_possible_v2(model, path, model_name=None, from_prefix=None,
                                      to_prefix=None, name_pattern=None):
    """ Restores checkpoint.

    Args:
        model: A keras model.
        path: The path to the neurst checkpoint or the path/key for the converter.
        model_name: The converter name for converting checkpoints.
        from_prefix: The name prefix.
        to_prefix: The target name prefix.
        name_pattern: A regex.

    Returns: The ckpt path if successfully restored else None.
    """
    if not (model_name or from_prefix or to_prefix):
        return restore_checkpoint_if_possible(model, path, name_pattern)

    converter: Converter = build_converter(model_name)
    if converter is None:
        latest_ckpt_path = tf.train.latest_checkpoint(path)
        if not latest_ckpt_path:
            latest_ckpt_path = path
            try:
                tf.train.list_variables(latest_ckpt_path)
            except (tf.errors.NotFoundError, ValueError, tf.errors.DataLossError):
                return None
    else:
        logging.info(f"Loading {model_name} ({path}).")
        tmp_ckpt = "ram://tmp_ckpt"
        converter.convert(path, tmp_ckpt)
        latest_ckpt_path = tf.train.latest_checkpoint(tmp_ckpt)
    if from_prefix is None:
        from_prefix = [checkpoint_scope_name(latest_ckpt_path)]
    else:
        from_prefix = [x.strip("/") for x in tf.nest.flatten(from_prefix)]
    all_vars = model.weights
    if to_prefix is None:
        to_prefix = [all_vars[0].name.split("/")[0]]
    else:
        to_prefix = [x.strip("/") for x in tf.nest.flatten(to_prefix)]
    assert len(from_prefix) == len(to_prefix)
    n2v = {}
    for var in all_vars:
        if name_pattern is None or re.search(name_pattern, var.name) is not None:
            n = var.name.split(":")[0]
            for _from, _to in zip(from_prefix, to_prefix):
                if n.startswith(_to):
                    n = n.replace(_to, _from, 1)
                    break
            n2v[n] = var
    checkpoint = tf.train.Checkpoint(**n2v)
    return restore_custom_checkpoint(checkpoint, latest_ckpt_path, model)
