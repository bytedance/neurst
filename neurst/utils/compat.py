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
import re
from distutils.version import LooseVersion

import tensorflow as tf
from absl import logging

from neurst.utils.checkpoints import NameBasedCheckpointManager

GLOBAL_SETTING = dict()
FLOAT_MIN = -1.e9
CUSTOM_GLOBAL_FLOATX = "float32"

IS_PREV_TF_2_4_0 = LooseVersion(tf.__version__) < LooseVersion("2.4")


def _broadcast_global_setting(name, var):
    global GLOBAL_SETTING
    GLOBAL_SETTING[name] = var


def _get_global_setting_or_default(name, default=None):
    global GLOBAL_SETTING
    return GLOBAL_SETTING.get(name, default)


class ModeKeys(object):
    """ Standard names for model modes. """
    TRAIN = "train"
    INFER = "infer"
    EVAL = "eval"

    _fields = ["train", "infer", "eval"]


class DataStatus(object):
    """ Standard names for data sample status. """
    # the status of the data,
    # where RAW - the data is unprocessed,
    #       PROCESSED - the data is already processed,
    #       PROJECTED - the data is already projected into network readable format (e.g. integer)
    RAW = "raw"
    PROCESSED = "processed"
    PROJECTED = "projected"


class GlobalKeys:
    # Key to indicate various ops.
    INITIAL_GLOBAL_STEP = "global_step"
    SUMMARY_STEPS = "summary_steps"
    KERAS_TRAIN_MODEL = "train_model"

    # computation precision
    FLOAT_DTYPE = "FLOAT_DTYPE"
    FLOAT_MIN = "FLOAT_MIN"

    # Metrics prefix for tensorboard
    TBPREFIX_TRAINING = "training"
    TBPREFIX_VALID = "valid"

    # checkpoint manager
    SAVER = "checkpoint_manager"

    # distributed training workers
    DIST_WORKER_ID = "distributed_worker_id"
    DIST_NUM_WORKERS = "distributed_num_workers"
    DIST_STRATEGY = "distributed_strategy"


def register_initial_step(step):
    """ Memorizes the initial step. """
    _broadcast_global_setting(GlobalKeys.INITIAL_GLOBAL_STEP, step)


def get_registered_initial_step():
    """ Gets the registered initial step. """
    return _get_global_setting_or_default(GlobalKeys.INITIAL_GLOBAL_STEP, 0)


def register_distributed_worker_setting(worker_id, num_workers, strategy):
    """ Memorizes the current worker id and total number of workers. """
    logging.info(f"Register distribution strategy: {str(strategy)} "
                 f"on worker {worker_id} (total {num_workers}). ")
    _broadcast_global_setting(GlobalKeys.DIST_WORKER_ID, worker_id)
    _broadcast_global_setting(GlobalKeys.DIST_NUM_WORKERS, num_workers)
    _broadcast_global_setting(GlobalKeys.DIST_STRATEGY, strategy)


def get_distributed_worker_setting():
    return (_get_global_setting_or_default(GlobalKeys.DIST_WORKER_ID, 0),
            _get_global_setting_or_default(GlobalKeys.DIST_NUM_WORKERS, 1),
            _get_global_setting_or_default(GlobalKeys.DIST_STRATEGY, None))


def register_computation_dtype(floatx, float_min):
    """ Register computation type of this program. """
    global CUSTOM_GLOBAL_FLOATX
    global FLOAT_MIN
    CUSTOM_GLOBAL_FLOATX = floatx
    FLOAT_MIN = float_min


def get_saver_or_default(model=None, model_dir=None, **kwargs):
    """ Creates the default checkpoint manager.

    Args:
        model: A tf.keras.model.Model (should be already called once).
        model_dir: The save to path, either on disk or HDFS.
        kwargs: Arbitrary augments for initializing custom checkpoint manager.
    Returns: A checkpoint manager
    """
    checkpoint_manager = _get_global_setting_or_default(
        GlobalKeys.SAVER, default=None)
    if checkpoint_manager is None:
        assert model is not None, (
            "`model` must be provided for default saver creation.")
        assert model_dir, (
            "`model_dir` must be provided for default saver creation.")
        checkpoint_manager = NameBasedCheckpointManager(
            model=model, directory=model_dir, **kwargs)
        _broadcast_global_setting(GlobalKeys.SAVER, checkpoint_manager)
    assert checkpoint_manager is not None
    return checkpoint_manager


def hack_global_step(checkpoint_dir):
    """ Returns the global step according to the checkpoint name. """
    if tf.io.gfile.isdir(checkpoint_dir):
        checkpoint_dir = tf.train.latest_checkpoint(checkpoint_dir)
    if checkpoint_dir is None:
        return None
    base_name = os.path.basename(checkpoint_dir)
    re_search = re.search(r"\d{2,}", base_name)
    if re_search:
        return int(re_search.group())
    return None


def wrapper_var_name(name_from_ckpt):
    name = name_from_ckpt.replace(".S", "/")
    name = name.replace("/.ATTRIBUTES/VARIABLE_VALUE", "")
    return name


def is_tf_tensor(x):
    if IS_PREV_TF_2_4_0:
        return isinstance(x, tf.Tensor)
    return tf.is_tensor(x)
