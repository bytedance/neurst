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
import time
import traceback

import tensorflow as tf
from absl import logging

from neurst.exps import BaseExperiment, register_exp
from neurst.tasks import build_task
from neurst.training import Validator, build_validator
from neurst.utils import compat
from neurst.utils.configurable import ModelConfigs
from neurst.utils.flags_core import Flag, ModuleFlag


@register_exp
class Validation(BaseExperiment):
    """ Validation for all tasks during training. """

    def __init__(self, args, **kwargs):
        """ Initializes a util class for vaidation. """
        super(Validation, self).__init__(**kwargs)
        self._tb_log_dir = args["tb_log_dir"]
        self._waiting_interval = args["waiting_interval"]
        self._maximum_waiting_time = args["maximum_waiting_time"]
        self._validator = build_validator(args)

    @staticmethod
    def class_or_method_args():
        return [
            Flag("tb_log_dir", dtype=Flag.TYPE.STRING, default=None,
                 help="The path to store tensorboard summary, or `model_dir`/validation by default."),
            Flag("waiting_interval", dtype=Flag.TYPE.INTEGER, default=120,
                 help="The waiting interval between two evaluation steps."),
            Flag("maximum_waiting_time", dtype=Flag.TYPE.INTEGER, default=3600,
                 help="The maximum waiting time(in seconds)."),
            ModuleFlag(Validator.REGISTRY_NAME, help="The validation process during training."),
        ]

    def run(self):
        """ Repeats to call validator's validate function if new checkponts are observed.

        Step 1: Build model.
        Step 2: Fetch training status.
        while True:
            Step 3: Restore checkpoints.
            Step 4: Validate.
        """
        if self.task is None or self.model is None:
            model_cfg_waiting_rounds = self._maximum_waiting_time // self._waiting_interval
            for i in range(model_cfg_waiting_rounds):
                try:
                    args = ModelConfigs.load(self._model_dir)
                    break
                except FileNotFoundError:
                    logging.info(f"Fail to load model configs from directory: {self.model_dir}. "
                                 f"Wait for another {self._waiting_interval}s, "
                                 f"patience={model_cfg_waiting_rounds - 1 - i}.")
                    time.sleep(self._waiting_interval)
            self._task = build_task(args)
            self._model = self.task.build_model(args)
        # initialize the checkpoint manager
        saver = compat.get_saver_or_default(self.model, self.model_dir)
        # enable tensorboard
        if self._tb_log_dir is None:
            self._tb_log_dir = os.path.join(self.model_dir, "validation_{}".format(int(time.time())))
        file_writer = tf.summary.create_file_writer(self._tb_log_dir)
        file_writer.set_as_default()
        # create training
        self._validator.build(self.strategy, self.task, self.model)
        last_triggered_step = None
        accumulated_waiting_time = 0
        this_waiting_interval = next_waiting_interval = self._waiting_interval
        while True:
            bad_cnt = 0
            while bad_cnt < 5:
                try:
                    ckpt_state = tf.train.get_checkpoint_state(self.model_dir)
                    break
                except ValueError:
                    bad_cnt += 1
                    time.sleep(5)
                    logging.info(traceback.format_exc())
                    if bad_cnt >= 5:
                        ckpt_state = tf.train.get_checkpoint_state(self.model_dir)

            ckpts_to_be_restore = None
            if ckpt_state is None:
                logging.info(f"No checkpoint in directory: {self.model_dir}. Please wait.")
            else:
                all_ckpts = [(t, x) for t, x in zip(ckpt_state.all_model_checkpoint_timestamps,
                                                    ckpt_state.all_model_checkpoint_paths)]
                global_steps_to_be_restore = []
                ckpts_to_be_restore = []
                for ckpt in all_ckpts[::-1]:
                    step = compat.hack_global_step(ckpt[1])
                    if last_triggered_step is None or step > last_triggered_step:
                        ckpts_to_be_restore.insert(0, ckpt)
                        global_steps_to_be_restore.insert(0, step)
                if len(ckpts_to_be_restore) > 0:
                    accumulated_waiting_time = 0
                _start_time = time.time()
                for step, (timestamp, ckpt) in zip(global_steps_to_be_restore, ckpts_to_be_restore):
                    try:
                        stat = saver.restore(ckpt)
                    except tf.errors.NotFoundError:
                        logging.info(f"Not found checkpoint {ckpt}. Skip...")
                    if not stat:
                        logging.info(f"Fail to restore checkpoint from {ckpt}. Skip...")
                        continue
                    logging.info(f"Checkpoint with global_step={step} triggered on {timestamp}")
                    self._validator.validate(step)
                    last_triggered_step = step
                this_waiting_interval = max(this_waiting_interval - int(time.time() - _start_time), 10)
                tf.summary.flush(file_writer)
            if ckpts_to_be_restore is None:
                pass
            elif len(ckpts_to_be_restore) > 1:
                this_waiting_interval = int(this_waiting_interval * 1.
                                            * (len(ckpts_to_be_restore) // 2) / len(ckpts_to_be_restore))
                next_waiting_interval = this_waiting_interval
            elif len(ckpts_to_be_restore) == 0:
                next_waiting_interval = min(int(this_waiting_interval * 4. / 3.), self._waiting_interval)
                this_waiting_interval = this_waiting_interval // 2
            accumulated_waiting_time += this_waiting_interval
            if accumulated_waiting_time > self._maximum_waiting_time:
                logging.info(f"Waited for maximum patience: {self._maximum_waiting_time}s")
                break
            time.sleep(this_waiting_interval)
            this_waiting_interval = next_waiting_interval
