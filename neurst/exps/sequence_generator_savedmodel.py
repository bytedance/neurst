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
from absl import logging

from neurst.exps import register_exp
from neurst.exps.sequence_generator import SequenceGenerator
from neurst.models.model_utils import summary_model_variables
from neurst.training import training_utils
from neurst.utils.configurable import ModelConfigs
from neurst.utils.flags_core import Flag


@register_exp("generation_savedmodel")
class SequenceGeneratorSavedmodel(SequenceGenerator):
    """ Main exps for exporting savedmodel for sequence generation. """

    def __init__(self, args, **kwargs):
        """ Initializes a util class for exporting savedmodel for sequence generation. """
        self._export_path = args["export_path"]
        assert self._export_path, "`export_path` must be provided."
        self._version = args["version"]
        super(SequenceGeneratorSavedmodel, self).__init__(args, **kwargs)

    @staticmethod
    def class_or_method_args():
        this_args = super(SequenceGeneratorSavedmodel,
                          SequenceGeneratorSavedmodel).class_or_method_args()
        this_args.extend([
            Flag("export_path", dtype=Flag.TYPE.STRING, default=None,
                 help="The path to the savedmodel."),
            Flag("version", dtype=Flag.TYPE.INTEGER, default=1,
                 help="The version of the model."),
        ])
        return this_args

    def run(self):
        """ Export savedmodel for sequence generator.

        Step 1: Build model and restore checkpoints.
        Step 2: Export.
        """
        with training_utils.get_strategy_scope(self.strategy):
            model = self._build_and_restore_model()
            keras_model = self.build_generation_model(self.task, model, self._search_layer)
            keras_model.summary()
            summary_model_variables(keras_model)

        export_path = os.path.join(self._export_path, str(self._version))
        logging.info("Saving model to {}".format(export_path))
        tf.keras.models.save_model(
            keras_model,
            export_path,
            overwrite=True,
            include_optimizer=False,
            save_format=None,
            signatures=None,
            options=None)
        loaded = tf.saved_model.load(export_path)
        tf.io.gfile.copy(os.path.join(self.model_dir, ModelConfigs.MODEL_CONFIG_YAML_FILENAME),
                         os.path.join(export_path, ModelConfigs.MODEL_CONFIG_YAML_FILENAME),
                         overwrite=True)
        logging.info("========== signatures ==========")
        for x in loaded.signatures.keys():
            logging.info(f"structured outputs for {x}:")
            logging.info("    {}".format(str(loaded.signatures["serving_default"].structured_outputs)))
