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
import time

import tensorflow as tf
from absl import logging

from neurst.criterions import Criterion, build_criterion
from neurst.data.datasets.multiple_dataset import MultipleDataset
from neurst.exps import register_exp
from neurst.exps.base_experiment import BaseExperiment
from neurst.models.model_utils import summary_model_variables
from neurst.training import training_utils
from neurst.utils import compat
from neurst.utils.checkpoints import restore_checkpoint_if_possible
from neurst.utils.flags_core import ModuleFlag


@register_exp(["eval"])
class Evaluator(BaseExperiment):
    """ For evaluating on criterion. """

    def __init__(self, args, **kwargs):
        """ Initializes a util class for evaluating neural models. """
        super(Evaluator, self).__init__(**kwargs)
        self._criterion = build_criterion(args)

    @staticmethod
    def class_or_method_args():
        return [
            ModuleFlag(Criterion.REGISTRY_NAME, help="The criterion for evaluation."),
        ]

    @staticmethod
    def build_evaluation_model(task, model, criterion):
        """ Build keras model for evaluation.

        Args:
            task: The task object.
            model: An instance of neurst.models.model.BaseModel.
            criterion: The criterion for evaluation.

        Returns: the evaluation model.
        """
        inps = task.create_inputs(compat.ModeKeys.EVAL)
        formatted_inps = task.example_to_input(inps, compat.ModeKeys.EVAL)
        model_out = model(formatted_inps, is_training=False)
        keras_model = tf.keras.Model(inps, criterion(formatted_inps, model_out))
        return keras_model

    def run(self):
        """ Evaluation on a existing model.

        Step 1: Build model.
        Step 2: Builds evaluation dataset.
        Step 3: Restore checkpoints.
        Step 4: Evaluate and reduce metric.
        """

        with training_utils.get_strategy_scope(self.strategy):
            tfds = training_utils.build_datasets(compat.ModeKeys.EVAL, self.strategy,
                                                 self.custom_dataset, self.task, cache=True)
            keras_model = self.build_evaluation_model(self.task, self.model, self._criterion)
            keras_model.summary()
            summary_model_variables(keras_model)
            # Step 4: Restore checkpoints.
            stat = restore_checkpoint_if_possible(self.model, self.model_dir)
            if not stat:
                logging.info(f"WARNING: Fail to restore checkpoint from {self.model_dir}. "
                             "We assume this was done on purpose. ")
        # Step 5: Evaluate and reduce metric.
        start_time = time.time()
        results, avg_res, whole_res = training_utils.reduce_eval_results(
            self._criterion, self.custom_dataset, training_utils.make_predictions(
                self.strategy, keras_model, tfds, self.custom_dataset))
        logging.info("Evaluation elapsed: %.2fs", time.time() - start_time)

        def _display(res, name=None):
            if name:
                logging.info(f"Evaluation Results ({name}):")
            for k, v in res.items():
                logging.info("   %s: %.2f", k, v)

        if not isinstance(self.custom_dataset, MultipleDataset):
            _display(results)
        else:
            for name, res in results.items():
                _display(res, name)
            _display(avg_res, f"on average by weights {self.custom_dataset.sample_weights}")
            _display(whole_res, "mixed")
