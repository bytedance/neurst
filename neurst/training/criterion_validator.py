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
from neurst.data.datasets import Dataset, build_dataset
from neurst.data.datasets.multiple_dataset import MultipleDataset
from neurst.training import Validator, register_validator, training_utils
from neurst.utils import compat
from neurst.utils.flags_core import Flag, ModuleFlag


@register_validator
class CriterionValidator(Validator):

    def __init__(self, args):
        super(CriterionValidator, self).__init__(args)
        self.args = args  # lazy initialization
        self._criterion = None
        self._criterion_metric = None
        self._custom_dataset = None
        self._strategy = None
        self._eval_tfds = None
        self._criterion_model = None
        self._criterion_recorder = None
        self._avg_criterion_recorder = None
        self._mixed_criterion_recorder = None
        self._criterion_start_time = None
        self._validate_criterion = True
        self._eval_task_args = args["eval_task_args"] or {}
        self._eval_task_args["batch_size"] = args["eval_batch_size"]

    @staticmethod
    def class_or_method_args():
        this_args = super(CriterionValidator, CriterionValidator).class_or_method_args()
        this_args.extend([
            ModuleFlag("eval_criterion", Criterion.REGISTRY_NAME, help="The criterion for validation."),
            ModuleFlag("eval_dataset", Dataset.REGISTRY_NAME, help="The dataset for validation."),
            Flag("eval_batch_size", dtype=Flag.TYPE.INTEGER, default=32,
                 help="The batch size for validation process."),
            Flag("eval_task_args", dtype=Flag.TYPE.STRING, default=None,
                 help="Other parameters for building validation dataset.")
        ])
        return this_args

    def build(self, strategy, task, model):
        """ Initializes. """
        self._strategy = strategy
        self._criterion: Criterion = build_criterion(self.args["eval_criterion.class"],
                                                     **self.args["eval_criterion.params"])
        self._criterion.set_model(model)
        if self._criterion is None:
            logging.info("WARNING: no criterion is provided in CriterionValidator "
                         "for validation process.")
            self._validate_criterion = False
            return self
        self._custom_dataset = build_dataset(self.args["eval_dataset.class"], **self.args["eval_dataset.params"])
        if self._custom_dataset is None:
            logging.info("WARNING: no validation dataset is provided "
                         "in CriterionValidator for validation process.")
            self._validate_criterion = False
            return self
        from neurst.exps.evaluator import Evaluator
        with training_utils.get_strategy_scope(strategy):
            self._criterion_model = Evaluator.build_evaluation_model(task, model, self._criterion)
            self._eval_tfds = training_utils.build_datasets(
                compat.ModeKeys.EVAL, strategy, self._custom_dataset, task, True, self._eval_task_args)
        self._criterion_metric = self._criterion.as_metric()
        if isinstance(self._custom_dataset, MultipleDataset):
            self._criterion_recorder = {
                name: training_utils.TrainingStatusRecorder(
                    model=model, task=task, metric=self._criterion_metric)
                for name in self._custom_dataset.datasets
            }
            self._avg_criterion_recorder = training_utils.TrainingStatusRecorder(
                model=model, task=task, metric=self._criterion_metric)
            self._mixed_criterion_recorder = training_utils.TrainingStatusRecorder(
                model=model, task=task, metric=self._criterion_metric)
        else:
            self._criterion_recorder = training_utils.TrainingStatusRecorder(
                model=model, task=task, metric=self._criterion_metric)
        self._criterion_start_time = time.time()
        return self

    def validate(self, step):
        if not self._validate_criterion:
            return
        start_time = time.time()
        results, avg_res, mixed_res = training_utils.reduce_eval_results(
            self._criterion, self._custom_dataset, training_utils.make_predictions(
                self._strategy, self._criterion_model, self._eval_tfds, self._custom_dataset))
        elapsed = time.time() - start_time
        elapsed_from_start = time.time() - self._criterion_start_time

        def _display(res, best, name=None, tb_name=None):
            if tb_name is None:
                tb_name = name
            tb_name = "" if tb_name is None else (tb_name + "_")
            name = "" if name is None else f" ({name})"
            for k, v in res.items():
                logging.info("Evaluating (%s) validation set%s: %s=%.2f (Best %.2f)  "
                             "step=%d\tElapsed %.2fs  FromSTART %.2fs",
                             self._criterion_metric.flag, name, k, v, best[k],
                             step, elapsed, elapsed_from_start)
                tf.summary.scalar(
                    compat.GlobalKeys.TBPREFIX_VALID + f"/{tb_name}{k}", v, step=step)

        if isinstance(self._custom_dataset, MultipleDataset):
            for name, res in results.items():
                self._criterion_recorder[name].record(step, res)
                _display(res, self._criterion_recorder[name].best, name=name)
            self._avg_criterion_recorder.record(step, avg_res)
            _display(avg_res, self._avg_criterion_recorder.best,
                     f"on average by weights {self._custom_dataset.sample_weights}",
                     tb_name="AVERAGE")
            self._mixed_criterion_recorder.record(step, mixed_res)
            _display(mixed_res, self._mixed_criterion_recorder.best, "MIXED")
        else:
            self._criterion_recorder.record(step, results)
            _display(results, self._criterion_recorder.best)
