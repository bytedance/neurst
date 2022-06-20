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
import random
import time

import tensorflow as tf
from absl import logging

from neurst.data.datasets.multiple_dataset import MultipleDataset
from neurst.layers.search import SequenceSearch, build_search_layer
from neurst.metrics import Metric
from neurst.training import register_validator, training_utils
from neurst.training.criterion_validator import CriterionValidator
from neurst.utils import compat
from neurst.utils.flags_core import Flag, ModuleFlag


@register_validator
class SeqGenerationValidator(CriterionValidator):

    def __init__(self, args):
        super(SeqGenerationValidator, self).__init__(args)
        self._gen_metric = None
        self._gen_model = None
        self._gen_tfds = None
        self._gen_recorder = None
        self._avg_gen_recorder = None
        self._mixed_gen_recorder = None
        self._gen_start_time = None
        self._validate_gen = True
        self._postprocess_fn = None
        self._custom_ds_targets = None
        self._custom_ds_sources = None

    @staticmethod
    def class_or_method_args():
        this_args = super(SeqGenerationValidator, SeqGenerationValidator).class_or_method_args()
        this_args.extend([
            ModuleFlag("eval_metric", Metric.REGISTRY_NAME, help="The metric for evaluating generation results."),
            ModuleFlag("eval_search_method", SequenceSearch.REGISTRY_NAME,
                       help="The search layer for sequence generation."),
            Flag("eval_estop_patience", dtype=Flag.TYPE.INTEGER, default=0,
                 help="The training process will automatically shut down until the program "
                      "fail to acquire a better metric score anymore if `early_stop_patience` greater than 0."),
            Flag("eval_best_checkpoint_path", dtype=Flag.TYPE.STRING, default=None,
                 help="The path for checkpoints with best metric scores if provided,"
                      "otherwise, default \"`model_dir`_best\" will be used."),
            Flag("eval_auto_average_checkpoints", dtype=Flag.TYPE.BOOLEAN, default=True,
                 help="Whether to do checkpoint average on all model weights. An extra directory for averaged "
                      "weights will be created. It is only available when `eval_best_checkpoint_path` is provided."),
            Flag("eval_best_avg_checkpoint_path", dtype=Flag.TYPE.STRING, default=None,
                 help="The path to saving the averaged checkpoints."),
            Flag("eval_top_checkpoints_to_keep", dtype=Flag.TYPE.INTEGER, default=10,
                 help="The maximum number of checkpoints to be saved (`max_to_keep` for checkpoint manager), "
                      "and the number of latest checkpoints to be averaged if `eval_auto_average_checkpoints` is True. "
                      "If <= 0, no more checkpoints will be saved."),
        ])
        return this_args

    def build(self, strategy, task, model):
        super(SeqGenerationValidator, self).build(strategy, task, model)
        if self._custom_dataset is None:
            logging.info("WARNING: no validation dataset is provided "
                         "in SeqGenerationValidator for validation process.")
            self._validate_gen = False
            return self
        self._gen_metric = task.get_eval_metric(self.args, name="eval_metric", ds=self._custom_dataset)
        if self._gen_metric is None:
            logging.info("WARNING: no metric is provided "
                         "in SeqGenerationValidator for validation process.")
            self._validate_gen = False
            return self
        self._gen_metric.flag = self.args["eval_metric.class"]
        search_layer = build_search_layer(self.args["eval_search_method.class"],
                                          **self.args["eval_search_method.params"])
        if search_layer is None:
            logging.info("WARNING: no search method is provided "
                         "in SeqGenerationValidator for validation process.")
            self._validate_gen = False
            return self
        from neurst.exps.sequence_generator import SequenceGenerator
        with training_utils.get_strategy_scope(strategy):
            self._gen_model = SequenceGenerator.build_generation_model(
                task, model, search_layer)
            self._gen_tfds = training_utils.build_datasets(
                compat.ModeKeys.INFER, strategy, self._custom_dataset, task, True, self._eval_task_args)
            if isinstance(self._custom_dataset, MultipleDataset):
                for name in list(self._gen_tfds.keys()):
                    if self._custom_dataset.datasets[name].targets is None:
                        logging.info(f"WARNING: no ground truth found for validation dataset {name}.")
                        self._gen_tfds.pop(name)
                    else:
                        if self._custom_ds_targets is None:
                            self._custom_ds_targets = {}
                        if self._custom_ds_sources is None:
                            self._custom_ds_sources = {}
                        if (hasattr(self._custom_dataset.datasets[name], "raw_targets")
                            and self._custom_dataset.datasets[name].raw_targets):
                            self._custom_ds_targets[name] = self._custom_dataset.datasets[name].raw_targets
                        else:
                            self._custom_ds_targets[name] = self._custom_dataset.datasets[name].targets
                        if (hasattr(self._custom_dataset.datasets[name], "sources")
                            and self._custom_dataset.datasets[name].sources is not None):
                            self._custom_ds_sources[name] = self._custom_dataset.datasets[name].sources
                if len(self._gen_tfds) == 0:
                    logging.info("WARNING: no ground truth found for all validation datasets and "
                                 "no validation will be applied.")
                    self._validate_gen = False
                    return self
            else:
                if self._custom_dataset.targets is None:
                    logging.info("WARNING: no ground truth found for validation dataset and "
                                 "no validation will be applied.")
                    self._validate_gen = False
                    return self
                else:
                    if hasattr(self._custom_dataset, "raw_targets") and self._custom_dataset.raw_targets:
                        self._custom_ds_targets = self._custom_dataset.raw_targets
                    else:
                        self._custom_ds_targets = self._custom_dataset.targets
                    if hasattr(self._custom_dataset, "sources") and self._custom_dataset.sources is not None:
                        self._custom_ds_sources = self._custom_dataset.sources
        if isinstance(self._custom_dataset, MultipleDataset):
            self._gen_recorder = {
                name: training_utils.TrainingStatusRecorder(
                    model=model, task=task, metric=self._gen_metric)
                for name in self._gen_tfds
            }
            self._mixed_gen_recorder = training_utils.TrainingStatusRecorder(
                model=model, task=task, metric=self._gen_metric)
            self._avg_gen_recorder = training_utils.TrainingStatusRecorder(
                model=model, task=task, metric=self._gen_metric,
                estop_patience=self.args["eval_estop_patience"],
                best_checkpoint_path=self.args["eval_best_checkpoint_path"],
                auto_average_checkpoints=self.args["eval_auto_average_checkpoints"],
                best_avg_checkpoint_path=self.args["eval_best_avg_checkpoint_path"],
                top_checkpoints_to_keep=self.args["eval_top_checkpoints_to_keep"])
        else:
            self._gen_recorder = training_utils.TrainingStatusRecorder(
                model=model, task=task, metric=self._gen_metric,
                estop_patience=self.args["eval_estop_patience"],
                best_checkpoint_path=self.args["eval_best_checkpoint_path"],
                auto_average_checkpoints=self.args["eval_auto_average_checkpoints"],
                best_avg_checkpoint_path=self.args["eval_best_avg_checkpoint_path"],
                top_checkpoints_to_keep=self.args["eval_top_checkpoints_to_keep"])
        from neurst.exps.sequence_generator import SequenceGenerator
        self._postprocess_fn = lambda y: SequenceGenerator.postprocess_generation(task, y)
        self._gen_start_time = time.time()
        return self

    def validate(self, step):
        super(SeqGenerationValidator, self).validate(step)
        if not self._validate_gen:
            return
        start_time = time.time()
        results = training_utils.make_predictions(
            self._strategy, self._gen_model, self._gen_tfds, self._custom_dataset,
            map_func=self._postprocess_fn)
        elapsed = time.time() - start_time
        elapsed_from_start = time.time() - self._gen_start_time

        def _display_hypo(custom_ds_sources, custom_ds_targets, hypos, name=None):
            if name:
                logging.info(f"===== Generation examples from {name} (Total {len(hypos)}) =====")
            else:
                logging.info(f"===== Generation examples (Total {len(hypos)}) =====")
            for sample_idx in random.sample(list(range(0, len(hypos))), 5):
                logging.info("Sample %d", sample_idx)
                if custom_ds_sources is not None:
                    logging.info("  Data: %s", custom_ds_sources[sample_idx])
                logging.info("  Reference: %s", custom_ds_targets[sample_idx])
                logging.info("  Hypothesis: %s", hypos[sample_idx])

        def _display(res, best, name=None, tb_name=None):
            if tb_name is None:
                tb_name = name
            tb_name = "" if tb_name is None else (tb_name + "_")
            name = "" if name is None else f" ({name})"

            for k, v in res.items():
                logging.info("Evaluating (%s) validation set%s: %s=%.2f (Best %.2f)  "
                             "step=%d\tElapsed %.2fs  FromSTART %.2fs",
                             self._gen_metric.flag, name, k, v, best[k],
                             step, elapsed, elapsed_from_start)
                tf.summary.scalar(
                    compat.GlobalKeys.TBPREFIX_VALID + f"/{tb_name}{k}", v, step=step)

        if isinstance(self._custom_dataset, MultipleDataset):
            on_average = {}
            mixed_dsnames = []
            mixed_hypos = []
            mixed_refs = []
            sample_weights = {name: self._custom_dataset.sample_weights[name] for name in self._gen_tfds}
            sample_weight_sum = sum(sample_weights.values()) * 1.
            sample_weights = {name: weight / sample_weight_sum for name, weight in sample_weights.items()}
            for name, res in results.items():
                metric_res = self._gen_metric(res, self._custom_ds_targets[name])
                self._gen_recorder[name].record(step, metric_res)
                for k, v in metric_res.items():
                    if k not in on_average:
                        on_average[k] = 0.
                    on_average[k] += sample_weights[name] * v
                _display_hypo(self._custom_ds_sources.get("name", None),
                              self._custom_ds_targets[name], res, name=name)
                _display(metric_res, self._gen_recorder[name].best, name=name)
                mixed_dsnames.append(name)
                mixed_hypos.extend(res)
                mixed_refs.extend(self._custom_ds_targets[name])
            if len(mixed_dsnames) >= 1:
                self._avg_gen_recorder.record(step, on_average)
                if len(mixed_dsnames) > 1:
                    _display(on_average, self._avg_gen_recorder.best,
                             f"on average by weights {sample_weights}", tb_name="AVERAGE")
                    mixed_metric_result = self._gen_metric(mixed_hypos, mixed_refs)
                    self._mixed_gen_recorder.record(step, mixed_metric_result)
                    _display(mixed_metric_result, self._mixed_gen_recorder.best,
                             "mixed of {}".format(",".join(mixed_dsnames)), tb_name="MIXED")
        else:
            metric_res = self._gen_metric(results, self._custom_ds_targets)
            _display_hypo(self._custom_ds_sources, self._custom_ds_targets, results)
            self._gen_recorder.record(step, metric_res)
            _display(metric_res, self._gen_recorder.best)
