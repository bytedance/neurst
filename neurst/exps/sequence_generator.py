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
import json
import time

import numpy
import tensorflow as tf
from absl import logging

from neurst.data.datasets.multiple_dataset import MultipleDataset
from neurst.data.datasets.text_gen_dataset import TextGenDataset
from neurst.exps import BaseExperiment, register_exp
from neurst.layers.search import SequenceSearch, build_search_layer
from neurst.metrics import Metric
from neurst.models.encoder_decoder_ensemble_model import EncoderDecoderEnsembleModel
from neurst.models.model_utils import summary_model_variables
from neurst.training import training_utils
from neurst.utils import compat
from neurst.utils.checkpoints import restore_checkpoint_if_possible
from neurst.utils.configurable import ModelConfigs
from neurst.utils.flags_core import Flag, ModuleFlag
from neurst.utils.misc import flatten_string_list, to_numpy_or_python_type


@register_exp(["predict", "generation"])
class SequenceGenerator(BaseExperiment):
    """ Entry for sequence generation. """

    def __init__(self, args, **kwargs):
        """ Initializes a util class for sequence generation. """
        super(SequenceGenerator, self).__init__(**kwargs)
        self._output_file = args["output_file"]
        self._save_metric = args["save_metric"]
        self._metric = self.task.get_eval_metric(args, ds=self.custom_dataset)
        self._search_layer = build_search_layer(args)

    @staticmethod
    def class_or_method_args():
        return [
            ModuleFlag(Metric.REGISTRY_NAME,
                       help="The evaluation metric for the generation results."),
            ModuleFlag(SequenceSearch.REGISTRY_NAME, help="The search layer for sequence generation."),
            Flag("output_file", dtype=Flag.TYPE.STRING, default=None,
                 help="The path to a file for generated outputs. If MultipleDataset is provided, "
                      "it should be a dict like {dataset_name0: data_path0, ...}"),
            Flag("save_metric", dtype=Flag.TYPE.STRING, default=None,
                 help="The path to a file that metrics will be saved to, in json format."),
        ]

    @staticmethod
    def build_generation_model(task, model, search_layer, output_sequence_only=True):
        """ Build keras model for generation.

        Args:
            task: The task object.
            model: An instance of neurst.models.model.BaseModel
            search_layer: A sequence search object.
            output_sequence_only: Only generated sequences will output if True.

        Returns: the generation model.
        """
        if search_layer is None:
            raise ValueError(
                "The parameters for generation method must be provided: "
                "search_method, search_method.params, ...")
        inps = task.create_inputs(compat.ModeKeys.INFER)
        formatted_inps = task.example_to_input(inps, compat.ModeKeys.INFER)
        search_layer.set_model(model)
        generation_ops = search_layer(formatted_inps)
        if output_sequence_only:
            generation_ops = generation_ops[0]
        keras_model = tf.keras.Model(inps, generation_ops)
        return keras_model

    def _build_and_restore_model(self):
        """ Build a single model or ensemble model. """
        model_dirs = flatten_string_list(self.model_dir)
        if len(model_dirs) == 1:
            model = self.model
            stat = restore_checkpoint_if_possible(model, model_dirs[0])
            if not stat:
                logging.info("WARNING: Fail to restore checkpoint from {}. "
                             "We assume this was done on purpose. ".format(model_dirs[0]))
        else:
            logging.info("We assume models for ensemble are all based on the same task.")
            multiple_models = []
            for idx, one_model_dir in enumerate(model_dirs):
                name_prefix = "ensemble_{}".format(idx)
                logging.info("Create model for {} from {}".format(name_prefix, one_model_dir))
                cfg = ModelConfigs.load(one_model_dir)
                this_model = self.task.build_model(cfg, name=name_prefix)
                stat = restore_checkpoint_if_possible(this_model, one_model_dir)
                if not stat:
                    logging.info("WARNING: Fail to restore checkpoint from {}. "
                                 "We assume this was done on purpose. ".format(one_model_dir))
                multiple_models.append(this_model)
            model = EncoderDecoderEnsembleModel.new(multiple_models)
        return model

    @staticmethod
    def postprocess_generation(task, generations):
        generations = numpy.concatenate(to_numpy_or_python_type(generations), 0)
        postprocess_fn = task.get_data_postprocess_fn(compat.DataStatus.PROJECTED)
        generations = [postprocess_fn(x) for x in generations]
        return generations

    def run(self):
        """ Sequence generation from an existing model checkpoint.

        Step 1: Build model and restore checkpoints.
        Step 2: Build test dataset.
        Step 3: Sequence generation.
        Step 4: Evaluation using metric.
        """
        # Step 3: Build model.
        with training_utils.get_strategy_scope(self.strategy):
            model = self._build_and_restore_model()
            keras_model = self.build_generation_model(self.task, model, self._search_layer)
            tfds = training_utils.build_datasets(compat.ModeKeys.INFER, self.strategy,
                                                 self.custom_dataset, self.task, cache=True)
            keras_model.summary()
            summary_model_variables(keras_model)

        # Step 5: Sequence Generation.
        start_time = time.time()
        results = training_utils.make_predictions(
            self.strategy, keras_model, tfds, self.custom_dataset,
            map_func=lambda y: SequenceGenerator.postprocess_generation(self.task, y))
        logging.info("Generation elapsed: %.2fs", time.time() - start_time)

        if self._output_file:
            if isinstance(self.custom_dataset, MultipleDataset):
                if isinstance(self._output_file, dict):
                    for name in results:
                        if self._output_file.get(name, None):
                            with tf.io.gfile.GFile(self._output_file[name], "w") as fw:
                                fw.write("\n".join(results[name]) + "\n")
                            logging.info("Saving generation of dataset {} results into {}".format(
                                name, self._output_file[name]))
                else:
                    logging.info("Unsupported type of `output_file`={}({}) for MultipleDataset.".format(
                        self._output_file, type(self._output_file)))
            else:
                if isinstance(self._output_file, str):
                    with tf.io.gfile.GFile(self._output_file, "w") as fw:
                        fw.write("\n".join(results) + "\n")
                    logging.info("Saving generation results into {}".format(self._output_file))
                else:
                    logging.info(f"WARNING: No generation results are saved due to unsupported type "
                                 f"of `output_file`: {self._output_file} ({type(self._output_file)})")

        # Step 6: evaluation using metric
        def _display(res, name=None):
            if name:
                logging.info(f"Evaluation Result ({name}):")
            else:
                logging.info("Evaluation Result:")
            for k, v in res.items():
                logging.info("   %s=%.2f", k, v)

        if self._metric is not None:
            saving_metrics = dict()
            if isinstance(self.custom_dataset, MultipleDataset):
                on_average = {}
                mixed_dsnames = []
                mixed_hypos = []
                mixed_refs = []
                for name in tfds:
                    assert isinstance(self.custom_dataset.datasets[name], TextGenDataset)
                    if (hasattr(self.custom_dataset.datasets[name], "raw_targets")
                        and self.custom_dataset.datasets[name].raw_targets):
                        targets = self.custom_dataset.datasets[name].raw_targets
                    else:
                        targets = self.custom_dataset.datasets[name].targets
                    if targets:
                        metric_result = self._metric(results[name], targets)
                        for k, v in metric_result.items():
                            if k not in on_average:
                                on_average[k] = 0.
                            on_average[k] += self.custom_dataset.sample_weights[name] * v
                        _display(metric_result, name)
                        mixed_dsnames.append(name)
                        mixed_hypos.extend(results[name])
                        mixed_refs.extend(targets)
                        saving_metrics[name] = metric_result
                if len(mixed_dsnames) > 1:
                    _display(on_average, f"on average by weights {self._custom_dataset.sample_weights}")
                    mixed_metric_result = self._metric(mixed_hypos, mixed_refs)
                    _display(mixed_metric_result, "mixed of {}".format(",".join(mixed_dsnames)))
                    saving_metrics["MIXED"] = mixed_metric_result

            else:
                assert isinstance(self.custom_dataset, TextGenDataset)
                if hasattr(self.custom_dataset, "raw_targets") and self.custom_dataset.raw_targets:
                    targets = self.custom_dataset.raw_targets
                else:
                    targets = self.custom_dataset.targets
                if targets:
                    metric_result = self._metric(results, targets)
                    _display(metric_result)
                    saving_metrics = metric_result
            if self._save_metric is not None:
                logging.info(f"Saving metric results into {self._save_metric}")
                with tf.io.gfile.GFile(self._save_metric, "w") as fw:
                    json.dump(saving_metrics, fw)
