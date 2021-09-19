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
import tensorflow as tf
from absl import logging

from neurst.data.datasets.multiple_dataset import MultipleDataset
from neurst.exps import register_exp
from neurst.exps.evaluator import Evaluator
from neurst.models.model_utils import summary_model_variables
from neurst.training import training_utils
from neurst.utils import compat
from neurst.utils.checkpoints import restore_checkpoint_if_possible
from neurst.utils.flags_core import Flag


@register_exp(["seq_eval"])
class SequenceEvaluator(Evaluator):
    """ For evaluating on criterion. """

    def __init__(self, args, **kwargs):
        """ Initializes a util class for evaluating neural models. """
        super(SequenceEvaluator, self).__init__(args, **kwargs)
        self._output_file = args["output_file"]
        assert self._output_file, "`output_file` must be provided."

    @staticmethod
    def class_or_method_args():
        this_flags = super(SequenceEvaluator, SequenceEvaluator).class_or_method_args()
        this_flags.append(
            Flag("output_file", dtype=Flag.TYPE.STRING, default=None,
                 help="The output file for evaluation results for each sample."))
        return this_flags

    def run(self):
        """ Evaluation on a existing model.

        Step 1: Build model.
        Step 2: Builds evaluation dataset.
        Step 3: Restore checkpoints.
        Step 4: Evaluate and reduce metric.
        """
        assert not isinstance(self.custom_dataset, MultipleDataset), (
            "SequenceEvaluator only supports single dataset.")
        with training_utils.get_strategy_scope(self.strategy):
            tfds = training_utils.build_datasets(compat.ModeKeys.EVAL, self.strategy,
                                                 self.custom_dataset, self.task)
            keras_model = self.build_evaluation_model(self.task, self.model, self._criterion)
            keras_model.summary()
            summary_model_variables(keras_model)
            # Step 4: Restore checkpoints.
            stat = restore_checkpoint_if_possible(self.model, self.model_dir)
            if not stat:
                logging.info(f"WARNING: Fail to restore checkpoint from {self.model_dir}. "
                             "We assume this was done on purpose. ")
            # Step 5: Evaluate and reduce metric.
            predict_fn = keras_model.make_predict_function()
            iterator = iter(training_utils.maybe_distribution_dataset(
                self.strategy, tfds.prefetch(tf.data.experimental.AUTOTUNE)))
            with tf.io.gfile.GFile(self._output_file, "w") as fw:
                while True:
                    try:
                        preds = predict_fn(iterator)
                        for pred in self._criterion.reduce_sample_metrics(preds):
                            fw.write(str(pred) + "\n")
                    except (StopIteration, tf.errors.OutOfRangeError):
                        break
