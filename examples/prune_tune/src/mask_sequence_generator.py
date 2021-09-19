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
import pickle

import tensorflow as tf
import tensorflow.keras.backend as K
from absl import logging

from neurst.exps import register_exp
from neurst.exps.sequence_generator import SequenceGenerator
from neurst.utils import compat
from neurst.utils.flags_core import Flag


@register_exp(["mask_predict", "mask_generation"])
class MaskSequenceGenerator(SequenceGenerator):
    """ Entry for sequence generation. """

    def __init__(self, args, **kwargs):
        """ Initializes a util class for sequence generation. """
        self._loaded_mask = None
        if args["mask_pkl"]:
            logging.info(f"Loading mask from {args['mask_pkl']}")
            with tf.io.gfile.GFile(args["mask_pkl"], 'rb') as f:
                self._loaded_mask = pickle.load(f)
        super(MaskSequenceGenerator, self).__init__(args, **kwargs)

    @staticmethod
    def class_or_method_args():
        this_flags = super(MaskSequenceGenerator, MaskSequenceGenerator).class_or_method_args()
        this_flags.append(Flag("mask_pkl", dtype=Flag.TYPE.STRING, default=None,
                               help="The path to the mask pkl file."), )
        return this_flags

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

    def apply_mask(self, model, masks):
        tuples = []
        for (weight, mask) in list(zip(model.trainable_weights, masks)):
            masked_weight = weight * tf.cast(mask, weight.dtype.base_dtype)
            tuples.append((weight, masked_weight))

        K.batch_set_value(tuples)

    def _build_and_restore_model(self):
        """ Build a single model or ensemble model. """
        model = super(MaskSequenceGenerator, self)._build_and_restore_model()
        if self._loaded_mask is not None:
            self.apply_mask(model, self._loaded_mask)
        return model
