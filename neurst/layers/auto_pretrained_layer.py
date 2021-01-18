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


class AutoPretrainedLayer(tf.keras.layers.Layer):

    def __init__(self, pretrained_model_name_or_path, name=None):
        """

        Args:
            pretrained_model_name_or_path:
                (1) a string with the `shortcut name` of a pre-trained model to load from cache
                    or download, e.g.: ``bert-base-uncased``.
                (2) a string with the `identifier name` of a pre-trained model that was user-uploaded
                    to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                (3) a path to a `directory` containing model weights saved using
                    :func:`~transformers.TFPreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                (4) a path or url to a `TF 2.0 checkpoint file` (e.g. `./tf_model/model.ckpt.index`).
                    In the case of a PyTorch checkpoint, ``from_pt`` should be set to True and a configuration
                    object should be provided as ``config`` argument.
            name:
        """
        super(AutoPretrainedLayer, self).__init__(name=name)
        self._pretrained_model_name_or_path = pretrained_model_name_or_path

    def get_config(self):
        return dict(
            pretrained_model_name_or_path=self._pretrained_model_name_or_path,
            name=self.name)

    def build(self, input_shape):
        _ = input_shape
        try:
            from transformers import TFAutoModel
            _ = TFAutoModel
        except ImportError:
            raise ImportError('Please install transformers with: pip3 install transformers')
        self._pretrained_model = TFAutoModel.from_pretrained(self._pretrained_model_name_or_path)
        super(AutoPretrainedLayer, self).build(input_shape)

    def call(self, inputs, is_training=False):
        return self._pretrained_model(inputs, training=is_training)
