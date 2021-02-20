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
import sys

import tensorflow as tf

from neurst.layers.quantization import QuantLayer
from neurst.models.transformer import Transformer
from neurst.tasks import build_task
from neurst.utils.checkpoints import restore_checkpoint_if_possible
from neurst.utils.configurable import ModelConfigs

model_dir = sys.argv[1]
model_configs = ModelConfigs.load(model_dir)
QuantLayer.global_init(model_configs["enable_quant"], **model_configs["quant_params"])
task = build_task(model_configs)
model: Transformer = task.build_model(model_configs)
restore_checkpoint_if_possible(model, model_dir)

clip_max = model._encoder._stacking_layers[0][1]._layer._conv1.traced["kernel"].clip_max

weight_clip_max = tf.maximum(clip_max, 0.0)
weight_clip_max = tf.cast(weight_clip_max, tf.float32)
bits_tmp = float(2 ** (QuantLayer.quant_bits - 1))
weight_clip_min = -weight_clip_max * bits_tmp / (bits_tmp - 1)

print("The quantized weight of encoder layer0's first ffn")
print(tf.quantization.quantize(model._encoder._stacking_layers[0][1]._layer._conv1.kernel,
                               weight_clip_min, clip_max, tf.qint8))
