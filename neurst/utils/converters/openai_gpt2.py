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
import re

import numpy
import tensorflow as tf
import yaml
from absl import logging

from neurst.models.gpt2 import GPT2
from neurst.utils.converters import Converter, register_converter
from neurst.utils.misc import download_with_tqdm

# _URL_PREFIX = "https://storage.googleapis.com/gpt-2"
_URL_PREFIX = "http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/resources/gpt-2"
_GPT2_PRETRAIN_MODELS = {
    "117M": _URL_PREFIX + "/models/117M",
    "345M": _URL_PREFIX + "/models/345M",
}

_DIRECT_MAPPINGS = {
    "model/wte": "gpt2/posenc_wrapper/embeddings/shared/weights",
    "model/wpe": "gpt2/posenc_wrapper/position_embeddings/weights",
    "model/ln_f/b": "gpt2/decoder/output_ln/beta",
    "model/ln_f/g": "gpt2/decoder/output_ln/gamma",
}

_POSTFIX_MAPPINGS = {
    "attn/c_attn/w": "self_attention_prepost_wrapper/self_attention/qkv_transform/kernel",
    "attn/c_attn/b": "self_attention_prepost_wrapper/self_attention/qkv_transform/bias",
    "attn/c_proj/w": "self_attention_prepost_wrapper/self_attention/output_transform/kernel",
    "attn/c_proj/b": "self_attention_prepost_wrapper/self_attention/output_transform/bias",
    "ln_1/b": "self_attention_prepost_wrapper/ln/beta",
    "ln_1/g": "self_attention_prepost_wrapper/ln/gamma",
    "ln_2/b": "ffn_prepost_wrapper/ln/beta",
    "ln_2/g": "ffn_prepost_wrapper/ln/gamma",
    "mlp/c_fc/w": "ffn_prepost_wrapper/ffn/dense1/kernel",
    "mlp/c_fc/b": "ffn_prepost_wrapper/ffn/dense1/bias",
    "mlp/c_proj/w": "ffn_prepost_wrapper/ffn/dense2/kernel",
    "mlp/c_proj/b": "ffn_prepost_wrapper/ffn/dense2/bias"
}


@register_converter(["openai_gpt2", "gpt2"])
class OpenAIGPT2(Converter):
    """ Converts from the from openai gpt-2.
    https://github.com/openai/gpt-2
    """

    @staticmethod
    def convert_model_config(path):
        with tf.io.gfile.GFile(os.path.join(path, "hparams.json")) as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)
        return {
            "model.class": GPT2.__name__,
            "model.params": {
                "timing": {
                    "timing": "emb",
                    "max_positions": cfg["n_ctx"],
                },
                "num_layers": cfg["n_layer"],
                "hidden_size": cfg["n_embd"],
                "num_attention_heads": cfg["n_head"]
            }
        }

    @staticmethod
    def convert_task_config(path):
        raise NotImplementedError

    @staticmethod
    def download(key):
        if key in _GPT2_PRETRAIN_MODELS:
            url = _GPT2_PRETRAIN_MODELS[key]
        elif key.startswith("http://") or key.startswith("https://"):
            url = key
        else:
            return None
        logging.info(f"Downloading openai gpt2: {key}")
        this_dir = os.path.dirname(__file__)
        model_dir = os.path.join(this_dir, f"GPT2_{key}")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        for filename in ['checkpoint', 'encoder.json', 'hparams.json',
                         'model.ckpt.data-00000-of-00001',
                         'model.ckpt.index', 'model.ckpt.meta', 'vocab.bpe']:
            this_url = url + "/" + filename
            save_filename = os.path.join(model_dir, filename)
            if not os.path.exists(save_filename):
                logging.info(f"Downloading {this_url}")
                download_with_tqdm(this_url, save_filename)
        return model_dir

    @staticmethod
    def convert_checkpoint(path, save_path):
        ckpt = os.path.join(path, "model.ckpt")
        gpt2_var_names = [x[0] for x in tf.train.list_variables(ckpt)]
        new_vars = []
        processed_var_names = []
        for var_name in gpt2_var_names:
            var_value = numpy.squeeze(tf.train.load_variable(ckpt, var_name))
            new_var_name = None
            if var_name in _DIRECT_MAPPINGS:
                new_var_name = _DIRECT_MAPPINGS[var_name]
            elif var_name.startswith("model/h"):
                lid = re.search(r"h\d+", var_name).group()[1:]
                postfix = var_name.split(f"h{lid}/")[1]
                if postfix in _POSTFIX_MAPPINGS:
                    new_var_name = f"gpt2/decoder/layer_{lid}/{_POSTFIX_MAPPINGS[postfix]}"
                else:
                    raise Exception("wrong")

            if new_var_name:
                processed_var_names.append(new_var_name)
                logging.info(f"convert {var_name}")
                logging.info(f"\t ==> {new_var_name}")
                new_vars.append(tf.Variable(
                    initial_value=var_value,
                    trainable=False,
                    name=new_var_name,
                    dtype=str(var_value.dtype)))
            else:
                logging.info(f"No matching variable for {var_name}")
        ckpt_saver = tf.train.Checkpoint(**{x.name.split(":")[0]: x for x in new_vars})
        save_ckpt = os.path.join(save_path, "ckpt")
        logging.info(f"Saving checkpoint to {save_ckpt}")
        ckpt_saver.save(save_ckpt)
