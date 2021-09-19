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
import zipfile

import numpy
import tensorflow as tf
import yaml
from absl import logging

from neurst.models.bert import Bert
from neurst.utils.converters import Converter, register_converter
from neurst.utils.misc import download_with_tqdm

# _URL_PREFIX = "https://storage.googleapis.com/bert_models"
_URL_PREFIX = "http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/resources/bert_models"
_BERT_PRETRAIN_MODELS = {
    "bert-base-uncased": _URL_PREFIX + "/2018_10_18/uncased_L-12_H-768_A-12.zip",
    # "bert-base-uncased": _URL_PREFIX + "/2020_02_20/uncased_L-12_H-768_A-12.zip"
    "bert-large-uncased": _URL_PREFIX + "/2018_10_18/uncased_L-24_H-1024_A-16.zip",
    "bert-base-chinese": _URL_PREFIX + "/2018_11_03/chinese_L-12_H-768_A-12.zip",

}

_DIRECT_MAPPINGS = {
    "bert/embeddings/word_embeddings": "bert/bert_embedding/word_embedding",
    "bert/embeddings/token_type_embeddings": "bert/bert_embedding/token_type_embedding",
    "bert/embeddings/position_embeddings": "bert/bert_embedding/position_embedding",
    "bert/pooler/dense/bias": "bert/pooler/bias",
    "bert/pooler/dense/kernel": "bert/pooler/kernel",
    "bert/embeddings/LayerNorm/beta": "bert/bert_embedding/ln/beta",
    "bert/embeddings/LayerNorm/gamma": "bert/bert_embedding/ln/gamma",
}

_POSTFIX_MAPPINGS = {
    "attention/output/dense/kernel": "self_attention_prepost_wrapper/self_attention/output_transform/kernel",
    "attention/output/dense/bias": "self_attention_prepost_wrapper/self_attention/output_transform/bias",
    "attention/output/LayerNorm/beta": "self_attention_prepost_wrapper/ln/beta",
    "attention/output/LayerNorm/gamma": "self_attention_prepost_wrapper/ln/gamma",
    "intermediate/dense/kernel": "ffn_prepost_wrapper/ffn/dense1/kernel",
    "intermediate/dense/bias": "ffn_prepost_wrapper/ffn/dense1/bias",
    "output/dense/kernel": "ffn_prepost_wrapper/ffn/dense2/kernel",
    "output/dense/bias": "ffn_prepost_wrapper/ffn/dense2/bias",
    "output/LayerNorm/beta": "ffn_prepost_wrapper/ln/beta",
    "output/LayerNorm/gamma": "ffn_prepost_wrapper/ln/gamma",
}


@register_converter
class GoogleBert(Converter):
    """ Converts from the google bert.
    https://github.com/google-research/bert
    """

    @staticmethod
    def convert_model_config(path):
        with tf.io.gfile.GFile(os.path.join(path, "bert_config.json")) as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)
        return {
            "model.class": Bert.__name__,
            "model.params": {
                "max_position_embeddings": cfg["max_position_embeddings"],
                "num_layers": cfg["num_hidden_layers"],
                "hidden_size": cfg["hidden_size"],
                "num_attention_heads": cfg["num_attention_heads"],
                "filter_size": cfg["intermediate_size"],
                "ffn_activation": cfg["hidden_act"],
                "attention_dropout_rate": cfg["attention_probs_dropout_prob"],
                "attention_type": "dot_product",
                "ffn_dropout_rate": cfg["hidden_dropout_prob"],
                "layer_postprocess_dropout_rate": cfg["hidden_dropout_prob"]
            }
        }

    @staticmethod
    def convert_task_config(path):
        raise NotImplementedError

    @staticmethod
    def download(key):
        if key in _BERT_PRETRAIN_MODELS:
            url = _BERT_PRETRAIN_MODELS[key]
        elif key.startswith("http://") or key.startswith("https://"):
            url = key
        else:
            return None
        bert_name = os.path.basename(url).split(".")[0]
        this_dir = os.path.dirname(__file__)
        extract_path = os.path.join(this_dir, bert_name)
        if not os.path.exists(extract_path):
            logging.info(f"Downloading google bert: {key}")
            tarball = os.path.join(this_dir, os.path.basename(url))
            download_with_tqdm(url, tarball)
            tf.io.gfile.makedirs(extract_path)
            with zipfile.ZipFile(tarball) as zip_ref:
                zip_ref.extractall(extract_path)
            os.remove(tarball)
        if os.path.isdir(os.path.join(extract_path, bert_name)):
            return os.path.join(extract_path, bert_name)
        return extract_path

    @staticmethod
    def convert_checkpoint(path, save_path):
        ckpt = os.path.join(path, "bert_model.ckpt")
        bert_var_names = [x[0] for x in tf.train.list_variables(ckpt)]
        new_vars = []
        processed_var_names = []
        for var_name in bert_var_names:
            var_value = tf.train.load_variable(ckpt, var_name)
            new_var_name = None
            if var_name in _DIRECT_MAPPINGS:
                new_var_name = _DIRECT_MAPPINGS[var_name]
            elif var_name.startswith("bert/encoder/layer_"):
                lid = re.search(r"layer_\d+", var_name).group().split("_")[-1]
                postfix = var_name.split(f"layer_{lid}/")[1]
                if postfix in _POSTFIX_MAPPINGS:
                    new_var_name = f"bert/encoder/layer_{lid}/{_POSTFIX_MAPPINGS[postfix]}"
                elif (postfix.startswith("attention/self/key")
                      or postfix.startswith("attention/self/query")
                      or postfix.startswith("attention/self/value")):
                    tensor_name = postfix.split("/")[-1]
                    new_var_name = (f"bert/encoder/layer_{lid}/self_attention_prepost_wrapper"
                                    f"/self_attention/qkv_transform/{tensor_name}")
                    if new_var_name in processed_var_names:
                        continue
                    q_value = tf.train.load_variable(
                        ckpt, f"bert/encoder/layer_{lid}/attention/self/query/{tensor_name}")
                    k_value = tf.train.load_variable(
                        ckpt, f"bert/encoder/layer_{lid}/attention/self/key/{tensor_name}")
                    v_value = tf.train.load_variable(
                        ckpt, f"bert/encoder/layer_{lid}/attention/self/value/{tensor_name}")
                    var_value = numpy.concatenate([q_value, k_value, v_value], axis=-1)

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
