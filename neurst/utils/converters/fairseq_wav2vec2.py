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

import numpy
import tensorflow as tf
from absl import logging

from neurst.models.wav2vec2 import Wav2Vec2
from neurst.utils.converters import Converter, register_converter
from neurst.utils.misc import download_with_tqdm

# _URL_PREFIX = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/"
_URL_PREFIX = "http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/resources/fairseq_wav2vec2/"
_W2V_PRETRAIN_MODELS = {
    "wav2vec2_base": _URL_PREFIX + "wav2vec_small.pt",
    "wav2vec2_large": _URL_PREFIX + "libri960_big.pt",
}


@register_converter(["fairseq_wav2vec2", "wav2vec2"])
class FairseqWav2vec2(Converter):
    """ Converts from the fairseq's wav2vec2.0.
    https://github.com/pytorch/fairseq/tree/master/examples/wav2vec

    For fine-tuning usage, not pretraining.
    """

    @staticmethod
    def convert_model_config(path):
        import torch
        with tf.io.gfile.GFile(path, "rb") as fp:
            cp = torch.load(fp, map_location=torch.device('cpu'))
        args = cp["args"].__dict__
        activation_fn = args.get("activation_fn", "gelu")
        return {
            "model.class": Wav2Vec2.__name__,
            "model.params": {
                "conv_bias": args.get("conv_bias", False),
                "conv_feature_layers": args.get("conv_feature_layers",
                                                "[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2"),
                "extractor_mode": args.get("extractor_mode", "default"),
                "encoder_embed_dim": args["encoder_embed_dim"],
                "conv_pos": args.get("conv_pos", 128),
                "conv_pos_groups": args.get("conv_pos_groups", 16),
                "dropout_input": args["dropout_input"],
                "encoder_layerdrop": args.get("encoder_layerdrop", 0.0),
                "encoder_dropout": args["dropout"],
                "encoder_layers": args["encoder_layers"],
                "encoder_attention_heads": args["encoder_attention_heads"],
                "encoder_filter_size": args["encoder_ffn_embed_dim"],
                "encoder_activation_fn": "gelu_nonapprox" if activation_fn == "gelu" else activation_fn,
                "encoder_pre_norm": args.get("layer_norm_first", False)
            }
        }

    @staticmethod
    def convert_task_config(path):
        raise NotImplementedError

    @staticmethod
    def download(key):
        if key in _W2V_PRETRAIN_MODELS:
            url = _W2V_PRETRAIN_MODELS[key]
        elif key.startswith("http://") or key.startswith("https://"):
            url = key
        else:
            return None
        model_file_name = os.path.basename(url)
        this_dir = os.path.dirname(__file__)
        extract_path = os.path.join(this_dir, model_file_name)
        if not os.path.exists(extract_path):
            logging.info(f"Downloading Wav2vec2.0: {key}")
            download_with_tqdm(url, extract_path)

        return extract_path

    @staticmethod
    def convert_checkpoint(path, save_path):
        import torch
        with tf.io.gfile.GFile(path, "rb") as fp:
            pyvars = torch.load(fp, map_location=torch.device('cpu'))["model"]
        cfgs: dict = FairseqWav2vec2.convert_model_config(path)["model.params"]
        py_model_prefix = ""
        for n in pyvars:
            if n.startswith("w2v_encoder.w2v_model."):
                py_model_prefix = "w2v_encoder.w2v_model."
                break
        # ========= feature extractor ========
        max_feconv_layers = 0
        fe_pyprefix = py_model_prefix + "feature_extractor.conv_layers"
        fe_tfprefix = "wav2vec2/feature_extractor"
        new_var_dict = dict()
        _name = f"{fe_pyprefix}.{max_feconv_layers}.0.weight"
        while _name in pyvars:
            _tfname = f"{fe_tfprefix}/conv_block{max_feconv_layers}/conv/kernel"
            new_var_dict[_tfname] = pyvars[_name].detach().numpy().transpose([2, 1, 0])
            max_feconv_layers += 1
            _name = f"{fe_pyprefix}.{max_feconv_layers}.0.weight"
        if cfgs["extractor_mode"] == "default":
            _name = f"{fe_pyprefix}.0.2."
            _tfname = f"{fe_tfprefix}/conv_block0/gn/"
            new_var_dict[_tfname + "gamma"] = pyvars[_name + "weight"].detach().numpy()
            new_var_dict[_tfname + "beta"] = pyvars[_name + "bias"].detach().numpy()
        else:
            raise NotImplementedError
        if cfgs["conv_bias"]:
            raise NotImplementedError
        _name = "wav2vec2/feature_extractor_norm/"
        new_var_dict[_name + "gamma"] = pyvars[py_model_prefix + "layer_norm.weight"].detach().numpy()
        new_var_dict[_name + "beta"] = pyvars[py_model_prefix + "layer_norm.bias"].detach().numpy()
        # post extract projection
        _name = "wav2vec2/post_extract_proj/"
        new_var_dict[_name + "kernel"] = pyvars[
            py_model_prefix + "post_extract_proj.weight"].detach().numpy().transpose()
        new_var_dict[_name + "bias"] = pyvars[py_model_prefix + "post_extract_proj.bias"].detach().numpy()
        # position conv
        _name = "wav2vec2/pos_conv/"
        new_var_dict[_name + "wn/g"] = pyvars[
            py_model_prefix + "encoder.pos_conv.0.weight_g"].detach().numpy()[0][0]
        new_var_dict[_name + "wn/kernel"] = pyvars[
            py_model_prefix + "encoder.pos_conv.0.weight_v"].detach().numpy().transpose([2, 1, 0])
        new_var_dict[_name + "wn/bias"] = pyvars[py_model_prefix + "encoder.pos_conv.0.bias"].detach().numpy()
        new_var_dict[_name + "ln/gamma"] = pyvars[py_model_prefix + "encoder.layer_norm.weight"].detach().numpy()
        new_var_dict[_name + "ln/beta"] = pyvars[py_model_prefix + "encoder.layer_norm.bias"].detach().numpy()
        # encoder
        for i in range(cfgs["encoder_layers"]):
            _name = f"wav2vec2/encoder/layer_{i}/"
            _pyname = py_model_prefix + f"encoder.layers.{i}."
            _selfatt_name = _name + "self_attention_prepost_wrapper/self_attention/"
            new_var_dict[_selfatt_name + "qkv_transform/kernel"] = numpy.concatenate([
                pyvars[_pyname + "self_attn.q_proj.weight"].detach().numpy().transpose(),
                pyvars[_pyname + "self_attn.k_proj.weight"].detach().numpy().transpose(),
                pyvars[_pyname + "self_attn.v_proj.weight"].detach().numpy().transpose()], axis=1)
            new_var_dict[_selfatt_name + "qkv_transform/bias"] = numpy.concatenate([
                pyvars[_pyname + "self_attn.q_proj.bias"].detach().numpy(),
                pyvars[_pyname + "self_attn.k_proj.bias"].detach().numpy(),
                pyvars[_pyname + "self_attn.v_proj.bias"].detach().numpy()], axis=0)
            new_var_dict[_selfatt_name + "output_transform/kernel"] = pyvars[
                _pyname + "self_attn.out_proj.weight"].detach().numpy().transpose()
            new_var_dict[_selfatt_name + "output_transform/bias"] = pyvars[
                _pyname + "self_attn.out_proj.bias"].detach().numpy()
            new_var_dict[_name + "self_attention_prepost_wrapper/ln/gamma"] = pyvars[
                _pyname + "self_attn_layer_norm.weight"].detach().numpy()
            new_var_dict[_name + "self_attention_prepost_wrapper/ln/beta"] = pyvars[
                _pyname + "self_attn_layer_norm.bias"].detach().numpy()
            _ffn_name = _name + "ffn_prepost_wrapper/ffn/"
            new_var_dict[_ffn_name + "dense1/kernel"] = pyvars[_pyname + "fc1.weight"].detach().numpy().transpose()
            new_var_dict[_ffn_name + "dense1/bias"] = pyvars[_pyname + "fc1.bias"].detach().numpy()
            new_var_dict[_ffn_name + "dense2/kernel"] = pyvars[_pyname + "fc2.weight"].detach().numpy().transpose()
            new_var_dict[_ffn_name + "dense2/bias"] = pyvars[_pyname + "fc2.bias"].detach().numpy()
            new_var_dict[_name + "ffn_prepost_wrapper/ln/gamma"] = pyvars[
                _pyname + "final_layer_norm.weight"].detach().numpy()
            new_var_dict[_name + "ffn_prepost_wrapper/ln/beta"] = pyvars[
                _pyname + "final_layer_norm.bias"].detach().numpy()

        ckpt_saver = tf.train.Checkpoint(**{
            name: tf.Variable(initial_value=numpy_var, trainable=False,
                              name=name, dtype="float32")
            for name, numpy_var in new_var_dict.items()})
        save_ckpt = os.path.join(save_path, "ckpt")
        logging.info(f"Saving checkpoint to {save_ckpt}")
        ckpt_saver.save(save_ckpt)
