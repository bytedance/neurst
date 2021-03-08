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

from neurst.models.transformer import Transformer
from neurst.utils.converters import Converter, register_converter


@register_converter
class FairseqTransformer2(Converter):
    """ Converts from the fairseq's Transformer model.
    There are many versions...
    """

    @staticmethod
    def convert_model_config(path):
        import torch
        with tf.io.gfile.GFile(path, "rb") as fp:
            cp = torch.load(fp, map_location=torch.device('cpu'))
        args = cp["cfg"].__dict__["_content"]["model"]._val.__dict__
        return {
            "model.class": Transformer.__name__,
            "model.params": {
                "modality.share_source_target_embedding": args["share_all_embeddings"],
                "modality.share_embedding_and_softmax_weights": args["share_decoder_input_output_embed"],
                "modality.source.dim": args["encoder_embed_dim"],
                "modality.target.dim": args["decoder_embed_dim"],
                "modality.source.timing": {"timing": "sinusoids",
                                           "max_positions": args["max_source_positions"] - 16,
                                           "sinusoids_as_variable": True},
                "modality.target.timing": {"timing": "sinusoids",
                                           "max_positions": args["max_target_positions"] - 16,
                                           "sinusoids_as_variable": True},
                "encoder.num_layers": args["encoder_layers"],
                "encoder.hidden_size": args["encoder_embed_dim"],
                "encoder.num_attention_heads": args["encoder_attention_heads"],
                "encoder.filter_size": args["encoder_ffn_embed_dim"],
                "encoder.attention_dropout_rate": args["dropout"],
                "encoder.attention_type": "dot_product",
                "encoder.ffn_activation": "relu",
                "encoder.ffn_dropout_rate": args["dropout"],
                "encoder.post_normalize": (not args["encoder_normalize_before"]),
                "encoder.layer_postprocess_dropout_rate": args["dropout"],
                "decoder.num_layers": args["decoder_layers"],
                "decoder.hidden_size": args["encoder_embed_dim"],
                "decoder.num_attention_heads": args["decoder_attention_heads"],
                "decoder.filter_size": args["decoder_ffn_embed_dim"],
                "decoder.attention_dropout_rate": args["dropout"],
                "decoder.attention_type": "dot_product",
                "decoder.ffn_activation": "relu",
                "decoder.ffn_dropout_rate": args["dropout"],
                "decoder.post_normalize": (not args["decoder_normalize_before"]),
                "decoder.layer_postprocess_dropout_rate": args["dropout"]
            }
        }

    @staticmethod
    def convert_task_config(path):
        # return {
        #     "task.params": {
        #         "target_begin_of_sentence": "eos"
        #     }
        # }
        return {}

    @staticmethod
    def convert_checkpoint(path, save_path):
        import torch
        with tf.io.gfile.GFile(path, "rb") as fp:
            pyvars = torch.load(fp, map_location=torch.device('cpu'))["model"]
        cfgs: dict = FairseqTransformer2.convert_model_config(path)["model.params"]
        pyvar_names = [x for x in pyvars]
        new_var_dict = {}

        def get_pyvar(name):
            pyvar_names.remove(name)
            return pyvars[name].detach().numpy()

        def reform_emb(emb):
            # vocab: 0-4  --  bos, pad, eos, unk
            weight = emb[4:]
            bos = emb[0:1]
            eos = emb[2:3]
            unk = emb[3:4]
            return numpy.concatenate([weight, unk, bos, eos], axis=0)

        def get_position_emb(num, dim):
            from fairseq.modules.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
            return SinusoidalPositionalEmbedding.get_embedding(num + 2, dim, 1).detach().numpy()[2:]

        # embedding table
        tf_prefix = "SequenceToSequence"
        if cfgs["modality.share_source_target_embedding"]:
            _name = "shared_symbol_modality"
            if not cfgs["modality.share_embedding_and_softmax_weights"]:
                raise NotImplementedError
            emb_table = reform_emb(get_pyvar("encoder.embed_tokens.weight"))
            new_var_dict[f"{tf_prefix}/{_name}_posenc_wrapper/{_name}/shared/weights"] = emb_table
            new_var_dict[f"{tf_prefix}/{_name}_posenc_wrapper/position_embeddings/weights"] = get_position_emb(
                cfgs["modality.target.timing"]["max_positions"], cfgs["modality.target.dim"])
            new_var_dict[f"{tf_prefix}/{_name}_posenc_wrapper/{_name}/shared/bias"] = numpy.zeros(
                [emb_table.shape[0], ], dtype=float)
        else:
            _src_name = "input_symbol_modality"
            new_var_dict[f"{tf_prefix}/{_src_name}_posenc_wrapper/{_src_name}/emb/weights"] = reform_emb(
                get_pyvar("encoder.embed_tokens.weight"))
            _trg_name = "target_symbol_modality"
            _emb_name = "emb"
            if cfgs["modality.share_embedding_and_softmax_weights"]:
                _emb_name = "shared"
            trg_emb_table = reform_emb(get_pyvar("decoder.embed_tokens.weight"))
            new_var_dict[f"{tf_prefix}/{_trg_name}_posenc_wrapper/{_trg_name}/{_emb_name}/weights"] = trg_emb_table
            new_var_dict[f"{tf_prefix}/{_src_name}_posenc_wrapper/position_embeddings/weights"] = get_position_emb(
                cfgs["modality.source.timing"]["max_positions"], cfgs["modality.source.dim"])
            new_var_dict[f"{tf_prefix}/{_trg_name}_posenc_wrapper/position_embeddings/weights"] = get_position_emb(
                cfgs["modality.target.timing"]["max_positions"], cfgs["modality.target.dim"])
            new_var_dict[f"{tf_prefix}/{_trg_name}_posenc_wrapper/{_trg_name}/{_emb_name}/bias"] = numpy.zeros(
                [trg_emb_table.shape[0], ], dtype=float)
        # encoder stack
        for i in range(cfgs["encoder.num_layers"]):
            _name = f"{tf_prefix}/TransformerEncoder/layer_{i}/"
            _pyname = f"encoder.layers.{i}."
            _selfatt_name = _name + "self_attention_prepost_wrapper/self_attention/"
            # new_var_dict[_selfatt_name + "qkv_transform/kernel"] = get_pyvar(
            #     _pyname + "self_attn.in_proj_weight").transpose()
            # new_var_dict[_selfatt_name + "qkv_transform/bias"] = get_pyvar(_pyname + "self_attn.in_proj_bias")
            new_var_dict[_selfatt_name + "qkv_transform/kernel"] = numpy.concatenate([
                get_pyvar(_pyname + "self_attn.q_proj.weight").transpose(),
                get_pyvar(_pyname + "self_attn.k_proj.weight").transpose(),
                get_pyvar(_pyname + "self_attn.v_proj.weight").transpose()], axis=1)
            new_var_dict[_selfatt_name + "qkv_transform/bias"] = numpy.concatenate([
                get_pyvar(_pyname + "self_attn.q_proj.bias"),
                get_pyvar(_pyname + "self_attn.k_proj.bias"),
                get_pyvar(_pyname + "self_attn.v_proj.bias")], axis=0)
            new_var_dict[_selfatt_name + "output_transform/kernel"] = get_pyvar(
                _pyname + "self_attn.out_proj.weight").transpose()
            new_var_dict[_selfatt_name + "output_transform/bias"] = get_pyvar(_pyname + "self_attn.out_proj.bias")
            new_var_dict[_name + "self_attention_prepost_wrapper/ln/gamma"] = get_pyvar(
                _pyname + "self_attn_layer_norm.weight")
            new_var_dict[_name + "self_attention_prepost_wrapper/ln/beta"] = get_pyvar(
                _pyname + "self_attn_layer_norm.bias")
            _ffn_name = _name + "ffn_prepost_wrapper/ffn/"
            new_var_dict[_ffn_name + "dense1/kernel"] = get_pyvar(_pyname + "fc1.weight").transpose()
            new_var_dict[_ffn_name + "dense1/bias"] = get_pyvar(_pyname + "fc1.bias")
            new_var_dict[_ffn_name + "dense2/kernel"] = get_pyvar(_pyname + "fc2.weight").transpose()
            new_var_dict[_ffn_name + "dense2/bias"] = get_pyvar(_pyname + "fc2.bias")
            new_var_dict[_name + "ffn_prepost_wrapper/ln/gamma"] = get_pyvar(_pyname + "final_layer_norm.weight")
            new_var_dict[_name + "ffn_prepost_wrapper/ln/beta"] = get_pyvar(_pyname + "final_layer_norm.bias")
        if not cfgs["encoder.post_normalize"]:
            new_var_dict[f"{tf_prefix}/TransformerEncoder/output_ln/gamma"] = get_pyvar("encoder.layer_norm.weight")
            new_var_dict[f"{tf_prefix}/TransformerEncoder/output_ln/beta"] = get_pyvar("encoder.layer_norm.bias")

        # decoder stack
        for i in range(cfgs["decoder.num_layers"]):
            _name = f"{tf_prefix}/TransformerDecoder/layer_{i}/"
            _pyname = f"decoder.layers.{i}."
            _selfatt_name = _name + "self_attention_prepost_wrapper/self_attention/"

            new_var_dict[_selfatt_name + "qkv_transform/kernel"] = numpy.concatenate([
                get_pyvar(_pyname + "self_attn.q_proj.weight").transpose(),
                get_pyvar(_pyname + "self_attn.k_proj.weight").transpose(),
                get_pyvar(_pyname + "self_attn.v_proj.weight").transpose()], axis=1)
            new_var_dict[_selfatt_name + "qkv_transform/bias"] = numpy.concatenate([
                get_pyvar(_pyname + "self_attn.q_proj.bias"),
                get_pyvar(_pyname + "self_attn.k_proj.bias"),
                get_pyvar(_pyname + "self_attn.v_proj.bias")], axis=0)
            # new_var_dict[_selfatt_name + "qkv_transform/kernel"] = get_pyvar(
            #     _pyname + "self_attn.in_proj_weight").transpose()
            # new_var_dict[_selfatt_name + "qkv_transform/bias"] = get_pyvar(_pyname + "self_attn.in_proj_bias")
            new_var_dict[_selfatt_name + "output_transform/kernel"] = get_pyvar(
                _pyname + "self_attn.out_proj.weight").transpose()
            new_var_dict[_selfatt_name + "output_transform/bias"] = get_pyvar(_pyname + "self_attn.out_proj.bias")
            new_var_dict[_name + "self_attention_prepost_wrapper/ln/gamma"] = get_pyvar(
                _pyname + "self_attn_layer_norm.weight")
            new_var_dict[_name + "self_attention_prepost_wrapper/ln/beta"] = get_pyvar(
                _pyname + "self_attn_layer_norm.bias")

            _encatt_name = _name + "encdec_attention_prepost_wrapper/encdec_attention/"
            k_w = get_pyvar(_pyname + "encoder_attn.k_proj.weight")
            k_b = get_pyvar(_pyname + "encoder_attn.k_proj.bias")
            v_w = get_pyvar(_pyname + "encoder_attn.v_proj.weight")
            v_b = get_pyvar(_pyname + "encoder_attn.v_proj.bias")
            # qkv_w = get_pyvar(_pyname + "encoder_attn.in_proj_weight")
            # qkv_b = get_pyvar(_pyname + "encoder_attn.in_proj_bias")
            # new_var_dict[_encatt_name + "kv_transform/kernel"] = qkv_w[cfgs["decoder.hidden_size"]:].transpose()
            # new_var_dict[_encatt_name + "kv_transform/bias"] = qkv_b[cfgs["decoder.hidden_size"]:]
            # new_var_dict[_encatt_name + "q_transform/kernel"] = qkv_w[:cfgs["decoder.hidden_size"]].transpose()
            # new_var_dict[_encatt_name + "q_transform/bias"] = qkv_b[:cfgs["decoder.hidden_size"]]
            new_var_dict[_encatt_name + "kv_transform/kernel"] = numpy.concatenate([
                k_w.transpose(), v_w.transpose()], axis=1)
            new_var_dict[_encatt_name + "kv_transform/bias"] = numpy.concatenate([k_b, v_b], axis=0)
            new_var_dict[_encatt_name + "q_transform/kernel"] = get_pyvar(
                _pyname + "encoder_attn.q_proj.weight").transpose()
            new_var_dict[_encatt_name + "q_transform/bias"] = get_pyvar(_pyname + "encoder_attn.q_proj.bias")
            new_var_dict[_encatt_name + "output_transform/kernel"] = get_pyvar(
                _pyname + "encoder_attn.out_proj.weight").transpose()
            new_var_dict[_encatt_name + "output_transform/bias"] = get_pyvar(_pyname + "encoder_attn.out_proj.bias")
            new_var_dict[_name + "encdec_attention_prepost_wrapper/ln/gamma"] = get_pyvar(
                _pyname + "encoder_attn_layer_norm.weight")
            new_var_dict[_name + "encdec_attention_prepost_wrapper/ln/beta"] = get_pyvar(
                _pyname + "encoder_attn_layer_norm.bias")

            _ffn_name = _name + "ffn_prepost_wrapper/ffn/"
            new_var_dict[_ffn_name + "dense1/kernel"] = get_pyvar(_pyname + "fc1.weight").transpose()
            new_var_dict[_ffn_name + "dense1/bias"] = get_pyvar(_pyname + "fc1.bias")
            new_var_dict[_ffn_name + "dense2/kernel"] = get_pyvar(_pyname + "fc2.weight").transpose()
            new_var_dict[_ffn_name + "dense2/bias"] = get_pyvar(_pyname + "fc2.bias")
            new_var_dict[_name + "ffn_prepost_wrapper/ln/gamma"] = get_pyvar(_pyname + "final_layer_norm.weight")
            new_var_dict[_name + "ffn_prepost_wrapper/ln/beta"] = get_pyvar(_pyname + "final_layer_norm.bias")

        if not cfgs["decoder.post_normalize"]:
            new_var_dict[f"{tf_prefix}/TransformerDecoder/output_ln/gamma"] = get_pyvar("decoder.layer_norm.weight")
            new_var_dict[f"{tf_prefix}/TransformerDecoder/output_ln/beta"] = get_pyvar("decoder.layer_norm.bias")

        ckpt_saver = tf.train.Checkpoint(**{
            name: tf.Variable(initial_value=numpy_var, trainable=False,
                              name=name, dtype="float32")
            for name, numpy_var in new_var_dict.items()})
        save_ckpt = os.path.join(save_path, "ckpt")
        logging.info("Unrecognized py varnames: " + str(pyvar_names))
        logging.info(f"Saving checkpoint to {save_ckpt}")
        ckpt_saver.save(save_ckpt)
