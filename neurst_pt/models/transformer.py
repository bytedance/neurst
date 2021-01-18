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
import re

import torch

from neurst.utils.flags_core import Flag
from neurst_pt.layers.decoders import build_decoder
from neurst_pt.layers.encoders import build_encoder
from neurst_pt.models import register_model
from neurst_pt.models.encoder_decoder_model import EncoderDecoderModel


@register_model("transformer")
class Transformer(EncoderDecoderModel):
    """ Defines the Transformer model. """

    def __init__(self, args, *largs, **kwargs):
        """ Initializes a sequence to sequence model.

        Args:
            args: A dict, containing the model configuration.
            src_meta: A dict containing source-side vocabulary meta data, e.g. eos_id, unk_id.
            trg_meta: A dict containing target-side vocabulary meta data, e.g. eos_id, unk_id.
            src_modality: The source side modality.
            trg_modality: The target side modality.
            encoder: The encoder.
            decoder: The decoder.
            name: The name of the model.options = tf.data.Options()
        """
        super(Transformer, self).__init__(args, *largs, **kwargs)
        self._args = args

    @staticmethod
    def class_or_method_args():
        this_args = [x for x in super(Transformer, Transformer).class_or_method_args() if x.name not in [
            "encoder.class", "decoder.class", "encoder.params", "decoder.params"]]
        this_args += [
            Flag("encoder.num_layers", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The number of stacking layers of the encoder."),
            Flag("encoder.hidden_size", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The number of hidden units of the encoder."),
            Flag("encoder.num_attention_heads", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The number of heads of encoder self-attention."),
            Flag("encoder.filter_size", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The number of the filter size of encoder ffn."),
            Flag("encoder.ffn_activation", dtype=Flag.TYPE.STRING, default="relu",
                 help="The activation function of encoder ffn layer."),
            Flag("encoder.attention_dropout_rate", dtype=Flag.TYPE.FLOAT, default=0.,
                 help="The dropout rate of encoder self-attention layer."),
            Flag("encoder.attention_type", dtype=Flag.TYPE.STRING, default="dot_product",
                 help="The type of the attention function of encoder self-attention layer."),
            Flag("encoder.ffn_dropout_rate", dtype=Flag.TYPE.FLOAT, default=0.,
                 help="The dropout rate of encoder ffn layer."),
            Flag("encoder.layer_postprocess_dropout_rate", dtype=Flag.TYPE.FLOAT, default=0.,
                 help="The dropout rate for each layer's post process in encoder."),
            Flag("encoder.post_normalize", dtype=Flag.TYPE.BOOLEAN, default=False,
                 help="Whether to apply layer norm after each encoder block."),
            Flag("encoder.layer_postprocess_epsilon", dtype=Flag.TYPE.FLOAT, default=1e-6,
                 help="The epsilon for layer normalization in encoder."),
            Flag("decoder.num_layers", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The number of stacking layers of the decoder."),
            Flag("decoder.hidden_size", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The number of hidden units of the decoder."),
            Flag("decoder.num_attention_heads", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The number of heads of decoder self-attention and encoder-decoder attention."),
            Flag("decoder.filter_size", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The number of the filter size of decoder ffn."),
            Flag("decoder.ffn_activation", dtype=Flag.TYPE.STRING, default="relu",
                 help="The activation function of decoder ffn layer."),
            Flag("decoder.attention_dropout_rate", dtype=Flag.TYPE.FLOAT, default=0.,
                 help="The dropout rate of decoder self-attention and encoder-decoder attention."),
            Flag("decoder.attention_type", dtype=Flag.TYPE.STRING, default="dot_product",
                 help="The type of the attention function of decoder self-attention and encoder-decoder attention."),
            Flag("decoder.ffn_dropout_rate", dtype=Flag.TYPE.FLOAT, default=0.,
                 help="The dropout rate of decoder ffn layer."),
            Flag("decoder.layer_postprocess_dropout_rate", dtype=Flag.TYPE.FLOAT, default=0.,
                 help="The dropout rate for each layer's post process in decoder."),
            Flag("decoder.post_normalize", dtype=Flag.TYPE.BOOLEAN, default=False,
                 help="Whether to apply layer norm after each decoder block."),
            Flag("decoder.layer_postprocess_epsilon", dtype=Flag.TYPE.FLOAT, default=1e-6,
                 help="The epsilon for layer normalization in decoder."),
        ]
        return this_args

    @classmethod
    def new(cls, args, src_meta, trg_meta):
        """ Builds a sequence to sequence model.

        Args:
            args: A dict containing all model parameters.
            src_meta: A dict containing source-side vocabulary meta data, e.g. eos_id, vocab_size.
            trg_meta: A dict containing target-side vocabulary meta data, e.g. eos_id, vocab_size.
            name: The name of the model.

        Returns:
            An encoder decoder model.
        """
        # build source and target modality
        src_modality, trg_modality = cls.build_modalities(args, src_meta, trg_meta)
        encoder_params, decoder_params = {}, {}
        for f in cls.class_or_method_args():
            if f.name in args:
                if f.name.startswith("encoder."):
                    encoder_params[f.name[8:]] = args[f.name]
                elif f.name.startswith("decoder."):
                    decoder_params[f.name[8:]] = args[f.name]
        # build encoder and decoder
        encoder = build_encoder({
            "encoder.class": "TransformerEncoder",
            "encoder.params": encoder_params})
        decoder = build_decoder({
            "decoder.class": "TransformerDecoder",
            "decoder.params": decoder_params})
        model = cls(args, src_meta, trg_meta, src_modality, trg_modality,
                    encoder, decoder)
        _ = model({"src": torch.LongTensor([[1, 2, 3]]),
                   "src_padding": torch.FloatTensor([[0, 0., 0]]),
                   "trg_input": torch.LongTensor([[1, 2, 3]])})
        return model

    @classmethod
    def build_model_args_by_name(cls, name):
        if name == "transformer_toy":
            dmodel = 8
            num_heads = 2
            num_encoder_layers = 2
            num_decoder_layers = 2
            num_encoder_filter_size = 10
            num_decoder_filter_size = 10
            dropout_rate = 0.1
        elif name == "transformer_base":
            dmodel = 512
            num_heads = 8
            num_encoder_layers = 6
            num_decoder_layers = 6
            num_encoder_filter_size = 2048
            num_decoder_filter_size = 2048
            dropout_rate = 0.1
        elif name == "transformer_s":
            dmodel = 256
            num_heads = 4
            num_encoder_layers = 6
            num_decoder_layers = 6
            num_encoder_filter_size = 2048
            num_decoder_filter_size = 2048
            dropout_rate = 0.1
        elif name == "transformer_big":
            dmodel = 1024
            num_heads = 16
            num_encoder_layers = 6
            num_decoder_layers = 6
            num_encoder_filter_size = 4096
            num_decoder_filter_size = 4096
            dropout_rate = 0.3
        elif name == "transformer_big_dp01":
            dmodel = 1024
            num_heads = 16
            num_encoder_layers = 6
            num_decoder_layers = 6
            num_encoder_filter_size = 4096
            num_decoder_filter_size = 4096
            dropout_rate = 0.1
        elif re.match(r"^transformer_\d+_\d+e_\d+d(_\d+h)?(_dp0\.\d+)?$", name):
            eles = name.split("_")
            dmodel = int(eles[1])
            num_encoder_layers = int(eles[2][:-1])
            num_decoder_layers = int(eles[3][:-1])
            num_heads = 8
            this_idx = 4
            if len(eles) > this_idx:
                if eles[this_idx][-1] == "h":
                    num_heads = int(eles[this_idx][:-1])
                    this_idx += 1
            assert dmodel % num_heads == 0, (
                "Invalid arguments in hparams_set: "
                "dimension({}) must be divisible by head({})".format(dmodel, num_heads))
            dropout_rate = 0.1
            if len(eles) > this_idx:
                if eles[this_idx][0:2] == "dp":
                    dropout_rate = float(eles[this_idx][2:])
            num_encoder_filter_size = dmodel * 4
            num_decoder_filter_size = dmodel * 4
        else:
            return None
        return {
            "model.class": cls.__name__,
            "model.params": {
                "modality.share_source_target_embedding": False,
                "modality.share_embedding_and_softmax_weights": True,
                "modality.dim": dmodel,
                "modality.timing": "sinusoids",
                "encoder.num_layers": num_encoder_layers,
                "encoder.hidden_size": dmodel,
                "encoder.num_attention_heads": num_heads,
                "encoder.filter_size": num_encoder_filter_size,
                "encoder.attention_dropout_rate": dropout_rate,
                "encoder.attention_type": "dot_product",
                "encoder.ffn_activation": "relu",
                "encoder.ffn_dropout_rate": dropout_rate,
                "encoder.post_normalize": False,
                "encoder.layer_postprocess_dropout_rate": dropout_rate,
                "decoder.num_layers": num_decoder_layers,
                "decoder.hidden_size": dmodel,
                "decoder.num_attention_heads": num_heads,
                "decoder.filter_size": num_decoder_filter_size,
                "decoder.attention_dropout_rate": dropout_rate,
                "decoder.attention_type": "dot_product",
                "decoder.ffn_activation": "relu",
                "decoder.ffn_dropout_rate": dropout_rate,
                "decoder.post_normalize": False,
                "decoder.layer_postprocess_dropout_rate": dropout_rate
            },
            "optimizer.class": "Adam",
            "optimizer.params": {
                "epsilon": 1.e-9,
                "beta_1": 0.9,
                "beta_2": 0.98
            },
            "lr_schedule.class": "noam",
            "lr_schedule.params": {
                "initial_factor": 1.0,
                "dmodel": dmodel,
                "warmup_steps": 4000
            },
        }
