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

from neurst.layers.decoders import build_decoder
from neurst.layers.encoders import build_encoder
from neurst.models import register_model
from neurst.models.encoder_decoder_model import EncoderDecoderModel
from neurst.utils.flags_core import Flag
from neurst.utils.hparams_sets import register_hparams_set


@register_model
class LightConvolutionModel(EncoderDecoderModel):
    """ Defines the Transformer model.

    All other encoder-decoder structure should inherit this class.
    """

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
        super(LightConvolutionModel, self).__init__(args, *largs, **kwargs)
        self._args = args

    @staticmethod
    def class_or_method_args():
        this_args = [x for x in super(LightConvolutionModel, LightConvolutionModel).class_or_method_args()
                     if x.name not in ["encoder.class", "decoder.class", "encoder.params", "decoder.params"]]
        this_args += [
            Flag("encoder.num_layers", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The number of stacking layers of the encoder."),
            Flag("encoder.conv_kernel_size_list", dtype=Flag.TYPE.STRING, default=None,
                 help="A list of kernel sizes for each encoder layers. "
                      "The length of the list must be equal to `encoder.num_layers`."),
            Flag("encoder.num_conv_heads", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The number of heads of encoder convolution shared weights."),
            Flag("encoder.conv_hidden_size", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The number of hidden units of the encoder convolution layer."),
            Flag("encoder.conv_type", dtype=Flag.TYPE.STRING, default="lightweight",
                 help="The type of encoder conv layer, one of lightweight or dynamic."),
            Flag("encoder.filter_size", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The number of the filter size of encoder ffn."),
            Flag("encoder.ffn_activation", dtype=Flag.TYPE.STRING, default="relu",
                 help="The activation function of encoder ffn layer."),
            Flag("encoder.conv_weight_dropout_rate", dtype=Flag.TYPE.FLOAT, default=0.,
                 help="The dropout rate of encoder convolution weights."),
            Flag("encoder.glu_after_proj", dtype=Flag.TYPE.BOOLEAN, default=False,
                 help="Whether to apply glu activation after input projection in encoder convolution layer."),
            Flag("encoder.ffn_dropout_rate", dtype=Flag.TYPE.FLOAT, default=0.,
                 help="The dropout rate of encoder ffn layer."),
            Flag("encoder.layer_postprocess_dropout_rate", dtype=Flag.TYPE.FLOAT, default=0.,
                 help="The dropout rate for each layer's post process in encoder."),
            Flag("encoder.layer_postprocess_epsilon", dtype=Flag.TYPE.FLOAT, default=1e-6,
                 help="The epsilon for layer normalization in encoder."),
            Flag("decoder.num_layers", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The number of stacking layers of the decoder."),
            Flag("decoder.conv_kernel_size_list", dtype=Flag.TYPE.STRING, default=None,
                 help="A list of kernel sizes for each decoder layers. "
                      "The length of the list must be equal to `decoder.num_layers`."),
            Flag("decoder.num_conv_heads", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The number of heads of decoder convolution shared weights."),
            Flag("decoder.conv_hidden_size", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The number of hidden units of the decoder convolution layer."),
            Flag("decoder.num_attention_heads", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The number of heads of decoder's encoder-decoder attention."),
            Flag("decoder.conv_type", dtype=Flag.TYPE.STRING, default="lightweight",
                 help="The type of decoder conv layer, one of lightweight or dynamic."),
            Flag("decoder.filter_size", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The number of the filter size of decoder ffn."),
            Flag("decoder.ffn_activation", dtype=Flag.TYPE.STRING, default="relu",
                 help="The activation function of decoder ffn layer."),
            Flag("decoder.attention_dropout_rate", dtype=Flag.TYPE.FLOAT, default=0.,
                 help="The dropout rate of decoder's encoder-decoder attention."),
            Flag("decoder.conv_weight_dropout_rate", dtype=Flag.TYPE.FLOAT, default=0.,
                 help="The dropout rate of decoder convolution weights."),
            Flag("decoder.glu_after_proj", dtype=Flag.TYPE.BOOLEAN, default=False,
                 help="Whether to apply glu activation after input projection in decoder convolution layer."),
            Flag("decoder.attention_type", dtype=Flag.TYPE.STRING, default="dot_product",
                 help="The type of the attention function of decoder's encoder-decoder attention."),
            Flag("decoder.ffn_dropout_rate", dtype=Flag.TYPE.FLOAT, default=0.,
                 help="The dropout rate of decoder ffn layer."),
            Flag("decoder.layer_postprocess_dropout_rate", dtype=Flag.TYPE.FLOAT, default=0.,
                 help="The dropout rate for each layer's post process in decoder."),
            Flag("decoder.layer_postprocess_epsilon", dtype=Flag.TYPE.FLOAT, default=1e-6,
                 help="The epsilon for layer normalization in decoder."),
        ]
        return this_args

    @classmethod
    def new(cls, args, src_meta, trg_meta, name=None):
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
            "encoder.class": "LightConvolutionEncoder",
            "encoder.params": encoder_params})
        decoder = build_decoder({
            "decoder.class": "LightConvolutionDecoder",
            "decoder.params": decoder_params})
        model = cls(args, src_meta, trg_meta, src_modality, trg_modality, encoder, decoder, name=name)
        fake_inputs = {"src": tf.convert_to_tensor([[1, 2, 3]], tf.int64),
                       "src_padding": tf.convert_to_tensor([[0, 0., 0]], tf.float32),
                       "trg_input": tf.convert_to_tensor([[1, 2, 3]], tf.int64), }
        _ = model(fake_inputs)
        return model


def _common_hparams(dmodel, num_heads, filter_size,
                    encoder_layers, encoder_kernels,
                    decoder_layers, decoder_kernels,
                    encoder_conv_type, decoder_conv_type,
                    attention_dropout, weight_dropout, dropout):
    return {
        "model.class": "LightConvolutionModel",
        "model.params": {
            "modality.share_source_target_embedding": False,
            "modality.share_embedding_and_softmax_weights": True,
            "modality.dim": dmodel,
            "modality.timing": "sinusoids",
            "encoder.num_layers": encoder_layers,
            "encoder.conv_kernel_size_list": encoder_kernels,
            "encoder.num_conv_heads": num_heads,
            "encoder.conv_hidden_size": dmodel,
            "encoder.conv_type": encoder_conv_type,
            "encoder.filter_size": filter_size,
            "encoder.glu_after_proj": True,
            "encoder.conv_weight_dropout_rate": weight_dropout,
            "encoder.ffn_activation": "relu",
            "encoder.ffn_dropout_rate": attention_dropout,
            "encoder.layer_postprocess_dropout_rate": dropout,
            "decoder.num_layers": decoder_layers,
            "decoder.conv_kernel_size_list": decoder_kernels,
            "decoder.num_conv_heads": num_heads,
            "decoder.conv_hidden_size": dmodel,
            "decoder.num_attention_heads": num_heads,
            "decoder.conv_type": decoder_conv_type,
            "decoder.filter_size": filter_size,
            "decoder.attention_dropout_rate": attention_dropout,
            "decoder.glu_after_proj": True,
            "decoder.conv_weight_dropout_rate": weight_dropout,
            "decoder.attention_type": "dot_product",
            "decoder.ffn_activation": "relu",
            "decoder.ffn_dropout_rate": attention_dropout,
            "decoder.layer_postprocess_dropout_rate": dropout
        },
        "optimizer.class": "Adam",
        "optimizer.params": {
            "epsilon": 1.e-9,
            "beta_1": 0.9,
            "beta_2": 0.98
        },
        "lr_schedule.class": "noam",
        "lr_schedule.params": {
            "initial_factor": 2.0,
            "dmodel": dmodel,
            "warmup_steps": 10000
        },
    }


@register_hparams_set("lightweight_conv_big")
def lightweight_conv_big():
    return _common_hparams(
        dmodel=1024, num_heads=16, filter_size=4096,
        encoder_layers=7, encoder_kernels=[3, 7, 15, 31, 31, 31, 31],
        decoder_layers=6, decoder_kernels=[3, 7, 15, 31, 31, 31],
        encoder_conv_type="lightweight", decoder_conv_type="lightweight",
        attention_dropout=0.1, weight_dropout=0.1, dropout=0.3)


@register_hparams_set("lightweight_conv_big_dp01")
def lightweight_conv_big_dp01():
    return _common_hparams(
        dmodel=1024, num_heads=16, filter_size=4096,
        encoder_layers=7, encoder_kernels=[3, 7, 15, 31, 31, 31, 31],
        decoder_layers=6, decoder_kernels=[3, 7, 15, 31, 31, 31],
        encoder_conv_type="lightweight", decoder_conv_type="lightweight",
        attention_dropout=0.1, weight_dropout=0.1, dropout=0.1)


@register_hparams_set("lightweight_conv_toy")
def lightweight_conv_toy():
    return _common_hparams(
        dmodel=8, num_heads=4, filter_size=32,
        encoder_layers=2, encoder_kernels=[3, 7],
        decoder_layers=2, decoder_kernels=[3, 5],
        encoder_conv_type="lightweight", decoder_conv_type="lightweight",
        attention_dropout=0.1, weight_dropout=0.1, dropout=0.1)


@register_hparams_set("dynamic_conv_big")
def dynamic_conv_big():
    return _common_hparams(
        dmodel=1024, num_heads=16, filter_size=4096,
        encoder_layers=7, encoder_kernels=[3, 7, 15, 31, 31, 31, 31],
        decoder_layers=6, decoder_kernels=[3, 7, 15, 31, 31, 31],
        encoder_conv_type="dynamic", decoder_conv_type="dynamic",
        attention_dropout=0.1, weight_dropout=0.1, dropout=0.3)


@register_hparams_set("dynamic_conv_big_dp01")
def dynamic_conv_big_dp01():
    return _common_hparams(
        dmodel=1024, num_heads=16, filter_size=4096,
        encoder_layers=7, encoder_kernels=[3, 7, 15, 31, 31, 31, 31],
        decoder_layers=6, decoder_kernels=[3, 7, 15, 31, 31, 31],
        encoder_conv_type="dynamic", decoder_conv_type="dynamic",
        attention_dropout=0.1, weight_dropout=0.1, dropout=0.1)


@register_hparams_set("dynamic_conv_toy")
def dynamic_conv_toy():
    return _common_hparams(
        dmodel=8, num_heads=4, filter_size=32,
        encoder_layers=2, encoder_kernels=[3, 7],
        decoder_layers=2, decoder_kernels=[3, 5],
        encoder_conv_type="dynamic", decoder_conv_type="dynamic",
        attention_dropout=0.1, weight_dropout=0.1, dropout=0.1)
