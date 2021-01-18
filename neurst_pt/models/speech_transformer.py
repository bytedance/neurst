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
import numpy
import torch

from neurst.utils.flags_core import Flag
from neurst_pt.layers.common_layers import PositionEmbeddingWrapper
from neurst_pt.layers.decoders import build_decoder
from neurst_pt.layers.encoders import build_encoder
from neurst_pt.layers.modalities.audio_modalities import AudioConvSubsamplingLayer
from neurst_pt.models import register_model
from neurst_pt.models.encoder_decoder_model import EncoderDecoderModel
from neurst_pt.models.model_utils import input_length_to_padding


@register_model
class SpeechTransformer(EncoderDecoderModel):
    """ Defines the Speech Transformer model. """

    def __init__(self, args, *largs, **kwargs):
        super(SpeechTransformer, self).__init__(args, *largs, **kwargs)
        self._args = args

    @staticmethod
    def class_or_method_args():
        return [
            Flag("modality.share_embedding_and_softmax_weights", dtype=Flag.TYPE.BOOLEAN, default=False,
                 help="Whether to share the target embedding table and softmax weights."),
            Flag("modality.dim", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The default embedding dimension for both source and target side."),
            Flag("modality.source.dim", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The source-side embedding dimension, or `modality.dim` if not provided."),
            Flag("modality.target.dim", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The target-side embedding dimension, or `modality.dim` if not provided."),
            Flag("modality.timing", dtype=Flag.TYPE.STRING, default=None,
                 help="The arbitrary parameters for positional encoding of both source and target side."),
            Flag("modality.source.timing", dtype=Flag.TYPE.STRING, default=None,
                 help="The arbitrary parameters for source-side positional encoding, "
                      "or `modality.timing` if not provided."),
            Flag("modality.target.timing", dtype=Flag.TYPE.STRING, default=None,
                 help="The arbitrary parameters for target-side positional encoding, "
                      "or `modality.timing` if not provided."),
            Flag("modality.source.kernel_size", dtype=Flag.TYPE.INTEGER, default=3,
                 help="The kernel size for the first two conv layer"),
            Flag("modality.source.strides", dtype=Flag.TYPE.INTEGER, default=2,
                 help="The stride size for the first two conv layer"),
            Flag("modality.source.channels", dtype=Flag.TYPE.INTEGER, default=256,
                 help="The channels for the first two conv layer"),
            Flag("modality.source.layer_norm", dtype=Flag.TYPE.BOOLEAN, default=False,
                 help="Whether to apply layer norm in convolution layers."),
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
            Flag("encoder.layer_postprocess_epsilon", dtype=Flag.TYPE.FLOAT, default=1e-6,
                 help="The epsilon for layer normalization in decoder."),
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
            Flag("decoder.layer_postprocess_epsilon", dtype=Flag.TYPE.FLOAT, default=1e-6,
                 help="The epsilon for layer normalization in decoder."),
        ]

    @classmethod
    def build_modalities(cls, model_args, src_meta, trg_meta):
        """ Create source and target modality. """
        # if modality.source.dim is not defined, then use modality.dim as default
        src_dim = model_args["modality.source.dim"] or model_args["modality.dim"]
        # if modality.target.dim is not defined, then use modality.dim as default
        trg_dim = model_args["modality.target.dim"] or model_args["modality.dim"]
        # whether to share source and target embedding
        # creates target embedding table
        target_modality = cls.build_modality(
            vocab_size=trg_meta["vocab_size"], emb_dim=trg_dim,
            timing=(model_args["modality.target.timing"] or model_args["modality.timing"]),
            share_embedding_and_softmax_weights=model_args["modality.share_embedding_and_softmax_weights"])

        # creates source audio modality
        input_modality = AudioConvSubsamplingLayer(
            embedding_dim=src_dim,
            input_channels=src_meta["audio_feature_channels"],
            input_dimension=src_meta["audio_feature_dim"],
            kernel_size=model_args["modality.source.kernel_size"],
            strides=model_args["modality.source.strides"],
            channels=model_args["modality.source.channels"],
            layer_norm=model_args["modality.source.layer_norm"])
        src_timing = model_args["modality.source.timing"] or model_args["modality.timing"]
        if src_timing:
            if isinstance(src_timing, str):
                src_timing = {"timing": src_timing}
            elif not isinstance(src_timing, dict):
                raise ValueError("Unknown type of timing params: {}".format(str(src_timing)))
            input_modality = PositionEmbeddingWrapper(embedding_layer=input_modality, **src_timing)
        return input_modality, target_modality

    @classmethod
    def new(cls, args, src_meta, trg_meta):
        """ Builds a sequence to sequence model.

        Args:
            args: A dict containing all model parameters.
            src_meta: A dict containing source-side vocabulary meta data, e.g. eos_id, vocab_size.
            trg_meta: A dict containing target-side vocabulary meta data, e.g. eos_id, vocab_size.

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
        model = cls(args, src_meta, trg_meta, src_modality, trg_modality, encoder, decoder)
        fake_src = numpy.random.rand(1, 11, src_meta["audio_feature_dim"], src_meta["audio_feature_channels"])
        fake_inputs = {"src": torch.FloatTensor(fake_src),
                       "src_length": torch.LongTensor([11]),
                       "trg_input": torch.LongTensor([[1, 2, 3]]), }
        _ = model(fake_inputs)
        return model

    def get_symbols_to_logits_fn(self, inputs, *args, **kwargs):
        strides = self.args["modality.source.strides"]

        def _length_after_conv(_l):
            return ((_l + strides - 1) // strides + strides - 1) // strides

        inputs["src_padding"] = input_length_to_padding(
            _length_after_conv(inputs["src_length"]), _length_after_conv(inputs["src"].size()[1]))
        return super(SpeechTransformer, self).get_symbols_to_logits_fn(inputs, *args, **kwargs)

    @classmethod
    def build_model_args_by_name(cls, name):
        from neurst.models.speech_transformer import SpeechTransformer as TFSpeechTransformer
        return TFSpeechTransformer.build_model_args_by_name(name)
