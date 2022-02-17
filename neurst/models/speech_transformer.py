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
import tensorflow as tf

from neurst.layers.common_layers import PositionEmbeddingWrapper
from neurst.layers.decoders import build_decoder
from neurst.layers.encoders import build_encoder
from neurst.layers.modalities.audio_modalities import AudioConv2dSubsamplingLayer
from neurst.models import register_model
from neurst.models.encoder_decoder_model import EncoderDecoderModel
from neurst.utils import compat
from neurst.utils.flags_core import Flag


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
            Flag("encoder.post_normalize", dtype=Flag.TYPE.BOOLEAN, default=False,
                 help="Whether to apply layer norm after each encoder block."),
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
            Flag("decoder.post_normalize", dtype=Flag.TYPE.BOOLEAN, default=False,
                 help="Whether to apply layer norm after each decoder block."),
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
        input_name = "input_audio_modality"
        target_name = "target_symbol_modality"
        # creates target embedding table
        target_modality = cls.build_modality(
            vocab_size=trg_meta["vocab_size"], emb_dim=trg_dim, name=target_name,
            timing=(model_args["modality.target.timing"] or model_args["modality.timing"]),
            share_embedding_and_softmax_weights=model_args["modality.share_embedding_and_softmax_weights"])

        # creates source audio modality
        input_modality = AudioConv2dSubsamplingLayer(
            embedding_dim=src_dim,
            kernel_size=model_args["modality.source.kernel_size"],
            strides=model_args["modality.source.strides"],
            channels=model_args["modality.source.channels"],
            layer_norm=model_args["modality.source.layer_norm"],
            name=input_name)
        src_timing = model_args["modality.source.timing"] or model_args["modality.timing"]
        if src_timing:
            if isinstance(src_timing, str):
                src_timing = {"timing": src_timing}
            elif not isinstance(src_timing, dict):
                raise ValueError("Unknown type of timing params: {}".format(str(src_timing)))
            input_modality = PositionEmbeddingWrapper(
                embedding_layer=input_modality, name=input_name + "_posenc_wrapper", **src_timing)
        return input_modality, target_modality

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
            "encoder.class": "TransformerEncoder",
            "encoder.params": encoder_params})
        decoder = build_decoder({
            "decoder.class": "TransformerDecoder",
            "decoder.params": decoder_params})
        model = cls(args, src_meta, trg_meta, src_modality, trg_modality, encoder, decoder, name=name)
        fake_src = numpy.random.rand(1, 4, src_meta["audio_feature_dim"], src_meta["audio_feature_channels"])
        fake_inputs = {"src": tf.convert_to_tensor(fake_src, tf.float32),
                       "src_length": tf.convert_to_tensor([4], tf.int64),
                       "trg_input": tf.convert_to_tensor([[1, 2, 3]], tf.int64), }
        _ = model(fake_inputs)
        return model

    def get_symbols_to_logits_fn(self, inputs, *args, **kwargs):
        strides = self.args["modality.source.strides"]

        def _length_after_conv(_l):
            return ((_l + strides - 1) // strides + strides - 1) // strides

        inputs["src_padding"] = 1. - tf.sequence_mask(
            lengths=tf.cast(_length_after_conv(inputs["src_length"]), tf.int32),
            maxlen=tf.cast(_length_after_conv(tf.shape(inputs["src"])[1]), tf.int32),
            dtype=tf.dtypes.as_dtype(compat.CUSTOM_GLOBAL_FLOATX))
        return super(SpeechTransformer, self).get_symbols_to_logits_fn(inputs, *args, **kwargs)

    @classmethod
    def build_model_args_by_name(cls, name):
        if not name.startswith("speech_transformer"):
            return None

        kernel_size = 3
        strides = 2
        channels = 256
        layer_norm = True
        if name == "speech_transformer_toy":
            dmodel = 8
            num_heads = 2
            num_encoder_layers = 2
            num_decoder_layers = 2
            num_encoder_filter_size = 10
            num_decoder_filter_size = 10
            channels = 5
            dropout_rate = 0.1
        elif name == "speech_transformer_s":
            dmodel = 256
            num_heads = 4
            num_encoder_layers = 12
            num_decoder_layers = 6
            num_encoder_filter_size = 2048
            num_decoder_filter_size = 2048
            dropout_rate = 0.1
        elif name == "speech_transformer_m":
            dmodel = 512
            num_heads = 8
            num_encoder_layers = 12
            num_decoder_layers = 6
            num_encoder_filter_size = 2048
            num_decoder_filter_size = 2048
            dropout_rate = 0.1
        elif name == "speech_transformer_l":
            dmodel = 1024
            num_heads = 16
            num_encoder_layers = 12
            num_decoder_layers = 6
            num_encoder_filter_size = 4096
            num_decoder_filter_size = 4096
            channels = 512
            dropout_rate = 0.1
        else:
            return None
        return {
            "model.class": cls.__name__,
            "model.params": {
                "modality.source.kernel_size": kernel_size,
                "modality.source.strides": strides,
                "modality.source.channels": channels,
                "modality.source.layer_norm": layer_norm,
                "modality.dim": dmodel,
                "modality.share_embedding_and_softmax_weights": True,
                "modality.timing": "sinusoids",
                "encoder.num_layers": num_encoder_layers,
                "encoder.hidden_size": dmodel,
                "encoder.num_attention_heads": num_heads,
                "encoder.filter_size": num_encoder_filter_size,
                "encoder.attention_dropout_rate": dropout_rate,
                "encoder.attention_type": "dot_product",
                "encoder.ffn_activation": "relu",
                "encoder.ffn_dropout_rate": dropout_rate,
                "encoder.layer_postprocess_dropout_rate": dropout_rate,
                "decoder.num_layers": num_decoder_layers,
                "decoder.hidden_size": dmodel,
                "decoder.num_attention_heads": num_heads,
                "decoder.filter_size": num_decoder_filter_size,
                "decoder.attention_dropout_rate": dropout_rate,
                "decoder.attention_type": "dot_product",
                "decoder.ffn_activation": "relu",
                "decoder.ffn_dropout_rate": dropout_rate,
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
                "initial_factor": 5.0 if dmodel > 256 else 3.5,
                "end_factor": 2.0 if dmodel > 256 else 1.5,
                "dmodel": dmodel,
                "warmup_steps": 25000,
                "start_decay_at": 50000,
                "decay_steps": 50000,
            },
        }
