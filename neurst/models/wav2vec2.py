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

from neurst.layers.encoders.transformer_encoder import TransformerEncoder
from neurst.layers.modalities.audio_modalities import PositionalConv, Wav2vec2FeatureExtractor
from neurst.models import register_model
from neurst.models.model import BaseModel
from neurst.utils import compat
from neurst.utils.flags_core import Flag
from neurst.utils.hparams_sets import register_hparams_set


@register_model("wav2vec2")
class Wav2Vec2(BaseModel):
    """ Defines the Wav2vec2 model.
    Described in https://arxiv.org/abs/2006.11477.
    """

    def __init__(self,
                 args: dict,
                 feature_extractor,
                 post_extract_proj,
                 positional_conv,
                 encoder,
                 name=None):
        """ Initializes a BERT model.

        Args:
            args: A dict, containing the model configuration.
            feature_extractor: The convolutional layers for raw audio feature extraction.
            post_extract_proj: A dense layer or None.
            positional_conv: The positional convolution layer.
            encoder: The transformer encoder for feature transformation.
            name: The name of the model.options = tf.data.Options()
        """
        super(Wav2Vec2, self).__init__(args, name=name or "wav2vec2")
        self._feature_extractor = feature_extractor
        self._feature_extractor_norm_layer = tf.keras.layers.LayerNormalization(
            epsilon=1.e-5, dtype="float32", name="feature_extractor_norm")
        self._post_extract_proj = post_extract_proj
        self._pos_conv = positional_conv
        self._encoder = encoder
        self._dropout_input = args["dropout_input"]
        self._encoder_dropout = args["encoder_dropout"]

    @staticmethod
    def class_or_method_args():
        return [
            Flag("conv_bias", dtype=Flag.TYPE.BOOLEAN, default=False,
                 help="Whether to use bias tensor in feature extractor."),
            Flag("conv_feature_layers", dtype=Flag.TYPE.STRING,
                 default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
                 help="The string describing convolutional feature extraction layers "
                      "in form of a python list that contains [(dim, kernel_size, stride), ...]"),
            Flag("extractor_mode", dtype=Flag.TYPE.STRING, default="default",
                 choices=["default", "layer_norm"],
                 help="The mode for feature extractor. default has a single group norm with d "
                      "groups in the first conv block, whereas layer_norm has layer norms in "
                      "every block (meant to use with normalize=True)"),
            Flag("encoder_embed_dim", dtype=Flag.TYPE.INTEGER, default=768,
                 help="The dimension of embedding."),
            Flag("conv_pos", dtype=Flag.TYPE.INTEGER, default=128,
                 help="The number of filters for convolutional positional embeddings."),
            Flag("conv_pos_groups", dtype=Flag.TYPE.INTEGER, default=16,
                 help="The number of groups for convolutional positional embedding."),
            Flag("dropout_input", dtype=Flag.TYPE.FLOAT, default=0.1,
                 help="The dropout rate to apply to the input (after feat extr)."),
            Flag("encoder_layerdrop", dtype=Flag.TYPE.FLOAT, default=0.05,
                 help="The probability of dropping a transformer layer."),
            Flag("encoder_dropout", dtype=Flag.TYPE.FLOAT, default=0.1,
                 help="The dropout probability for the transformer encoder."),
            Flag("encoder_layers", dtype=Flag.TYPE.INTEGER, default=12,
                 help="The number of layers in the transformer encoder."),
            Flag("encoder_attention_heads", dtype=Flag.TYPE.INTEGER, default=12,
                 help="The number of attention heads in the transformer encoder."),
            Flag("encoder_filter_size", dtype=Flag.TYPE.INTEGER, default=3072,
                 help="The number of units of the ffn layer in the transformer encoder."),
            Flag("encoder_activation_fn", dtype=Flag.TYPE.STRING, default="gelu_nonapprox",
                 help="The activation function to be used in the ffn layer of transformer encoder."),
            Flag("encoder_pre_norm", dtype=Flag.TYPE.BOOLEAN, default=False,
                 help="The norm mode for the transformer encoder."),
            Flag("quantize_input", dtype=Flag.TYPE.BOOLEAN, default=False,
                 help="Whether to use quantized inputs.")
        ]

    @classmethod
    def new(cls, args: dict, name=None):
        """ Builds a sequence to sequence model.

        Args:
            args: A dict containing all model parameters.
            name: The name of the model.

        Returns:
            A Wav2Vec2 model.
        """
        args["conv_feature_layers"] = eval(args["conv_feature_layers"])
        feature_extractor = Wav2vec2FeatureExtractor(conv_layers=args["conv_feature_layers"],
                                                     mode=args["extractor_mode"],
                                                     conv_bias=args["conv_bias"],
                                                     dropout=0., name="feature_extractor")
        post_extract_proj = None
        if (args["conv_feature_layers"][-1][0] != args["encoder_embed_dim"]
            and not args["quantize_input"]):
            post_extract_proj = tf.keras.layers.Dense(args["encoder_embed_dim"], name="post_extract_proj")
        pos_conv = PositionalConv(args["encoder_embed_dim"], kernel_size=args["conv_pos"],
                                  groups=args["conv_pos_groups"], dropout=args["encoder_dropout"], name="pos_conv")
        encoder = TransformerEncoder(num_layers=args["encoder_layers"],
                                     hidden_size=args["encoder_embed_dim"],
                                     num_attention_heads=args["encoder_attention_heads"],
                                     filter_size=args["encoder_filter_size"],
                                     ffn_activation=args["encoder_activation_fn"],
                                     attention_dropout_rate=args["encoder_dropout"],
                                     ffn_dropout_rate=args["encoder_dropout"],
                                     layer_postprocess_dropout_rate=args["encoder_dropout"],
                                     post_normalize=(not args["encoder_pre_norm"]),
                                     layer_postprocess_epsilon=1e-5, name="encoder")
        model = cls(args, feature_extractor, post_extract_proj, pos_conv, encoder)
        _src = tf.convert_to_tensor(numpy.random.rand(1, 5000), dtype=compat.CUSTOM_GLOBAL_FLOATX)
        _ = model({"src": _src, "src_padding": tf.zeros_like(_src)})
        return model

    def call(self, inputs, is_training=True):
        """ Forward pass of the BERT classifier.

        Args:
            inputs: A dict containing model inputs.
                - src: The raw audio signals with shape [batch, max_signals]
                - src_padding: A bool tensor indicating padding or not, with shape [batch, max_signals]
            is_training:

        Returns: A dict containing

        """
        src = inputs["src"]
        src_padding = inputs.get("src_padding", None)
        features = self._feature_extractor(src, is_training=is_training)
        # features_pen = tf.reduce_mean(tf.math.pow(features, 2))
        features = tf.cast(self._feature_extractor_norm_layer(features), compat.CUSTOM_GLOBAL_FLOATX)
        if src_padding is not None:
            ori_signal_size = tf.shape(src_padding)[1]
            feature_frames = tf.shape(features)[1]
            extra = ori_signal_size // feature_frames
            src_padding = tf.cast(tf.reduce_all(tf.cast(
                tf.reshape(src_padding[:, :extra * feature_frames],
                           [tf.shape(features)[0], feature_frames, -1]), tf.bool), axis=-1), features.dtype)

        if self._post_extract_proj is not None:
            features = self._post_extract_proj(features)
        if is_training:
            features = tf.nn.dropout(features, rate=self._dropout_input)
            # TODO: time mask
        x = self._pos_conv(features, src_padding, is_training=is_training)
        if is_training:
            x = tf.nn.dropout(x, rate=self._encoder_dropout)
        # TODO layer drop for training
        y = self._encoder(x, src_padding, is_training=False)
        return {
            "contextualized_representation": y,
            "contextualized_representation_padding": src_padding
        }


@register_hparams_set("wav2vec2_base")
def wav2vec2_base():
    return {
        "model.class": Wav2Vec2.__name__,
        "model.params": {
            "conv_bias": False,
            "conv_feature_layers": "[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
            "extractor_mode": "default",
            "encoder_embed_dim": 768,
            "conv_pos": 128,
            "quantize_input": False,
            "conv_pos_groups": 16,
            "dropout_input": 0.1,
            "encoder_layerdrop": 0.05,
            "encoder_dropout": 0.1,
            "encoder_layers": 12,
            "encoder_attention_heads": 12,
            "encoder_filter_size": 3072,
            "encoder_activation_fn": "gelu_nonapprox",
            "encoder_pre_norm": False
        }
    }


@register_hparams_set("wav2vec2_large")
def wav2vec2_large():
    return {
        "model.class": Wav2Vec2.__name__,
        "model.params": {
            "conv_bias": False,
            "conv_feature_layers": "[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
            "extractor_mode": "default",
            "encoder_embed_dim": 1024,
            "conv_pos": 128,
            "quantize_input": False,
            "conv_pos_groups": 16,
            "dropout_input": 0.1,
            "encoder_layerdrop": 0.2,
            "encoder_dropout": 0.1,
            "encoder_layers": 24,
            "encoder_attention_heads": 16,
            "encoder_filter_size": 4096,
            "encoder_activation_fn": "gelu_nonapprox",
            "encoder_pre_norm": False
        }
    }


@register_hparams_set("wav2vec2_toy")
def wav2vec2_toy():
    return {
        "model.class": Wav2Vec2.__name__,
        "model.params": {
            "conv_bias": False,
            "conv_feature_layers": "[(16, 10, 5)] + [(16, 3, 2)] * 4 + [(16,2,2)] + [(16,2,2)]",
            "extractor_mode": "default",
            "encoder_embed_dim": 16,
            "conv_pos": 2,
            "quantize_input": False,
            "conv_pos_groups": 1,
            "dropout_input": 0.1,
            "encoder_layerdrop": 0,
            "encoder_dropout": 0.1,
            "encoder_layers": 3,
            "encoder_attention_heads": 2,
            "encoder_filter_size": 32,
            "encoder_activation_fn": "gelu_nonapprox",
            "encoder_pre_norm": False
        }
    }
