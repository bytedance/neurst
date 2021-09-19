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

from neurst.layers.encoders.transformer_encoder import TransformerEncoder
from neurst.layers.modalities.text_modalities import BertEmbedding
from neurst.models import register_model
from neurst.models.model import BaseModel
from neurst.utils.flags_core import Flag
from neurst.utils.hparams_sets import register_hparams_set


@register_model
class Bert(BaseModel):
    """ Defines the BERT model. """

    def __init__(self,
                 args,
                 vocab_meta,
                 embedding,
                 encoder,
                 pooler,
                 name=None):
        """ Initializes a BERT model.

        Args:
            args: A dict, containing the model configuration.
            vocab_meta: A dict containing vocabulary meta data, e.g. unk_id, mask_id, sep_id.
            word_embedding: The embedding layer (wrapped by positional encoding).
            token_type_embedding: The embedding layer of token types.
            encoder: The encoder.
            pooler: A dense layer that converters the encoded sequence tensor
                to a tensor of shape [batch, hidden].
            name: The name of the model.options = tf.data.Options()
        """
        super(Bert, self).__init__(args, name=name or "bert")
        self._vocab_meta = vocab_meta
        self._embedding = embedding
        self._encoder = encoder
        self._pooler = pooler

    @staticmethod
    def class_or_method_args():
        return [
            Flag("max_position_embeddings", dtype=Flag.TYPE.INTEGER, default=512,
                 help="The maximum numbers of positions."),
            Flag("num_layers", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The number of stacking layers of the encoder."),
            Flag("hidden_size", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The number of hidden units of the encoder."),
            Flag("num_attention_heads", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The number of heads of encoder self-attention."),
            Flag("filter_size", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The number of the filter size of encoder ffn."),
            Flag("ffn_activation", dtype=Flag.TYPE.STRING, default="gelu",
                 help="The activation function of encoder ffn layer."),
            Flag("attention_dropout_rate", dtype=Flag.TYPE.FLOAT, default=0.,
                 help="The dropout rate of encoder self-attention layer."),
            Flag("attention_type", dtype=Flag.TYPE.STRING, default="dot_product",
                 help="The type of the attention function of encoder self-attention layer."),
            Flag("ffn_dropout_rate", dtype=Flag.TYPE.FLOAT, default=0.,
                 help="The dropout rate of encoder ffn layer."),
            Flag("layer_postprocess_dropout_rate", dtype=Flag.TYPE.FLOAT, default=0.,
                 help="The dropout rate for each layer's post process in encoder."),
            Flag("layer_postprocess_epsilon", dtype=Flag.TYPE.FLOAT, default=1e-12,
                 help="The epsilon for layer normalization."),
        ]

    @classmethod
    def new(cls, args: dict, vocab_meta, name=None):
        """ Builds a sequence to sequence model.

        Args:
            args: A dict containing all model parameters.
            vocab_meta: A dict containing source-side vocabulary meta data, e.g. eos_id, vocab_size.
            name: The name of the model.

        Returns:
            A BERT model.
        """
        # build source and target modality
        embedding = BertEmbedding(
            embedding_dim=args["hidden_size"],
            vocab_size=vocab_meta["vocab_size"],
            max_positions=args["max_position_embeddings"],
            token_types=2,
            dropout_rate=args["ffn_dropout_rate"],
            epsilon=args["layer_postprocess_epsilon"],
            name="bert_embedding")
        encoder = TransformerEncoder(
            num_layers=args["num_layers"],
            hidden_size=args["hidden_size"],
            num_attention_heads=args["num_attention_heads"],
            filter_size=args["filter_size"],
            ffn_activation=args["ffn_activation"],
            attention_dropout_rate=args["attention_dropout_rate"],
            attention_type=args["attention_type"],
            ffn_dropout_rate=args["ffn_dropout_rate"],
            layer_postprocess_dropout_rate=args["layer_postprocess_dropout_rate"],
            layer_postprocess_epsilon=args["layer_postprocess_epsilon"],
            post_normalize=True, return_all_layers=True,
            name="encoder")
        pooler = tf.keras.layers.Dense(args["hidden_size"], activation=tf.nn.tanh,
                                       use_bias=True, name="pooler")
        model = cls(args, vocab_meta, embedding, encoder, pooler, name=name)
        _ = model({"tokens": tf.convert_to_tensor([[1, 2, 3]], tf.int64),
                   "padding": tf.convert_to_tensor([[0, 0., 0]], tf.float32)})
        return model

    def call(self, inputs, is_training=True):
        """ Forward pass of the BERT classifier.

        Args:
            inputs: A dict containing model inputs.
                - tokens: float tensor of shape [batch, length]
                - tokens_type: int tensor of shape [batch, length]
                - padding: int tensor of shape [batch, length], where 1 denotes the padding.
            is_training:

        Returns: A dict containing
            - encoder_output: [batch, length, hidden size]
            - pooled_output: [batch, hidden size]

        """
        tokens = inputs["tokens"]
        padding = inputs["padding"]
        tokens_type = inputs.get("tokens_type", None)
        embedded_inputs = self._embedding(tokens, tokens_type, is_training)
        # [batch, length, hidden size]
        encoder_outputs = self._encoder(embedded_inputs, padding, is_training=is_training)[-1]
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        first_token_tensor = tf.squeeze(encoder_outputs[:, 0:1, :], axis=1)
        pooled_output = self._pooler(first_token_tensor)
        return {"encoder_outputs": encoder_outputs,
                "pooled_output": pooled_output}


def _bert_hparams(num_layers,
                  hidden_size,
                  filter_size,
                  dropout_rate,
                  num_heads,
                  max_positions):
    return {
        "model.class": Bert.__name__,
        "model.params": {
            "max_position_embeddings": max_positions,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "num_attention_heads": num_heads,
            "filter_size": filter_size,
            "ffn_activation": "gelu",
            "attention_dropout_rate": dropout_rate,
            "attention_type": "dot_product",
            "ffn_dropout_rate": dropout_rate,
            "layer_postprocess_dropout_rate": dropout_rate,
            "layer_postprocess_epsilon": 1e-12,
        }
    }


@register_hparams_set("bert_base")
def bert_base():
    return _bert_hparams(
        num_layers=12,
        hidden_size=768,
        filter_size=3072,
        dropout_rate=0.1,
        num_heads=12,
        max_positions=512)


@register_hparams_set("bert_large")
def bert_large():
    return _bert_hparams(
        num_layers=24,
        hidden_size=1024,
        filter_size=4096,
        dropout_rate=0.1,
        num_heads=16,
        max_positions=512)


@register_hparams_set("bert_toy")
def bert_toy():
    return _bert_hparams(
        num_layers=4,
        hidden_size=8,
        filter_size=16,
        dropout_rate=0.1,
        num_heads=2,
        max_positions=512)
