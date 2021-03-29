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
from absl import logging

from neurst.layers.decoders import build_decoder
from neurst.layers.encoders import build_encoder
from neurst.models import register_model
from neurst.models.bert import Bert
from neurst.models.encoder_decoder_model import EncoderDecoderModel
from neurst.models.model_utils import input_length_to_padding
from neurst.utils import compat
from neurst.utils.flags_core import Flag
from neurst.utils.hparams_sets import get_hyper_parameters, register_hparams_set


@register_model("ctnmt")
class CtnmtTransformer(EncoderDecoderModel):
    """ Defines the Transformer model. """

    def __init__(self, args, bert_model, *largs, **kwargs):
        """ Initializes a sequence to sequence model.

        Args:
            args: A dict, containing the model configuration.
            bert_model: None or a bert model.
            name: The name of the model.options = tf.data.Options()
        """
        super(CtnmtTransformer, self).__init__(args, *largs, **kwargs)
        self._args = args
        self._bert_model = bert_model
        assert bert_model is not None
        self._bert_integrate_mode = args["bert_mode"]
        assert self._bert_integrate_mode in ["bert_as_encoder", "dynamic_switch", "bert_distillation"]
        if self._bert_integrate_mode == "dynamic_switch":
            self._ds_gate_W = tf.keras.layers.Dense(self._bert_model.args["hidden_size"],
                                                    activation=None, use_bias=False, name="ds_gate_W")
            self._ds_gate_U = tf.keras.layers.Dense(self._bert_model.args["hidden_size"],
                                                    activation=None, name="ds_gate_U")
        elif self._bert_integrate_mode == "bert_distillation":
            logging.info("NOTICE: When training with `bert_mode`='bert_distillation', "
                         "one must add an extra distillation loss.")

    @staticmethod
    def class_or_method_args():
        this_args = [x for x in super(CtnmtTransformer, CtnmtTransformer).class_or_method_args()
                     if x.name not in ["encoder", "decoder"]]
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

            Flag("bert_config", dtype=Flag.TYPE.STRING, default=None,
                 help="The hyper parameter set of BERT module."),
            Flag("bert_mode", dtype=Flag.TYPE.STRING, default="dynamic_switch",
                 choices=["bert_as_encoder", "dynamic_switch", "bert_distillation"],
                 help="The mode of incorporating BERT."),
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
        encoder = None
        if args["bert_mode"] != "bert_as_encoder":
            encoder = build_encoder({
                "encoder.class": "TransformerEncoder",
                "encoder.params": encoder_params})
        decoder = build_decoder({
            "decoder.class": "TransformerDecoder",
            "decoder.params": decoder_params})
        with tf.name_scope(name or "ctnmt"):
            bert_model = Bert.new(get_hyper_parameters(args["bert_config"])["model.params"],
                                  vocab_meta=src_meta, name="bert")

        model = cls(args, bert_model, src_meta, trg_meta, src_modality, trg_modality,
                    encoder, decoder, name=(name or "ctnmt"))
        _ = model({"src": tf.convert_to_tensor([[1, 2, 3]], tf.int64),
                   "src_padding": tf.convert_to_tensor([[0, 0., 0]], tf.float32),
                   "trg_input": tf.convert_to_tensor([[1, 2, 3]], tf.int64)})
        return model

    def get_symbols_to_logits_fn(self, inputs, is_training, is_inference,
                                 decode_padded_length=None):
        """ Prepares for decoding.

        Args:
            inputs: A dict of model inputs.
                - src: int tensor with shape [batch_size, src_input_length].
                - src_padding: float tensor with shape [batch_size, src_input_length].
                - trg_input: int tensor with shape [batch_size, trg_length].
            is_training: A bool, whether in training mode or not.
            is_inference: A bool, whether in generation mode or not.
            decode_padded_length: The maximum decoding length when inference, for creating
                static-shape cache.

        Returns:  A tuple of (decoding_internal_states, decoder_input, symbol_to_logit_fn)
        """

        def dynamic_switch(h_lm, h_nmt):
            h_lm_ = tf.cast(h_lm, compat.CUSTOM_GLOBAL_FLOATX)
            gate = tf.math.sigmoid(self._ds_gate_W(h_lm_) + self._ds_gate_U(h_nmt))
            hidden = tf.multiply(gate, h_lm_) + tf.multiply(1 - gate, h_nmt)
            return hidden

        src = inputs["src"]
        src_padding = inputs.get("src_padding", None)
        if src_padding is None:
            src_padding = input_length_to_padding(inputs["src_length"], tf.shape(src)[1])
        bert_encoder_outputs = self._bert_model({"tokens": src, "padding": src_padding},
                                                is_training=is_training)["encoder_outputs"]
        if self._bert_integrate_mode == "bert_as_encoder":
            encoder_outputs = bert_encoder_outputs
        else:
            embedded_inputs = self._src_modality(src)
            if self._bert_integrate_mode == "dynamic_switch":
                embedded_inputs = dynamic_switch(bert_encoder_outputs, embedded_inputs)
            encoder_outputs = self._encoder(embedded_inputs, src_padding, is_training=is_training)

        decoder_internal_cache = self._decoder.create_decoding_internal_cache(
            encoder_outputs=encoder_outputs,
            encoder_inputs_padding=src_padding,
            is_inference=is_inference,
            decode_padded_length=decode_padded_length)

        def symbols_to_logits_fn(symbols, cache, time=None):
            """ Generate logits for next potential IDs

            Args:
                symbols: Current decoded sequence.
                cache: A dictionary of values storing the encoder output, encoder-decoder
                    attention bias, and previous decoder attention values.
                time: Loop index, or None for transformer training

            Returns: The logits Tensor.
            """
            decoder_output = self.get_decoder_output(symbols, cache, time,
                                                     is_training, decode_padded_length)
            logits = self.output_logits_layer(decoder_output)
            return logits

        generation_initializer = {
            "decoder_input": inputs["trg_input"],
            "decoder_internal_cache": decoder_internal_cache,
            "encoder_inputs_maxlen": tf.shape(encoder_outputs)[1],
            "encoder_outputs": encoder_outputs,
            "bert_outputs": bert_encoder_outputs,
            "eos_id": self._trg_meta["eos_id"],
            "unk_id": self._trg_meta["unk_id"]
        }
        return symbols_to_logits_fn, generation_initializer

    def call(self, inputs, is_training=True):
        """ Forward pass of the sequence to sequence model.

        Args:
            inputs: A dict of model inputs.
                - src: int tensor with shape [batch_size, src_input_length].
                - src_padding: float tensor with shape [batch_size, src_input_length].
                - trg_input: int tensor with shape [batch_size, trg_length].
            is_training: A bool, whether in training mode or not.

        Returns:
            A logits Tensor.
        """
        symbols_to_logits_fn, generation_initializer = self.get_symbols_to_logits_fn(
            inputs, is_training=is_training, is_inference=False)
        return {"logits": symbols_to_logits_fn(generation_initializer["decoder_input"],
                                               generation_initializer["decoder_internal_cache"]),
                "student_hidden_states": generation_initializer["encoder_outputs"],
                "teacher_hidden_states": generation_initializer["bert_outputs"]
                }

    @classmethod
    def build_model_args_by_name(cls, name):
        if name == "ctnmt_toy":
            dmodel = 8
            num_heads = 2
            num_encoder_layers = 2
            num_decoder_layers = 2
            num_encoder_filter_size = 10
            num_decoder_filter_size = 10
            dropout_rate = 0.1
            bert_config = "bert_toy"
        elif name == "ctnmt_base":
            dmodel = 768
            num_heads = 12
            num_encoder_layers = 12
            num_decoder_layers = 12
            num_encoder_filter_size = 3072
            num_decoder_filter_size = 3072
            dropout_rate = 0.2
            bert_config = "bert_base"
        elif name == "ctnmt_big":
            dmodel = 1024
            num_heads = 16
            num_encoder_layers = 12
            num_decoder_layers = 12
            num_encoder_filter_size = 4096
            num_decoder_filter_size = 4096
            dropout_rate = 0.3
            bert_config = "bert_large"
        elif name == "ctnmt_big_dp01":
            dmodel = 1024
            num_heads = 16
            num_encoder_layers = 12
            num_decoder_layers = 12
            num_encoder_filter_size = 4096
            num_decoder_filter_size = 4096
            dropout_rate = 0.1
            bert_config = "bert_large"
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
                "decoder.layer_postprocess_dropout_rate": dropout_rate,
                "bert_config": bert_config,
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


@register_hparams_set("ctnmt_toy")
def ctnmt_toy():
    return CtnmtTransformer.build_model_args_by_name("ctnmt_toy")


@register_hparams_set("ctnmt_base")
def ctnmt_base():
    return CtnmtTransformer.build_model_args_by_name("ctnmt_base")


@register_hparams_set("ctnmt_big")
def ctnmt_big():
    return CtnmtTransformer.build_model_args_by_name("ctnmt_big")


@register_hparams_set("ctnmt_big_dp01")
def ctnmt_big_dp01():
    return CtnmtTransformer.build_model_args_by_name("ctnmt_big_dp01")
