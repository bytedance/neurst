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

from neurst.layers import layer_utils
from neurst.layers.search import register_search_layer
from neurst.layers.search.sequence_search import SequenceSearch
from neurst.utils import compat
from neurst.utils.flags_core import Flag


def _calculate_length_penalty(lengths, alpha, dtype=None):
    """ Calculates length penalty, Referring
      to https://arxiv.org/abs/1609.08144.

    Args:
        lengths: The length tensor, with shape [n, ]
        alpha: The length penalty rate. Length penalty is given by
          (5+len(decode)/6) ^ -\alpha.
        dtype: The dtype for return.

    Returns: The length penalty tensor.
    """
    if dtype is None:
        dtype = compat.CUSTOM_GLOBAL_FLOATX
    if alpha is None or alpha < 0.0:
        return tf.constant(1., dtype=dtype) / tf.cast(lengths, dtype=dtype)
    return ((5.0 + tf.cast(lengths, dtype)) / 6.0) ** (-alpha)


class _StateKeys(object):
    """Keys to dictionary storing the state of the beam search loop."""

    # The decoding step.
    TIME_STEP = "TIME_STEP"

    # The decoder input ids with shape [batch_size * beam_size, ]
    INPUT_IDS = "INPUT_IDS"

    # Dictionary of cached values for each beam. The cache stores encoder
    # output (memory), encoder input paddings and the decoder attention
    # output from the previous iteration (usually for transformer)
    CACHE = "CACHE"

    # Flags indicating which sequences in the finished sequences are finished,
    # with shape [batch_size * beam_size, ]
    FINISHED_FLAGS = "FINISHED_FLAGS"

    # Log probabilities of each sequence with shape [batch_size * beam_size]
    LOG_PROBS = "LOG_PROBS"

    # The true decoding length of each beam
    DECODING_LENGTH = "LENGTH"

    # The predicted sequences, with shape [time_step, batch_size * beam_size]
    PREDICTED_IDS = "PREDICTED_IDS"


def _calculate_log_probs(state,
                         symbols_to_logits_fn,
                         eos_id,
                         unk_id,
                         ensemble_weights=None):
    """ Calculates one-step log probability.

    Finished beam will be masked and UNK will be masked
    if strategy == BASIC_NO_UNK.

    Args:
        state: A dictionary containing current state of beam search.
        symbols_to_logits_fn:
        eos_id: An int scalar, indicating the end-of-sentence token id, used to determine when a
            sequence has finished.
        unk_id: An int scalar, indicating the unknown token id.
        ensemble_weights: A list of float values, indicating the weights of each submodel's probability.

    Returns:
        A float tensor with the same shape as `logits`.
    """
    logits = symbols_to_logits_fn(
        state[_StateKeys.INPUT_IDS], state[_StateKeys.CACHE],
        state[_StateKeys.TIME_STEP])
    logits = tf.nest.flatten(logits)
    vocab_size = logits[0].get_shape().as_list()[-1]
    batch_beam_size = tf.shape(logits[0])[0]
    if len(logits) == 1:
        # [batch_size * beam_size, target_vocab_size]
        log_probs = tf.nn.log_softmax(logits[0])
    else:
        probs = tf.nest.map_structure(
            lambda x: tf.expand_dims(
                tf.reshape(tf.nn.softmax(x), shape=[-1]), axis=0),
            logits)
        original_shape = tf.shape(logits[0])
        # [num_models, xxx]
        probs = tf.concat(probs, axis=0)
        # [1, num_models]
        weights = tf.expand_dims(
            tf.convert_to_tensor(ensemble_weights, dtype=probs.dtype),
            axis=0)
        probs = tf.matmul(weights, probs)
        log_probs = tf.math.log(tf.reshape(probs, original_shape))

    # [batch_size * beam_size,]
    prev_finished_float = tf.cast(state[_StateKeys.FINISHED_FLAGS],
                                  log_probs.dtype)
    # mask the finished beam except only one entrance (target_eos_id)
    #   [target_vocab_size, ]: [float_min, float_min, float_min, ..., 0]
    #   this forces the beam with EOS continue to generate EOS
    finished_beam_bias = layer_utils.one_entry_bias(
        on_entry=eos_id, num_entries=vocab_size,
        on_value=0., off_value=compat.FLOAT_MIN,
        dtype=log_probs.dtype)
    # [batch_size * beam_size, target_vocab_size]: outer product
    finished_beam_bias = layer_utils.tile_tensor(
        finished_beam_bias, batch_beam_size, axis=0)
    finished_beam_bias *= tf.expand_dims(prev_finished_float, 1)
    # compute new probs, with finished flags & mask
    log_probs = log_probs * tf.expand_dims(1. - prev_finished_float, 1) + finished_beam_bias

    # we should use the trick for masking out the UNK in the probability
    if unk_id is not None:
        unk_beam_bias = layer_utils.one_entry_bias(
            on_entry=unk_id, num_entries=vocab_size,
            on_value=compat.FLOAT_MIN, off_value=0.,
            dtype=log_probs.dtype)
        unk_beam_bias = layer_utils.tile_tensor(
            unk_beam_bias, batch_beam_size, axis=0)
        log_probs += unk_beam_bias
    return log_probs


def _sample_next_word(state,
                      log_probs,
                      beam_size,
                      length_penalty):
    """ Generates next token ids.

    Args:
        state: A dictionary containing current state of beam search.
        log_probs: The log probs of current timestep with shape
            [batch_size * beam_size, vocab_size]
        beam_size: The beam width.
        length_penalty: A float scalar, defining the strength of
            length normalization.

    Returns: A tuple `(word_ids, beam_ids, next_log_probs, next_lengths)`, where
      `words_ids` is the ids of sampled word symbols; `beam_ids` indicates the index
      of beam which the symbol at the position is from; `next_log_probs` is the accumulated
      log probabilities of each beam; `next_lengths` is the decoding lengths of
      each beam.
      All of the Tensors have shape [batch_size * beam_size, ].
    """
    batch_beam_size = tf.shape(log_probs)[0]
    batch_size = batch_beam_size // beam_size
    vocab_size = log_probs.get_shape().as_list()[-1]
    # calculates the accumulated log probs
    #   [batch_size * beam_size, target_vocab_size]
    log_probs = log_probs + tf.expand_dims(state[_StateKeys.LOG_PROBS], 1)
    # new decoding length: [batch_size * beam_size, ]
    next_length = state[_StateKeys.DECODING_LENGTH] + 1 - tf.cast(
        state[_StateKeys.FINISHED_FLAGS], tf.int32)
    # calculates beam scores
    #  length_penalty: [batch_size * beam_size,]
    penalty_term = _calculate_length_penalty(
        next_length, length_penalty, dtype=log_probs.dtype)
    scores = log_probs * tf.expand_dims(penalty_term, axis=1)

    # flatten: [batch_size, beam_size * target_vocab_size]
    scores = tf.reshape(tf.reshape(scores, [-1]),
                        [batch_size, -1])
    scores_flat = tf.cond(
        state[_StateKeys.TIME_STEP] > 0,
        lambda: scores,  # time > 0: all
        lambda: tf.slice(scores, [0, 0],
                         [-1, vocab_size]))  # time = 0: first logits in each batch
    # [batch_size, beam_size] will restore top beam results
    sample_scores, sample_ids = tf.nn.top_k(scores_flat, k=beam_size)
    # flatten: [batch_size * beam_size,]
    sample_ids = tf.reshape(sample_ids, [-1])
    # because we do topk to scores with dim: [batch, beam * vocab]
    #   we need to cover the true word ids
    word_ids = tf.math.floormod(sample_ids, vocab_size)

    # find beam_ids, indicating the current position is from which beam
    #  with shape [batch_size, beam_size], e.g. [[0, 0, ...], [1, 1,...], ..., [batch_size,...] ]
    batch_pos = layer_utils.compute_batch_indices(batch_size, beam_size)
    #  beam_base_pos: [batch_size * beam_size,]: [0, 0, ..., beam, beam,..., 2beam, 2beam, ...]
    beam_base_pos = tf.reshape(batch_pos * beam_size, [-1])
    # compute new beam_ids, [batch_size * beam_size, ]
    beam_ids = tf.math.floordiv(sample_ids, vocab_size) + beam_base_pos

    # gather states according to beam_ids
    next_lengths = tf.gather(next_length, beam_ids)

    # we need to recover log_probs according to scores's topk ids
    #   [batch_size * beam_size * vocab_size, ]
    log_probs_flat = tf.reshape(log_probs, [-1])
    log_probs_index = beam_base_pos * vocab_size + sample_ids
    next_log_probs = tf.gather(log_probs_flat, log_probs_index)

    return word_ids, beam_ids, next_log_probs, next_lengths


def _extract_beam_results(state,
                          beam_size,
                          top_k,
                          length_penalty):
    """ Extract `top_k` hypothesis from beam search results.

    Args:
        state: A dictionary containing current state of beam search.
        beam_size: The beam width.
        top_k: The number of hypothesis of each sample with highest scores
            with be returned. It must be <= `beam_sze`.
        length_penalty: A float scalar, defining the strength of
            length normalization.

    Returns: A tuple of sorted hypothesis and corresponding scores.
        => sorted_hypothesis: [batch_size * top_k, max_len_of_hypothesis]
        => sorted_socres: [batch_size * top_k, ]
    """
    log_probs = state[_StateKeys.LOG_PROBS]
    penalty = _calculate_length_penalty(
        state[_StateKeys.DECODING_LENGTH],
        length_penalty, dtype=log_probs.dtype)
    scores = log_probs * penalty
    # [_batch * _beam, ] => [_batch, _beam]
    scores_shaped = tf.reshape(scores, [-1, beam_size])
    # [_batch, _top_k]
    top_scores, top_indices = tf.nn.top_k(scores_shaped, k=top_k)
    batch_beam_pos = layer_utils.compute_batch_indices(
        tf.shape(top_indices)[0], k=top_k) * beam_size
    # [_batch * _top_k, ]
    top_indices = tf.reshape(top_indices + batch_beam_pos, [-1])
    # [_batch * _top_k, timesteps]
    sorted_hypothesis = tf.gather(state[_StateKeys.PREDICTED_IDS], top_indices)
    sorted_scores = tf.reshape(top_scores, [-1])
    return sorted_hypothesis, sorted_scores


def sequence_beam_search(symbols_to_logits_fn,
                         generation_initializer,
                         top_k=1,
                         beam_size=4,
                         length_penalty=0.6,
                         extra_decode_length=50,
                         maximum_decode_length=256,
                         minimum_decode_length=0,
                         ensemble_weights=None,
                         padded_decode=False,
                         enable_unk=False,
                         dtype=None):
    """ Search for target subtokens with the largest probability.

    Args:
        symbols_to_logits_fn: A callable function that takes ids, cache and current
            timestep as arguments. The passed in arguments will have following shape:
                ids -> A tensor with shape [batch_size * beam_size,].
                cache -> A nested dictionary of tensors [batch_size * beawm_size, ...].
                index -> A scalar.
            The function must return a logits tensor with shape
                [batch_size * beam_size, vocab_size].
        generation_initializer: A dict from model, containing:
            -> decoder_input_ids: An int32 tensor with shape [batch_size,].
                    The initial input for the decoder.
            -> decoding_cache: A nested dictionary, containing attention memory and other
                    starting variables information.
            -> encoder_inputs_maxlen: An int32 scalar tensor, indicating the maximum length of encoding input
                    sequence.
            -> eos_id: An int scalar, indicating the end-of-sentence symbol id, used to determine when a
                    sequence has finished.
            -> unk_id: An int scalar, indicating the unknown token id.
        top_k: An int scalar, indicating the number of hypothesis each sample will generate.
        beam_size: An int scalar, indicating the number of beams.
        length_penalty: A float scalar, defining the strength of length normalization.
        extra_decode_length: An int scalar. The real search steps will be
            `encoder_inputs_maxlen` + `extra_decode_length`.
        maximum_decode_length: An int scalar, indicating the maximum decode length. Prediction
            outputs will be padded to this length.
        minimum_decode_length: An int scalar, indicating the minimum decode length.
        ensemble_weights: A list of float values, indicating the weights of each submodel's probability.
        padded_decode: Whether the autoregressive decoding runs with input data padded to the decode_max_length.
        enable_unk: Whether to enable the search method to generate UNK.
        dtype: A string or a tensorflow data type used for score computation. If None, auto set
            to the GLOBAL_FLOATX.

    Returns:
        A tuple of (hypothesis, scores):
            hypothesis: int32 tensor with shape [batch_size * top_k, decode_length]
            scores: float tensor with shape [batch_size * top_k]
    """
    decoder_input_ids = generation_initializer["decoder_input"]
    decoding_cache = generation_initializer["decoder_internal_cache"]
    encoder_inputs_maxlen = generation_initializer.get("encoder_inputs_maxlen", None)
    eos_id = generation_initializer["eos_id"]
    unk_id = None if enable_unk else generation_initializer.get("unk_id", None)
    if dtype is None:
        dtype = tf.dtypes.as_dtype(compat.CUSTOM_GLOBAL_FLOATX)
    batch_size = tf.shape(decoder_input_ids)[0]
    initial_finished = tf.tile([False], [batch_size * beam_size])
    initial_decoder_input_ids = layer_utils.stack_beam_size(decoder_input_ids, beam_size)
    initial_time = tf.constant(0, dtype=tf.int32)
    initial_cache = tf.nest.map_structure(
        lambda x: layer_utils.stack_beam_size(x, beam_size), decoding_cache)

    initial_log_probs = tf.zeros_like(
        initial_decoder_input_ids, dtype=dtype)
    initial_length = tf.zeros_like(initial_decoder_input_ids, dtype=tf.int32)
    # [time, batch_size * beam_size]
    if padded_decode:
        initial_predicted_ids = tf.zeros([batch_size * beam_size, maximum_decode_length],
                                         dtype=tf.int32)
    else:
        initial_predicted_ids = tf.zeros([batch_size * beam_size, 0],
                                         dtype=tf.int32)
    # Create state dictionary
    search_state = {
        _StateKeys.TIME_STEP: initial_time,
        _StateKeys.INPUT_IDS: tf.cast(initial_decoder_input_ids, tf.int32),
        _StateKeys.CACHE: initial_cache,
        _StateKeys.FINISHED_FLAGS: initial_finished,
        _StateKeys.LOG_PROBS: initial_log_probs,
        _StateKeys.DECODING_LENGTH: initial_length,
        _StateKeys.PREDICTED_IDS: initial_predicted_ids
    }
    # Create state invariants for each value in the state dictionary. Each
    # dimension must be a constant or None. A None dimension means either:
    #   1) the dimension's value is a tensor that remains the same but may
    #      depend on the input sequence to the model (e.g. batch size).
    #   2) the dimension may have different values on different iterations.
    if padded_decode:
        search_state_shape_invariants = {
            _StateKeys.TIME_STEP: tf.TensorShape([]),
            _StateKeys.INPUT_IDS: tf.TensorShape([None]),
            _StateKeys.CACHE: tf.nest.map_structure(
                layer_utils.static_tensorshape, initial_cache),
            _StateKeys.FINISHED_FLAGS: tf.TensorShape([None]),
            _StateKeys.LOG_PROBS: tf.TensorShape([None]),
            _StateKeys.DECODING_LENGTH: tf.TensorShape([None]),
            _StateKeys.PREDICTED_IDS: tf.TensorShape([None, maximum_decode_length]),
        }
    else:
        search_state_shape_invariants = {
            _StateKeys.TIME_STEP: tf.TensorShape([]),
            _StateKeys.INPUT_IDS: tf.TensorShape([None]),
            _StateKeys.CACHE: tf.nest.map_structure(
                layer_utils.dynamic_tensorshape_except_last_dim, initial_cache),
            _StateKeys.FINISHED_FLAGS: tf.TensorShape([None]),
            _StateKeys.LOG_PROBS: tf.TensorShape([None]),
            _StateKeys.DECODING_LENGTH: tf.TensorShape([None]),
            _StateKeys.PREDICTED_IDS: tf.TensorShape([None, None]),
        }
    if encoder_inputs_maxlen is None:
        maximum_search_steps = maximum_decode_length
    else:
        maximum_search_steps = tf.minimum(
            encoder_inputs_maxlen + extra_decode_length,
            maximum_decode_length)
    maximum_search_steps = tf.maximum(maximum_search_steps, minimum_decode_length)

    def search_step(state):
        """ Beam search step. """
        # [batch_size * beam_size, vocab_size]
        log_probs = _calculate_log_probs(
            state=state, symbols_to_logits_fn=symbols_to_logits_fn,
            eos_id=eos_id, unk_id=unk_id, ensemble_weights=ensemble_weights)
        # masking out the EOS in the probability when decoding length < min_length
        eos_beam_bias = layer_utils.one_entry_bias(
            on_entry=eos_id, num_entries=log_probs.get_shape().as_list()[-1],
            on_value=compat.FLOAT_MIN, off_value=0.,
            dtype=log_probs.dtype)
        eos_beam_bias = layer_utils.tile_tensor(eos_beam_bias, tf.shape(log_probs)[0], axis=0)
        log_probs = tf.cond(tf.less(state[_StateKeys.TIME_STEP], minimum_decode_length - 1),
                            lambda: log_probs + eos_beam_bias,
                            lambda: log_probs)

        # compute log probs and generate next token ids according to beam scores
        sample_ids, beam_ids, next_log_probs, next_length = _sample_next_word(
            state=state, log_probs=log_probs, beam_size=beam_size,
            length_penalty=length_penalty)
        # re-order beams by beam_ids
        next_predicted_ids = tf.gather(state[_StateKeys.PREDICTED_IDS], beam_ids)
        if padded_decode:
            next_predicted_ids = tf.transpose(tf.tensor_scatter_nd_update(
                tf.transpose(next_predicted_ids), [[state[_StateKeys.TIME_STEP]]],
                tf.expand_dims(sample_ids, axis=0)))
        else:
            next_predicted_ids = tf.concat([next_predicted_ids, tf.expand_dims(
                sample_ids, axis=1)], axis=1)
        next_cache = tf.nest.map_structure(
            lambda x: tf.gather(x, beam_ids), state[_StateKeys.CACHE])
        next_finished = tf.equal(eos_id, sample_ids)
        state.update({
            _StateKeys.TIME_STEP: state[_StateKeys.TIME_STEP] + 1,
            _StateKeys.INPUT_IDS: sample_ids,
            _StateKeys.CACHE: next_cache,
            _StateKeys.FINISHED_FLAGS: next_finished,
            _StateKeys.LOG_PROBS: next_log_probs,
            _StateKeys.DECODING_LENGTH: next_length,
            _StateKeys.PREDICTED_IDS: next_predicted_ids})
        return [state]

    # beam search step by step
    res = tf.while_loop(
        lambda state: tf.logical_and(
            tf.logical_not(tf.reduce_all(state[_StateKeys.FINISHED_FLAGS])),
            tf.less(state[_StateKeys.TIME_STEP], maximum_search_steps)),
        search_step,
        loop_vars=[search_state],
        shape_invariants=[search_state_shape_invariants],
        parallel_iterations=1)
    res = res[0]
    sorted_hypothesis, sorted_scores = _extract_beam_results(
        state=res, beam_size=beam_size, top_k=top_k,
        length_penalty=length_penalty)
    # padding to fixed length output
    if not padded_decode:
        sorted_hypothesis = tf.pad(
            sorted_hypothesis,
            paddings=tf.convert_to_tensor([
                [0, 0], [0, maximum_decode_length - res[_StateKeys.TIME_STEP]]],
                dtype=tf.int32),
            mode="CONSTANT",
            constant_values=eos_id)
    return sorted_hypothesis, sorted_scores


@register_search_layer
class BeamSearch(SequenceSearch):

    def __init__(self, args):
        super(BeamSearch, self).__init__()
        self._beam_size = args["beam_size"]
        self._length_penalty = args["length_penalty"]
        self._top_k = args["top_k"]
        if self._top_k > 1:
            logging.info("Initializes BeamSearch with top_k = {}. "
                         "Be careful if the resulted predictions are "
                         "passed to evaluation metrics. ".format(self._top_k))
        self._maximum_decode_length = args["maximum_decode_length"]
        self._extra_decode_length = args["extra_decode_length"]
        self._minimum_decode_length = args["minimum_decode_length"]
        if not self._minimum_decode_length:
            self._minimum_decode_length = 0
        self._ensemble_weights = args["ensemble_weights"]
        self._padded_decode = args["padded_decode"]
        self._enable_unk = args["enable_unk"]

    @staticmethod
    def class_or_method_args():
        return [
            Flag("beam_size", dtype=Flag.TYPE.INTEGER, default=4,
                 help="The beam width of beam search inference."),
            Flag("length_penalty", dtype=Flag.TYPE.FLOAT, default=0.6,
                 help="The length penalty of beam search inference."),
            Flag("top_k", dtype=Flag.TYPE.INTEGER, default=1,
                 help="The number of reserved predictions with top scores of beam search inference."),
            Flag("maximum_decode_length", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The maximum decoding length of beam search inference."),
            Flag("minimum_decode_length", dtype=Flag.TYPE.INTEGER, default=0,
                 help="The minimum decoding length of beam search inference."),
            Flag("extra_decode_length", dtype=Flag.TYPE.INTEGER, default=50,
                 help="The extra decoding length versus source side for beam search inference. "
                      "The maximum decoding length of beam search inference will be "
                      "source_sequence_length + extra_decode_length if maximum_decode_length "
                      "if not provided."),
            Flag("padded_decode", dtype=Flag.TYPE.BOOLEAN, default=None,
                 help="Whether the autoregressive decoding runs with input data padded to "
                      "the decode_max_length. For TPU/XLA-GPU runs, this flag has to be "
                      "set due the static shape requirement. In addition, this method "
                      "will introduce unnecessary overheads which grow quadratically with "
                      "the max sequence length."),
            Flag("ensemble_weights", dtype=Flag.TYPE.STRING, default="average",
                 help="The weight scheme for model ensemble, which could be comma-separated numbers."),
            Flag("enable_unk", dtype=Flag.TYPE.BOOLEAN, default=None,
                 help="Whether to enable the search method to generating UNK."),
        ]

    def _get_ensemble_weights(self, ensemble_weights):
        """ Creates ensemble weights from `ensemble_weights`.

        Now, only accepts weight_scheme="average" or comma-separate numbers.

        Args:
            ensemble_weights: The passed-in argument.

        Returns: A list of floats. The size is the length of `self._model`.
        """
        if hasattr(self._model, "model_num"):
            model_num = self._model.model_num
        else:
            return None
        if ensemble_weights == "average":
            return [1. / float(model_num)] * model_num
        else:
            eles = ensemble_weights.strip().split(",")
            if len(eles) != model_num:
                raise ValueError("The number of manual weights must have equal to the models.")
            eles = [float(x) for x in eles]
            eles_sum = sum(eles)
            return [x / eles_sum for x in eles]

    def call(self, parsed_inp, **kwargs):
        """ Do beam search.

        Args:
            parsed_inp: A dict of parsed model inputs.

        Returns:
            The search results (must be a tuple).
        """
        maximum_decode_length = self._maximum_decode_length
        max_data_len = kwargs.get("max_data_len", None)
        if maximum_decode_length is None:
            assert max_data_len, ("`maximum_decode_length` must be provided "
                                  "when `max_data_len` is not provided.")
            maximum_decode_length = self._extra_decode_length + max_data_len
        if self._minimum_decode_length >= maximum_decode_length:
            raise ValueError("`minimum_decode_length` must be less than maximum decode length.")

        with tf.name_scope("beam_search"):
            decode_padded_length = maximum_decode_length if self._padded_decode else None
            symbols_to_logits_fn, generation_initializer = self._model.get_symbols_to_logits_fn(
                parsed_inp, is_training=False, is_inference=True,
                decode_padded_length=decode_padded_length)
        sorted_hypothesis, sorted_scores = sequence_beam_search(
            symbols_to_logits_fn,
            generation_initializer,
            top_k=self._top_k,
            beam_size=self._beam_size,
            length_penalty=self._length_penalty,
            extra_decode_length=self._extra_decode_length,
            maximum_decode_length=maximum_decode_length,
            minimum_decode_length=self._minimum_decode_length,
            ensemble_weights=self._get_ensemble_weights(self._ensemble_weights),
            padded_decode=self._padded_decode,
            enable_unk=self._enable_unk)
        return sorted_hypothesis, sorted_scores
