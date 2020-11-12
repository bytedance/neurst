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

from neurst.layers import layer_utils
from neurst.layers.search import register_search_layer
from neurst.layers.search.sequence_search import SequenceSearch
from neurst.utils import compat
from neurst.utils.flags_core import Flag


class _StateKeys(object):
    """Keys to dictionary storing the state of the sampling search loop."""

    # The decoding step.
    TIME_STEP = "TIME_STEP"

    # The decoder input ids with shape [batch_size * sample_num, ]
    INPUT_IDS = "INPUT_IDS"

    # Dictionary of cached values for each beam. The cache stores encoder
    # output (memory), encoder input paddings and the decoder attention
    # output from the previous iteration (usually for transformer)
    CACHE = "CACHE"

    # Flags indicating which sequences in the finished sequences are finished,
    # with shape [batch_size * sample_num, ]
    FINISHED_FLAGS = "FINISHED_FLAGS"

    # The true decoding length of each beam
    DECODING_LENGTH = "LENGTH"

    # The predicted sequences, with shape [time_step, batch_size * sample_num]
    PREDICTED_IDS = "PREDICTED_IDS"


def _calculate_logits(state, symbols_to_logits_fn, unk_id):
    """ Calculates one-step logits. UNK will be masked.

    Args:
        state: A dictionary containing current state of beam search.
        symbols_to_logits_fn:
        unk_id: An int scalar, indicating the unknown token id.

    Returns:
        A float tensor with the same shape as `logits`.
    """
    logits = symbols_to_logits_fn(
        state[_StateKeys.INPUT_IDS], state[_StateKeys.CACHE],
        state[_StateKeys.TIME_STEP])

    vocab_size = logits.get_shape().as_list()[-1]
    unk_bias = layer_utils.one_entry_bias(
        on_entry=unk_id, num_entries=vocab_size,
        on_value=compat.FLOAT_MIN, off_value=0.0, dtype=logits.dtype)
    unk_coefficient = layer_utils.one_entry_bias(
        on_entry=unk_id, num_entries=vocab_size,
        on_value=0.0, off_value=1.0, dtype=logits.dtype)

    logits = unk_coefficient * logits + unk_bias

    return logits


def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    values, _ = tf.nn.top_k(logits, k=k)
    min_values = values[:, -1, tf.newaxis]
    return tf.where(logits < min_values,
                    tf.ones_like(logits, dtype=logits.dtype) * compat.FLOAT_MIN, logits)


def top_p_logits(logits, p):
    sorted_logits = tf.sort(logits, direction="DESCENDING")
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits), axis=-1)
    t_sorted_indices_to_remove = cumulative_probs < p
    min_value = tf.reduce_min(logits)
    applyed_sorted_logits = tf.where(
        t_sorted_indices_to_remove,
        tf.ones_like(sorted_logits, dtype=sorted_logits.dtype) * min_value,
        sorted_logits,
    )
    threshold = tf.expand_dims(tf.reduce_max(applyed_sorted_logits, axis=-1), 1)
    return tf.where(
        logits < threshold, tf.ones_like(logits, dtype=logits.dtype) * compat.FLOAT_MIN, logits
    )


def sequence_sampling_search(symbols_to_logits_fn,
                             generation_initializer,
                             sample_next_word_fn,
                             sample_num,
                             extra_decode_length=50,
                             maximum_decode_length=256,
                             minimum_decode_length=0,
                             padded_decode=False,
                             dtype=None, ):
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
        sample_next_word_fn:
        sample_num:
        extra_decode_length: An int scalar. The real search steps will be
            `encoder_inputs_maxlen` + `extra_decode_length`.
        maximum_decode_length: An int scalar, indicating the maximum decode length. Prediction
            outputs will be padded to this length.
        minimum_decode_length: An int scalar, indicating the minimum decode length.
        padded_decode: whether the autoregressive decoding runs with input data padded to the decode_max_length.
        dtype: A string or a tensorflow data type used for score computation. If None, auto set
            to the GLOBAL_FLOATX.

    Returns:
        A tuple of (hypothesis, scores):
            hypothesis: int32 tensor with shape [batch_size * top_k, decode_length]
            scores: float tensor with shape [batch_size * top_k]
    """
    _ = dtype
    decoder_input_ids = generation_initializer["decoder_input"]
    decoding_cache = generation_initializer["decoder_internal_cache"]
    encoder_inputs_maxlen = generation_initializer["encoder_inputs_maxlen"]
    eos_id = generation_initializer["eos_id"]
    unk_id = generation_initializer["unk_id"]
    batch_size = tf.shape(decoder_input_ids)[0]
    initial_finished = tf.tile([False], [batch_size * sample_num])
    initial_decoder_input_ids = layer_utils.stack_beam_size(decoder_input_ids, sample_num)
    initial_time = tf.constant(0, dtype=tf.int32)
    initial_cache = tf.nest.map_structure(
        lambda x: layer_utils.stack_beam_size(x, sample_num), decoding_cache
    )
    # [time, batch_size * sample_num]
    if padded_decode:
        initial_predicted_ids = tf.zeros([batch_size * sample_num, maximum_decode_length],
                                         dtype=tf.int32)
    else:
        initial_predicted_ids = tf.zeros([batch_size * sample_num, 0],
                                         dtype=tf.int32)
    # Create state dictionary
    search_state = {
        _StateKeys.TIME_STEP: initial_time,
        _StateKeys.INPUT_IDS: tf.cast(initial_decoder_input_ids, tf.int32),
        _StateKeys.CACHE: initial_cache,
        _StateKeys.FINISHED_FLAGS: initial_finished,
        _StateKeys.PREDICTED_IDS: initial_predicted_ids,
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
                layer_utils.static_tensorshape, initial_cache
            ),
            _StateKeys.FINISHED_FLAGS: tf.TensorShape([None]),
            _StateKeys.PREDICTED_IDS: tf.TensorShape([None, maximum_decode_length]),
        }
    else:
        search_state_shape_invariants = {
            _StateKeys.TIME_STEP: tf.TensorShape([]),
            _StateKeys.INPUT_IDS: tf.TensorShape([None]),
            _StateKeys.CACHE: tf.nest.map_structure(
                layer_utils.dynamic_tensorshape_except_last_dim, initial_cache
            ),
            _StateKeys.FINISHED_FLAGS: tf.TensorShape([None]),
            _StateKeys.PREDICTED_IDS: tf.TensorShape([None, None]),
        }

    maximum_search_steps = tf.minimum(
        encoder_inputs_maxlen + extra_decode_length, maximum_decode_length)
    maximum_search_steps = tf.maximum(maximum_search_steps, minimum_decode_length)

    def search_step(state):
        """ Beam search step. """
        # [batch_size * beam_size, vocab_size]
        logits = _calculate_logits(
            state=state, unk_id=unk_id,
            symbols_to_logits_fn=symbols_to_logits_fn)
        # filter out eos
        vocab_size = logits.get_shape().as_list()[-1]
        eos_bias = layer_utils.one_entry_bias(
            on_entry=eos_id, num_entries=vocab_size,
            on_value=compat.FLOAT_MIN, off_value=0.0, dtype=logits.dtype)
        eos_coefficient = layer_utils.one_entry_bias(
            on_entry=eos_id, num_entries=vocab_size,
            on_value=0.0, off_value=1.0, dtype=logits.dtype)
        logits = tf.cond(tf.less(state[_StateKeys.TIME_STEP], minimum_decode_length - 1),
                         lambda: eos_coefficient * logits + eos_bias,
                         lambda: logits)

        # compute log probs and generate next token ids according to beam scores
        logits = sample_next_word_fn(logits=logits)
        # [batch*beam, 1]
        sample_ids = tf.random.categorical(logits, num_samples=1, dtype=tf.int32)
        next_predicted_ids = state[_StateKeys.PREDICTED_IDS]
        if padded_decode:
            next_predicted_ids = tf.transpose(tf.tensor_scatter_nd_update(
                tf.transpose(next_predicted_ids), [[state[_StateKeys.TIME_STEP]]],
                tf.transpose(sample_ids)))
        else:
            next_predicted_ids = tf.concat([next_predicted_ids, sample_ids], axis=1)

        next_cache = state[_StateKeys.CACHE]
        pre_finish = state[_StateKeys.FINISHED_FLAGS]
        cur_finished = tf.equal(eos_id, tf.squeeze(sample_ids, 1))
        next_finished = tf.where(pre_finish, True, cur_finished)
        state.update({
            _StateKeys.TIME_STEP: state[_StateKeys.TIME_STEP] + 1,
            _StateKeys.INPUT_IDS: tf.squeeze(sample_ids, 1),
            _StateKeys.CACHE: next_cache,
            _StateKeys.FINISHED_FLAGS: next_finished,
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
        parallel_iterations=1,
    )
    res = res[0]
    hypothesis = res[_StateKeys.PREDICTED_IDS]
    # padding to fixed length output
    if not padded_decode:
        hypothesis = tf.pad(
            hypothesis,
            paddings=tf.convert_to_tensor(
                [[0, 0], [0, maximum_decode_length - res[_StateKeys.TIME_STEP]]],
                dtype=tf.int32),
            mode="CONSTANT",
            constant_values=eos_id)

    return hypothesis


@register_search_layer("TopSampling")
class Sampling(SequenceSearch):
    def __init__(self, args):
        super(Sampling, self).__init__()
        self._top_p = args["top_p"]
        self._top_k = args["top_k"]
        self._sample_num = args["sample_num"]
        self._maximum_decode_length = args["maximum_decode_length"]
        self._extra_decode_length = args["extra_decode_length"]
        self._minimum_decode_length = args["minimum_decode_length"]
        if not self._minimum_decode_length:
            self._minimum_decode_length = 0
        if self._top_p < 1 and self._top_k > 0:
            raise NotImplementedError("Not implemented search logic when top_k > 0 and top_p < 1.")
        self._padded_decode = args["padded_decode"]

    @staticmethod
    def class_or_method_args():
        return [
            Flag("sample_num", dtype=Flag.TYPE.INTEGER,
                 default=1, help="The number of copies for each input item.", ),
            Flag("top_k", dtype=Flag.TYPE.INTEGER,
                 default=0, help="The number of token in each step for top_k sampling."),
            Flag("top_p", dtype=Flag.TYPE.FLOAT,
                 default=1.0, help="The threshold for cumulated probability."),
            Flag("maximum_decode_length", dtype=Flag.TYPE.INTEGER,
                 default=None, help="The maximum decoding length of sampling."),
            Flag("minimum_decode_length", dtype=Flag.TYPE.INTEGER,
                 default=0, help="The minimum decoding length of sampling."),
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
        ]

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
            assert max_data_len, (
                "`maximum_decode_length` must be provided " "when `max_data_len` is not provided.")
            maximum_decode_length = self._extra_decode_length + max_data_len
        if self._minimum_decode_length >= maximum_decode_length:
            raise ValueError("`minimum_decode_length` must be less than maximum decode length.")

        with tf.name_scope("top_sampling"):
            decode_padded_length = self._maximum_decode_length if self._padded_decode else None
            symbols_to_logits_fn, generation_initializer = self._model.get_symbols_to_logits_fn(
                parsed_inp, is_training=False, is_inference=True,
                decode_padded_length=decode_padded_length)

        def _sample_fn(logits):
            if self._top_p < 1:
                return top_p_logits(logits, self._top_p)
            return top_k_logits(logits, self._top_k)

        hypothesis = sequence_sampling_search(
            symbols_to_logits_fn,
            generation_initializer,
            sample_next_word_fn=_sample_fn,
            sample_num=self._sample_num,
            extra_decode_length=self._extra_decode_length,
            maximum_decode_length=self._maximum_decode_length,
            minimum_decode_length=self._minimum_decode_length,
            padded_decode=self._padded_decode)

        return hypothesis,
