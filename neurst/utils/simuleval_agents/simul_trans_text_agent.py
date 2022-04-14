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
import logging

global tf
import tensorflow as tf
from simuleval import DEFAULT_EOS, READ_ACTION, WRITE_ACTION
from simuleval.agents import TextAgent
from simuleval.states import ListEntry, QueueEntry, TextStates

from neurst.tasks import build_task
from neurst.utils.checkpoints import restore_checkpoint_if_possible
from neurst.utils.configurable import ModelConfigs
from neurst.utils.misc import flatten_string_list
from neurst.utils.simuleval_agents import register_agent

BOW_PREFIX = "\u2581"
logger = logging.getLogger('simuleval.agent')


def build_task_and_model(model_dir, wait_k):
    model_dirs = flatten_string_list(model_dir)
    cfgs = ModelConfigs.load(model_dirs[0])
    cfgs["task.params"]["wait_k"] = wait_k
    task = build_task(cfgs)
    models = []
    for md in model_dirs:
        models.append(task.build_model(ModelConfigs.load(md)))
        restore_checkpoint_if_possible(models[-1], md)
    return task, models


@register_agent
class SimulTransTextAgent(TextAgent):

    def __init__(self, args):
        super().__init__(args)
        # Initialize your agent here, for example load model, vocab, etc
        self.wait_k = args.wait_k
        self.task, self.models = build_task_and_model(args.model_dir, self.wait_k)
        self.force_segment = args.force_segment
        self.max_len = args.max_len
        self.src_pipeline = self.task._src_data_pipeline
        self.trg_pipeline = self.task._trg_data_pipeline
        self.words = ['lbs.', 'Dr.', 'Prof.', 'Mr.', 'Mrs.', 'Ms.']

    def build_states(self, args, client, sentence_id):
        # Initialize states here, for example add customized entry to states
        # This function will be called at beginning of every new sentence
        states = TextStates(args, client, sentence_id, self)
        self.initialize_states(states)
        return states

    def initialize_states(self, states):
        states.units.source = ListEntry()
        states.units.target = ListEntry()
        states.segments.source = ListEntry()
        states.segments.target = ListEntry()
        states.unit_queue.source = QueueEntry()
        states.unit_queue.target = QueueEntry()
        states.encoder_cache = [{} for _ in self.models]
        states.decoder_cache = [{} for _ in self.models]
        states.segment = False
        states.encoding_time, states.decoding_time = 0, 0

    @staticmethod
    def add_args(parser):
        parser.add_argument('--model-dir', type=str, required=True, dest="model_dir",
                            help='Path to the MT model(s).')
        parser.add_argument("-k", "--wait-k", type=int, dest="wait_k", default=3)
        parser.add_argument("--force-segment", default=False, action="store_true",
                            dest="force_segment")
        parser.add_argument("--max-len", type=int, default=200, dest="max_len",
                            help="Max length of translation")

    def segment_to_units(self, segment, states):
        """Split a full segment into sunword units."""
        # check if end with .?!.\"
        if self.force_segment:
            if segment.endswith('."') or segment.endswith('?"') or segment.endswith('!"'):
                if segment.endswith('..."'):
                    states.segment = True
                else:
                    if len(segment) > 2 and not segment[-3].isupper() and segment[:-1] not in self.words:
                        states.segment = True
            if segment.endswith('.') or segment.endswith('?') or segment.endswith('!'):
                if segment.endswith('...'):
                    states.segment = True
                else:
                    if len(segment) > 1 and not segment[-2].isupper() and segment not in self.words:
                        states.segment = True

        substrs = self.src_pipeline.encode(segment)[:-1]
        # Add eos at the end of this sub-sentence.
        if self.force_segment and states.segment:
            substrs.append(self.src_pipeline.meta["eos_id"])
        return substrs

    def units_to_segment(self, units, states):
        """ Merge subword units to a full segment.

        Args:
            units: states.unit_queue.target
            states: States dictionary.

        Returns:
            A segment or None.
        """
        # Get a EOS or superpass max_len, return EOS.
        # When using force_segment, finish present sentence and translate next one.
        if self.trg_pipeline.meta["eos_id"] == units[0] or len(states.segments.target) > self.max_len:
            index = units.pop()
            if self.force_segment and states.status["read"]:
                states.segment = False
                self.initialize_states(states)
                return None
            return DEFAULT_EOS
        # When the target language is Japanese, we only need to return it character by character.
        if self.task._trg_data_pipeline.meta["language"] == "ja":
            index = units.pop()
            token = self.trg_pipeline.tokens[index]
            if BOW_PREFIX == token:
                return None
            if token[0] == BOW_PREFIX:
                return token[1:]
            else:
                return token
        else:
            segment = []
            if None in units.value:
                units.value.remove(None)

            for index in units:
                if index is None:
                    units.pop()
                token = self.trg_pipeline.tokens[index]
                if token.startswith(BOW_PREFIX):
                    if len(segment) == 0:
                        segment += [token.replace(BOW_PREFIX, "")]
                    else:
                        for j in range(len(segment)):
                            units.pop()
                        string_to_return = ["".join(segment)]
                        if self.trg_pipeline.meta["eos_id"] == units[0]:
                            string_to_return += [DEFAULT_EOS]
                        return string_to_return
                else:
                    segment += [token.replace(BOW_PREFIX, "")]
            if (
                len(units) > 0
                and self.trg_pipeline.meta["eos_id"] == units[-1]
                or len(states.units.target) > self.max_len
            ):
                tokens = [self.trg_pipeline.tokens[unit] for unit in units][:-1]
                # finish present sentence and translate next one
                if self.force_segment and states.status["read"]:
                    states.segment = False
                    self.initialize_states(states)
                    return ["".join(tokens).replace(BOW_PREFIX, "")]
                return ["".join(tokens).replace(BOW_PREFIX, "")] + [DEFAULT_EOS]

            return None

    def _indices_from_states(self, states):
        src_indices = tf.convert_to_tensor([states.source.value])
        src_paddings = tf.expand_dims(tf.zeros(tf.shape(src_indices)[1]), 0)
        if self.task._target_begin_of_sentence == 'eos':
            tgt_indices = tf.convert_to_tensor([[self.trg_pipeline.meta["eos_id"]] + states.target.value])
        else:
            tgt_indices = tf.convert_to_tensor([[self.trg_pipeline.meta["bos_id"]] + states.target.value])
        return src_indices, src_paddings, tgt_indices

    def policy(self, states):
        """ Decide to read or write depending on states.

        Returns:
            READ_ACTION/WRITE_ACTION
        """
        # Finish all sentences.
        if self.force_segment and not states.status["read"]:
            # states.status["write"] = False
            return WRITE_ACTION
        # Finish reading last segment of the present sentence.
        if self.force_segment and states.segment:
            # The last segment is splitted into several units in unit_queue. Continue reading.
            if not states.unit_queue.source.empty():
                return READ_ACTION
            else:
                return WRITE_ACTION
        # Finish reading but need eos.
        if not states.status["read"] and states.units.source[-1] != self.src_pipeline.meta["eos_id"]:
            states.units.source.append(self.src_pipeline.meta["eos_id"])
        if not states.unit_queue.source.empty() and states.status["read"]:
            return READ_ACTION
        # consider waitk in segment level rather than subword level
        elif len(states.segments.source) - len(states.segments.target) < self.wait_k and not states.finish_read():
            return READ_ACTION
        else:
            return WRITE_ACTION

    def predict(self, states):
        if self.force_segment and not states.status["read"]:
            return self.trg_pipeline.meta["eos_id"]
        # over translation
        if len(states.units.target) > self.max_len:
            return self.trg_pipeline.meta["eos_id"]

        # predict
        src_indices = states.source.value[states.encoding_time:]
        src_length = len(src_indices)
        trg_input = (self.trg_pipeline.meta["bos_id"] if self.task._target_begin_of_sentence == "bos"
                     else self.trg_pipeline.meta["eos_id"])
        if len(states.target.value) > 0:
            trg_input = states.target.value[-1]
        log_probs_list = []
        for i, model in enumerate(self.models):
            # encode
            if src_length > 0:
                states.encoder_cache[i], states.decoder_cache[i] = model.incremental_encode(
                    {"src": [src_indices], "src_length": [src_length]},
                    states.encoder_cache[i], states.decoder_cache[i],
                    time=states.encoding_time)
            # decode
            logits, states.decoder_cache[i] = model.incremental_decode([trg_input], states.decoder_cache[i],
                                                                       time=states.decoding_time)
            log_probs = tf.nn.log_softmax(logits[0])
            log_probs_list.append(log_probs)
        states.encoding_time = len(states.source.value)
        states.decoding_time += 1
        total_log_probs = tf.math.reduce_logsumexp(
            tf.stack(log_probs_list, axis=0), axis=0) - tf.math.log(len(self.models) * 1.)
        top_k = tf.nn.top_k(total_log_probs, k=1)
        index = top_k.indices.numpy()[0]
        return index
