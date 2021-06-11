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
import collections

import tensorflow as tf
from absl import app, logging

import neurst.utils.flags_core as flags_core
from neurst.training.training_utils import minimal_multiple


def generate_vocab(input, output, min_frequency, max_vocab_size,
                   lowercase=False, extra_slots=None):
    with tf.io.gfile.GFile(input, "r") as finput:
        # Counter for all tokens in the vocabulary
        cnt = collections.Counter()

        for line in finput:
            if lowercase:
                line = line.lower()
            tokens = line.strip().split()
            tokens = [_ for _ in tokens if len(_) > 0]
            cnt.update(tokens)

    extra_slots = minimal_multiple(len(cnt) + extra_slots + 3, 8) - len(cnt) - 3
    extra_slots_list = []
    if extra_slots > 0:
        idx = 0
        while len(extra_slots_list) < extra_slots:
            _txt = ("EXTRA_SLOT%.3d" % idx)
            idx += 1
            if _txt in cnt:
                continue
            extra_slots_list.append(_txt)

    logging.info("Found %d unique tokens in the vocabulary.", len(cnt))

    # Filter tokens below the frequency threshold
    if min_frequency > 0:
        filtered_tokens = [(w, c) for w, c in cnt.most_common()
                           if c >= min_frequency]
        cnt = collections.Counter(dict(filtered_tokens))

    logging.info("Found %d unique tokens with frequency > %d.",
                 len(cnt), min_frequency)

    # Sort tokens by 1. frequency 2. lexically to break ties
    word_with_counts = cnt.most_common()
    word_with_counts = sorted(
        word_with_counts, key=lambda x: (x[1], x[0]), reverse=True)

    # Take only max-vocab
    if max_vocab_size is not None and max_vocab_size > 0:
        word_with_counts = word_with_counts[:max_vocab_size]

    with tf.io.gfile.GFile(output, "w") as foutput:
        # extra slots
        for x in extra_slots_list:
            foutput.write("{}\t{}\n".format(x, 1000))
        logging.info("Plus extra %d slots to the vocabulary in the front.", len(extra_slots_list))
        for word, count in word_with_counts:
            foutput.write("{}\t{}\n".format(word, count))


FLAG_LIST = [
    flags_core.Flag("min_frequency", dtype=flags_core.Flag.TYPE.INTEGER, default=0,
                    help="Minimum frequency of a word to be included in the vocabulary."),
    flags_core.Flag("max_vocab_size", dtype=flags_core.Flag.TYPE.INTEGER, default=None,
                    help="Maximum number of tokens in the vocabulary."),
    flags_core.Flag("lowercase", dtype=flags_core.Flag.TYPE.BOOLEAN, default=None,
                    help="If set to true, downcase all text before processing."),
    flags_core.Flag("input", dtype=flags_core.Flag.TYPE.STRING, default=None,
                    help="Input full vocabulary file."),
    flags_core.Flag("output", dtype=flags_core.Flag.TYPE.STRING, default=None,
                    help="Output final vocabulary file."),
    flags_core.Flag("extra_slots", dtype=flags_core.Flag.TYPE.INTEGER, default=0,
                    help="Extra slots in the vocabulary.")
]


def _main(_):
    arg_parser = flags_core.define_flags(FLAG_LIST, with_config_file=False)
    args, remaining_argv = flags_core.intelligent_parse_flags(FLAG_LIST, arg_parser)
    flags_core.verbose_flags(FLAG_LIST, args, remaining_argv)
    generate_vocab(
        input=args["input"],
        output=args["output"],
        min_frequency=args["min_frequency"],
        max_vocab_size=args["max_vocab_size"],
        lowercase=args["lowercase"],
        extra_slots=args["extra_slots"])


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app.run(_main, argv=["pseudo.py"])
