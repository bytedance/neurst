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
from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import logging
import math
import re
import traceback

import numpy as np
import sacrebleu

from neurst.data.text.character import Character
from neurst.data.text.moses_tokenizer import MosesTokenizer
from neurst.data.text.thai_tokenizer import ThaiTokenizer
from neurst.metrics import register_metric
from neurst.metrics.metric import Metric


def bleu_count(hypothesis, references, max_n=4):
    ret_len_hyp = 0
    ret_len_ref = 0
    ret_clip_count = [0] * max_n
    ret_count = [0] * max_n
    for m in range(len(hypothesis)):
        hyp, ref = hypothesis[m], references[m]
        x = hyp.split()
        y = [r.split() for r in ref]
        x_len = len(x)
        y_len = [len(s) for s in y]
        n_ref = len(ref)

        closest_diff = 9999
        closest_length = 9999
        ref_ngram = dict()

        for i in range(n_ref):
            diff = abs(y_len[i] - x_len)
            if diff < closest_diff:
                closest_diff = diff
                closest_length = y_len[i]
            elif diff == closest_diff and y_len[i] < closest_length:
                closest_length = y_len[i]

            for n in range(max_n):
                sent_ngram = dict()
                for st in range(0, y_len[i] - n):
                    ngram = "%d" % (n + 1)
                    for k in range(n + 1):
                        j = st + k
                        ngram += " %s" % (y[i][j])
                    if ngram not in sent_ngram:
                        sent_ngram[ngram] = 0
                    sent_ngram[ngram] += 1
                for ngram in sent_ngram.keys():
                    if ngram not in ref_ngram or ref_ngram[ngram] < sent_ngram[ngram]:
                        ref_ngram[ngram] = sent_ngram[ngram]

        ret_len_hyp += x_len
        ret_len_ref += closest_length

        for n in range(max_n):
            hyp_ngram = dict()
            for st in range(0, x_len - n):
                ngram = "%d" % (n + 1)
                for k in range(n + 1):
                    j = st + k
                    ngram += " %s" % (x[j])
                if ngram not in hyp_ngram:
                    hyp_ngram[ngram] = 0
                hyp_ngram[ngram] += 1
            for ngram in hyp_ngram.keys():
                if ngram in ref_ngram:
                    ret_clip_count[n] += min(ref_ngram[ngram], hyp_ngram[ngram])
                ret_count[n] += hyp_ngram[ngram]

    return ret_clip_count, ret_count, ret_len_hyp, ret_len_ref


def corpus_bleu(hypothesis, references, max_n=4):
    clip_count, count, total_len_hyp, total_len_ref = bleu_count(hypothesis, references, max_n=max_n)
    brevity_penalty = 1.0
    bleu_scores = []
    for n in range(max_n):
        if count[n] > 0:
            bleu_scores.append(clip_count[n] / count[n])
        else:
            bleu_scores.append(0)
    if total_len_hyp < total_len_ref:
        brevity_penalty = math.exp(1 - total_len_ref / total_len_hyp)

    def my_log(x):
        if x == 0:
            return -9999999999.0
        elif x < 0:
            raise ValueError("x in log(x) must >= 0.")
        return math.log(x)

    log_bleu = 0.0
    for n in range(max_n):
        log_bleu += my_log(bleu_scores[n])
    bleu = brevity_penalty * math.exp(log_bleu / float(max_n))
    return [bleu] + bleu_scores, [brevity_penalty, total_len_hyp / total_len_ref, total_len_hyp, total_len_ref]


def sentence_bleu(hypothesis, references, max_n=4):
    clip_count, count, total_len_hyp, total_len_ref = bleu_count([hypothesis], [references], max_n=max_n)
    brevity_penalty = 1.0
    bleu_scores = []
    for n in range(max_n):
        bleu_scores.append((clip_count[n] + 0.01) / (count[n] + 0.01))  # smoothing
    if total_len_hyp < total_len_ref:
        brevity_penalty = math.exp(1 - total_len_ref / total_len_hyp)

    def my_log(x):
        if x == 0:
            return -9999999999.0
        elif x < 0:
            raise ValueError("x in log(x) must >= 0.")
        return math.log(x)

    log_bleu = 0.0
    for n in range(max_n):
        log_bleu += my_log(bleu_scores[n])
    bleu = brevity_penalty * math.exp(log_bleu / float(max_n))
    return [bleu] + bleu_scores, [brevity_penalty, total_len_hyp / total_len_ref, total_len_hyp, total_len_ref]


def incremental_bleu_count(hypothesis, references, max_n=4):
    ret_len_hyp = []
    ret_len_ref = []
    ret_clip_count = []
    ret_count = []
    for m in range(len(hypothesis)):
        hyp, ref = hypothesis[m], references[m]
        x = hyp.split()
        y = [r.split() for r in ref]
        x_len = len(x)
        y_len = [len(s) for s in y]
        n_ref = len(ref)

        ref_ngram = dict()

        for i in range(n_ref):
            for n in range(max_n):
                sent_ngram = dict()
                for st in range(0, y_len[i] - n):
                    ngram = "%d" % (n + 1)
                    for k in range(n + 1):
                        j = st + k
                        ngram += " %s" % (y[i][j])
                    if ngram not in sent_ngram:
                        sent_ngram[ngram] = 0
                    sent_ngram[ngram] += 1
                for ngram in sent_ngram.keys():
                    if ngram not in ref_ngram or ref_ngram[ngram] < sent_ngram[ngram]:
                        ref_ngram[ngram] = sent_ngram[ngram]
        y_len = sorted(y_len)
        ret_len_hyp.append([])
        ret_len_ref.append([])
        ret_clip_count.append([])
        ret_count.append([])

        hyp_ngram = dict()
        p_closest = 0
        for i in range(x_len):
            if i == 0:
                ret_clip_count[-1].append([0] * max_n)
                ret_count[-1].append([0] * max_n)
            else:
                ret_clip_count[-1].append(copy.deepcopy(ret_clip_count[-1][-1]))
                ret_count[-1].append(copy.deepcopy(ret_count[-1][-1]))

            j = i + 1
            ret_len_hyp[-1].append(i + 1)
            if j > y_len[p_closest]:
                while j > y_len[p_closest] and p_closest < n_ref - 1:
                    p_closest += 1
            tmp_closest_diff = 9999
            tmp_closest_len = 9999
            if p_closest > 0 and (j - y_len[p_closest - 1]) < tmp_closest_diff:
                tmp_closest_diff = j - y_len[p_closest - 1]
                tmp_closest_len = y_len[p_closest - 1]
            if p_closest < n_ref and (y_len[p_closest] - j) < tmp_closest_diff:
                tmp_closest_diff = y_len[p_closest] - j
                tmp_closest_len = y_len[p_closest]

            ret_len_ref[-1].append(tmp_closest_len)
            for n in range(max_n):
                st = i - n
                if st >= 0:
                    ngram = "%d" % (n + 1)
                    for k in range(n + 1):
                        j = st + k
                        ngram += " %s" % (x[j])
                    if ngram not in hyp_ngram:
                        hyp_ngram[ngram] = 0
                    hyp_ngram[ngram] += 1
                    ret_count[-1][-1][n] += 1
                    if ngram in ref_ngram and hyp_ngram[ngram] <= ref_ngram[ngram]:
                        ret_clip_count[-1][-1][n] += 1

    return ret_clip_count, ret_count, ret_len_hyp, ret_len_ref


def incremental_sent_bleu(hypothesis, references, max_n=4):
    clip_count, count, total_len_hyp, total_len_ref = incremental_bleu_count([hypothesis], [references], max_n=max_n)
    clip_count = clip_count[0]
    count = count[0]
    total_len_hyp = total_len_hyp[0]
    total_len_ref = total_len_ref[0]
    n_len = len(clip_count)
    ret = []
    for i in range(n_len):
        brevity_penalty = 1.0
        bleu_scores = []
        bleu = 0
        for n in range(max_n):
            if count[i][n] > 0:
                bleu_scores.append(clip_count[i][n] / count[i][n])
            else:
                bleu_scores.append(0)
        if total_len_hyp[i] < total_len_ref[i]:
            brevity_penalty = math.exp(1 - total_len_ref[i] / total_len_hyp[i])

        def my_log(x):
            if x == 0:
                return -9999999999.0
            elif x < 0:
                raise ValueError("x in log(x) must >= 0.")
            return math.log(x)

        log_bleu = 0.0
        for n in range(max_n):
            log_bleu += my_log(bleu_scores[n])
        bleu = brevity_penalty * math.exp(log_bleu / float(max_n))
        ret.append(bleu)
    return ret


def incremental_test_corpus_bleu(hypothesis, references, max_n=4):
    assert (len(hypothesis) == len(references))
    tmp_clip_count, tmp_count, tmp_total_len_hyp, tmp_total_len_ref = incremental_bleu_count(hypothesis, references,
                                                                                             max_n=max_n)
    clip_count = [0] * 4
    count = [0] * 4
    total_len_hyp = 0
    total_len_ref = 0
    for i in range(len(hypothesis)):
        for n in range(4):
            clip_count[n] += tmp_clip_count[i][-1][n]
            count[n] += tmp_count[i][-1][n]
        total_len_hyp += tmp_total_len_hyp[i][-1]
        total_len_ref += tmp_total_len_ref[i][-1]
    brevity_penalty = 1.0
    bleu_scores = []
    bleu = 0
    for n in range(max_n):
        if count[n] > 0:
            bleu_scores.append(clip_count[n] / count[n])
        else:
            bleu_scores.append(0)
    if total_len_hyp < total_len_ref:
        brevity_penalty = math.exp(1 - total_len_ref / total_len_hyp)

    def my_log(x):
        if x == 0:
            return -9999999999.0
        elif x < 0:
            raise ValueError("x in log(x) must >= 0.")
        return math.log(x)

    log_bleu = 0.0
    for n in range(max_n):
        log_bleu += my_log(bleu_scores[n])
    bleu = brevity_penalty * math.exp(log_bleu / float(max_n))
    return [bleu] + bleu_scores, [brevity_penalty, total_len_hyp / total_len_ref, total_len_hyp, total_len_ref]


def commonly_tokenize(s):
    """ Tokenizes sentence according to multi-bleu-detok.perl & mteval-v13a.pl """
    # language-independent part
    s = re.sub(r"-\n", r"", s)  # strip end-of-line hyphenation and join lines
    s = re.sub(r"\n", r" ", s)  # join lines
    s = re.sub(r"&quot;", r'"', s)  # convert SGML tag for quote to "
    s = re.sub(r"&amp;", r"&", s)  # convert SGML tag for ampersand to &
    s = re.sub(r"&lt;", r"<", s)  # convert SGML tag for less-than to >
    s = re.sub(r"&gt;", r">", s)  # convert SGML tag for greater-than to <
    s = " " + s + " "
    # language-dependent part (assuming Western languages):
    s = re.sub(r"([\{-~\[-` -&\(-\+:-@\/])", r" \1 ", s)  # tokenize punctuation
    s = re.sub(r"([^0-9])([\.,])", r"\1 \2 ", s)  # tokenize period and comma unless preceded by a digit
    s = re.sub(r"([\.,])([^0-9])", r" \1 \2", s)  # tokenize period and comma unless followed by a digit
    # s = re.sub(r"([\.,])([0-9])", r" \1 \2", s)  # tokenize period and comma unless followed by a digit
    s = re.sub(r"([0-9])(-)", r"\1 \2 ", s)  # tokenize dash when preceded by a digit

    return " ".join(s.strip().split())


ESCAPE_AMPERSAND = r'&', r'&amp;'
ESCAPE_PIPE = r'|', r'&#124;'
ESCAPE_LEFT_ANGLE_BRACKET = r'<', r'&lt;'
ESCAPE_RIGHT_ANGLE_BRACKET = r'>', r'&gt;'
ESCAPE_SINGLE_QUOTE = r"'", r"&apos;"
ESCAPE_DOUBLE_QUOTE = r'"', r'&quot;'
ESCAPE_LEFT_SQUARE_BRACKET = r"[", r"&#91;"
ESCAPE_RIGHT_SQUARE_BRACKET = r"]", r"&#93;"

ESCAPE_LIST = [ESCAPE_AMPERSAND,
               ESCAPE_PIPE,
               ESCAPE_LEFT_ANGLE_BRACKET,
               ESCAPE_RIGHT_ANGLE_BRACKET,
               ESCAPE_SINGLE_QUOTE,
               ESCAPE_DOUBLE_QUOTE,
               ESCAPE_LEFT_SQUARE_BRACKET,
               ESCAPE_RIGHT_SQUARE_BRACKET]


def unescape(s):
    for repl, patt in ESCAPE_LIST:
        s = re.sub(patt, repl, s)
    return s


@register_metric(["sacre_bleu",
                  "tok_bleu",
                  "detok_bleu",
                  "chrf",
                  "uncased_sacre_bleu",
                  "uncased_tok_bleu",
                  "uncased_detok_bleu",
                  "uncased_chrf"])
class BLEU(Metric):

    def __init__(self, language="en", *args, **kwargs):
        """ Initializes.

        Args:
            language: The language.
        """
        _ = args
        _ = kwargs
        super(BLEU, self).__init__()
        self._language = language
        if language in ["zh", "ja", "ko", "km"]:
            self._tokenize_fn = lambda x: Character.to_character(x, language=language)
        elif language == "th":
            tokenizer = ThaiTokenizer()
            self._tokenize_fn = lambda x: tokenizer.tokenize(x, return_str=True)
        else:
            tokenizer = MosesTokenizer(language=language)
            self._tokenize_fn = lambda x: tokenizer.tokenize(x, return_str=True)
        self._sacre_tokenize_str = "13a"
        if language == "zh":
            self._sacre_tokenize_str = "zh"
        elif language == "ja":
            self._sacre_tokenize_str = "ja-mecab"

    @staticmethod
    def _tokenize(ss, tok_fn, lc=False):
        assert isinstance(ss, list)
        if isinstance(ss[0], str):
            return [tok_fn(unescape(x.lower() if lc else x)) for x in ss]
        return [[tok_fn(unescape(x.lower() if lc else x)) for x in xx] for xx in ss]

    def set_groundtruth(self, groundtruth):
        """ Setup inside groundtruth.

        Args:
            groundtruth: A list of references,
                [sent0_ref, sent1_ref, ...]
                    or a list of multiple references,
                    [[sent0_ref0, sent1_ref0, ...],
                    [sent0_ref1, sent1_ref1, ...],
                    ......]
        """
        assert isinstance(groundtruth, list)
        if isinstance(groundtruth[0], str):
            groundtruth = [groundtruth]
        self._refs_for_sacre = groundtruth
        refs = list(map(list, zip(*groundtruth)))
        self._refs_for_tok = self._tokenize(refs, self._tokenize_fn, lc=False)
        self._uncased_refs_for_tok = self._tokenize(refs, self._tokenize_fn, lc=True)
        self._refs_for_tok = [
            [self._tokenize_fn(r) for r in rr] for rr in refs]
        self._uncased_refs_for_tok = [
            [self._tokenize_fn(r.lower()) for r in rr] for rr in refs]

    def tok_bleu(self, hypo, groundtruth=None, lc=False):
        tok_hypos = self._tokenize(hypo, self._tokenize_fn, lc=lc)
        if groundtruth is None:
            ref = self._uncased_refs_for_tok if lc else self._refs_for_tok
        else:
            if isinstance(groundtruth[0], str):
                groundtruth = [groundtruth]
            ref = self._tokenize(list(map(list, zip(*groundtruth))), self._tokenize_fn, lc=lc)
        try:
            bleu, _ = corpus_bleu(tok_hypos, ref)
            bleu = bleu[0]
        except IndexError:
            logging.info("Found empty lines.")
            print(traceback.format_exc())
            bleu = 0.
        except ZeroDivisionError:
            logging.info("Empty reference")
            print(traceback.format_exc())
            bleu = 0.
        return bleu * 100

    def detok_bleu(self, hypo, groundtruth=None, lc=False):
        # tok_hypos = self._tokenize(hypo, self._default_tokenize_fn, lc=lc)
        # if groundtruth is None:
        #     ref = self._uncased_refs_for_detok if lc else self._refs_for_detok
        # else:
        #     if isinstance(groundtruth[0], str):
        #         groundtruth = [groundtruth]
        #     ref = self._tokenize(list(map(list, zip(*groundtruth))), self._default_tokenize_fn, lc=lc)
        # try:
        #     bleu, _ = corpus_bleu(tok_hypos, ref)
        #     bleu = bleu[0]
        # except IndexError:
        #     logging.info("Found empty lines.")
        #     print(traceback.format_exc())
        #     bleu = 0.
        # except ZeroDivisionError:
        #     logging.info("Empty reference")
        #     print(traceback.format_exc())
        #     bleu = 0.
        # return bleu * 100
        if self._language in ["ko", "km", "th"]:
            return self.tok_bleu(hypo, groundtruth, lc)
        return self.sacre_bleu(hypo, groundtruth, lc)

    def sacre_bleu(self, hypo, groundtruth=None, lc=False):
        if groundtruth is None:
            ref = self._refs_for_sacre
        else:
            if isinstance(groundtruth[0], str):
                ref = [groundtruth]
            else:
                ref = groundtruth
        try:
            bleu = sacrebleu.corpus_bleu(
                hypo, ref, lowercase=lc, tokenize=self._sacre_tokenize_str)
            return bleu.score
        except IndexError:
            logging.info("Found empty lines.")
            print(traceback.format_exc())
            return 0.
        except ZeroDivisionError:
            logging.info("Empty reference")
            print(traceback.format_exc())
            return 0.

    def chrf(self, hypo, groundtruth=None, lc=False):
        if groundtruth is None:
            ref = self._refs_for_sacre
        else:
            if isinstance(groundtruth[0], str):
                ref = [groundtruth]
            else:
                ref = groundtruth
        try:
            chrf = sacrebleu.corpus_chrf([(x.lower() if lc else x) for x in hypo],
                                         [[(x.lower() if lc else x) for x in y] for y in ref])
            return chrf.score
        except IndexError:
            logging.info("Found empty lines.")
            print(traceback.format_exc())
            return 0.
        except ZeroDivisionError:
            logging.info("Empty reference")
            print(traceback.format_exc())
            return 0.

    def get_value(self, result):
        if isinstance(result, (float, np.float32, np.float64)):
            return result
        if self._flag in result:
            return result[self._flag]
        if self._flag.lower() in result:
            return result[self._flag.lower()]
        if self._language in ["ko", "km"]:
            return result["tok_bleu"]
        return result["sacre_bleu"]

    def call(self, hypothesis, groundtruth=None):
        """ Returns the BLEU result dict. """
        return {
            "sacre_bleu": self.sacre_bleu(hypothesis, groundtruth),
            "tok_bleu": self.tok_bleu(hypothesis, groundtruth),
            "detok_bleu": self.detok_bleu(hypothesis, groundtruth),
            "chrf": self.chrf(hypothesis, groundtruth),
            "uncased_sacre_bleu": self.sacre_bleu(hypothesis, groundtruth, lc=True),
            "uncased_tok_bleu": self.tok_bleu(hypothesis, groundtruth, lc=True),
            "uncased_detok_bleu": self.detok_bleu(hypothesis, groundtruth, lc=True),
            "uncased_chrf": self.chrf(hypothesis, groundtruth, lc=True)}
