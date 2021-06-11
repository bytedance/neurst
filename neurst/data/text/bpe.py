import tensorflow as tf
from absl import logging

from neurst.data.text import register_tokenizer
from neurst.data.text.tokenizer import Tokenizer


@register_tokenizer
class BPE(Tokenizer):

    def __init__(self,
                 glossaries=None,
                 separator='@@',
                 vocabulary=None,
                 version=(0, 2),
                 subtokenizer_codes=None,
                 **kwargs):
        _ = kwargs
        super(BPE, self).__init__(language=None, glossaries=glossaries)
        self.glossaries = glossaries if glossaries else []
        self.version = version
        # some hacking to deal with duplicates (only consider first instance)

        self.separator = separator
        if vocabulary:
            if isinstance(vocabulary, str):
                with tf.io.gfile.GFile(vocabulary) as fp:
                    self.vocab = [line.strip().split()[0] for line in fp]
            else:
                assert isinstance(vocabulary, list), f"Unsupported type of vocabulary: {type(vocabulary)}"
                self.vocab = [line.strip().split()[0] for line in vocabulary]
        else:
            self.vocab = []
        self._built = False
        if subtokenizer_codes:
            self.init_subtokenizer(subtokenizer_codes)

    def init_subtokenizer(self, codes):
        if isinstance(codes, str):
            with tf.io.gfile.GFile(codes, "r") as fp:
                codes = [line.strip() for line in fp][1:]
        elif not isinstance(codes, list):
            raise ValueError("Not supported type of codes: {}.".format(type(codes)))
        if codes[0].startswith("#version"):
            codes = codes[1:]
        self.bpe_codes = [tuple(item.split()) for item in codes]
        self.bpe_codes = dict([(code, i) for (i, code) in reversed(list(enumerate(self.bpe_codes)))])
        self.bpe_codes_reverse = dict([(pair[0] + pair[1], pair) for pair, i in self.bpe_codes.items()])
        self._built = True

    @staticmethod
    def isolate_glossary(word, glossary):
        """
        Isolate a glossary present inside a word.

        Returns a list of subwords. In which all 'glossary' glossaries are isolated

        For example, if 'USA' is the glossary and '1934USABUSA' the word, the return value is:
            ['1934', 'USA', 'B', 'USA']
        """
        if word == glossary or glossary not in word:
            return [word]
        else:
            splits = word.split(glossary)
            segments = [segment.strip() for split in splits[:-1] for segment in [split, glossary] if segment != '']
            return segments + [splits[-1].strip()] if splits[-1] != '' else segments

    def _isolate_glossaries(self, word):
        word_segments = [word]
        for gloss in self.glossaries:
            word_segments = [out_segments for segment in word_segments
                             for out_segments in self.isolate_glossary(segment, gloss)]
        return word_segments

    @staticmethod
    def get_pairs(word):
        """Return set of symbol pairs in a word.

        word is represented as tuple of symbols (symbols being variable-length strings)
        """
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    @staticmethod
    def recursive_split(segment, bpe_codes, vocab, separator, final=False):
        """Recursively split segment into smaller units (by reversing BPE merges)
        until all units are either in-vocabulary, or cannot be split futher."""

        try:
            if final:
                left, right = bpe_codes[segment + '</w>']
                right = right[:-4]
            else:
                left, right = bpe_codes[segment]
        except KeyError:
            # sys.stderr.write('cannot split {0} further.\n'.format(segment))
            yield segment
            return

        if left + separator in vocab:
            yield left
        else:
            for item in BPE.recursive_split(left, bpe_codes, vocab, separator, False):
                yield item

        if (final and right in vocab) or (not final and right + separator in vocab):
            yield right
        else:
            for item in BPE.recursive_split(right, bpe_codes, vocab, separator, final):
                yield item

    @staticmethod
    def check_vocab_and_split(orig, bpe_codes, vocab, separator):
        """Check for each segment in word if it is in-vocabulary,
        and segment OOV segments into smaller units by reversing the BPE merge operations"""

        out = []

        for segment in orig[:-1]:
            if segment + separator in vocab:
                out.append(segment)
            else:
                # sys.stderr.write('OOV: {0}\n'.format(segment))
                for item in BPE.recursive_split(segment, bpe_codes, vocab, separator, False):
                    out.append(item)

        segment = orig[-1]
        if segment in vocab:
            out.append(segment)
        else:
            # sys.stderr.write('OOV: {0}\n'.format(segment))
            for item in BPE.recursive_split(segment, bpe_codes, vocab, separator, True):
                out.append(item)

        return out

    @staticmethod
    def bpe_encode(orig, bpe_codes, bpe_codes_reverse, vocab, separator, version, glossaries=None, cache={}):
        """Encode word based on list of BPE merge operations, which are applied consecutively
        """

        if orig in cache:
            return cache[orig]

        if orig in glossaries:
            cache[orig] = (orig,)
            return (orig,)

        if version == (0, 1):
            word = tuple(orig) + ('</w>',)
        elif version == (0, 2):  # more consistent handling of word-final segments
            word = tuple(orig[:-1]) + (orig[-1] + '</w>',)
        else:
            raise NotImplementedError

        pairs = BPE.get_pairs(word)

        if not pairs:
            return orig

        while True:
            bigram = min(pairs, key=lambda pair: bpe_codes.get(pair, float('inf')))
            if bigram not in bpe_codes:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = BPE.get_pairs(word)

        # don't print end-of-word symbols
        if word[-1] == '</w>':
            word = word[:-1]
        elif word[-1].endswith('</w>'):
            word = word[:-1] + (word[-1].replace('</w>', ''),)

        if vocab:
            word = BPE.check_vocab_and_split(word, bpe_codes_reverse, vocab, separator)

        cache[orig] = word
        return word

    def tokenize(self, text, return_str=False):
        if not self._built:
            raise ValueError("call `init_subtokenizer` at first to initialize the BPE.")
        try:
            tokens = self._convert_to_list(text)
            output = []
            for word in tokens:
                new_word = [out for segment in self._isolate_glossaries(word)
                            for out in BPE.bpe_encode(segment,
                                                      self.bpe_codes,
                                                      self.bpe_codes_reverse,
                                                      self.vocab,
                                                      self.separator,
                                                      self.version,
                                                      self.glossaries)]
                for item in new_word[:-1]:
                    output.append(item + self.separator)
                output.append(new_word[-1])
        except IndexError:
            logging.info(tokens)
            raise IndexError("string index out of range, string", tokens)
        return self._output_wrapper(output, return_str)

    @staticmethod
    def recov_bpe(text, seperator="@@"):
        """ Recovers BPE result.

        Args:
            text: A string.
            seperator:

        Returns:
            The recovered string.
        """
        text = Tokenizer._convert_to_str(text, delimiter=" ").replace(seperator + " ", "").strip()
        if text.endswith("@@"):
            text = text[:-2]
        return text

    def detokenize(self, text, return_str=True):
        return self._output_wrapper(
            self.recov_bpe(text, seperator=self.separator), return_str)
