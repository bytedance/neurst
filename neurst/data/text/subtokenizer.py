import collections
import re
import sys
import unicodedata

import six
import tensorflow as tf
from absl import logging

from neurst.data.text import register_tokenizer
from neurst.data.text.tokenizer import Tokenizer
from neurst.utils.misc import temp_download

RESERVED_TOKENS = []

# Regular expression for unescaping token strings.
# '\u' is converted to '_'
# '\\' is converted to '\'
# '\213;' is converted to unichr(213)
_UNESCAPE_REGEX = re.compile(r"\\u|\\\\|\\([0-9]+);")
_ESCAPE_CHARS = set(u"\\_u;0123456789")

# This set contains all letter (L) & number (N) of unicode chars.
_ALPHANUMERIC_CHAR_SET = set(
    six.unichr(i) for i in range(sys.maxunicode)
    if (unicodedata.category(six.unichr(i)).startswith("L")
        or unicodedata.category(six.unichr(i)).startswith("N")))

# min_count is the minimum number of times a subtoken must appear in the data
# before before it is added to the vocabulary. The value is found using binary
# search to obtain the target vocabulary size.
_MIN_MIN_COUNT = 1  # min value to use when binary searching for min_count
_MAX_MIN_COUNT = 1000  # max value to use when binary searching for min_count


def _generate_alphabet_dict(iterable, reserved_tokens=None):
    """Create set of characters that appear in any element in the iterable."""
    if reserved_tokens is None:
        reserved_tokens = RESERVED_TOKENS
    alphabet = {c for token in iterable for c in token}
    alphabet |= {c for token in reserved_tokens for c in token}
    alphabet |= _ESCAPE_CHARS  # Add escape characters to alphabet set.
    return alphabet


def _save_vocab_file(vocab_file, subtoken_list):
    """Save subtokens to file."""
    with tf.io.gfile.GFile(vocab_file, mode="w") as f:
        for subtoken in subtoken_list:
            f.write("'%s'\n" % subtoken)


def _list_to_index_dict(lst):
    """Create dictionary mapping list items to their indices in the list."""
    return {item: n for n, item in enumerate(lst)}


def _split_token_to_subtokens(token, subtoken_dict, max_subtoken_length):
    """Splits a token into subtokens defined in the subtoken dict."""
    ret = []
    start = 0
    token_len = len(token)
    while start < token_len:
        # Find the longest subtoken, so iterate backwards.
        for end in range(min(token_len, start + max_subtoken_length), start, -1):
            subtoken = token[start:end]
            if subtoken in subtoken_dict:
                ret.append(subtoken)
                start = end
                break
        else:  # Did not break
            # If there is no possible encoding of the escaped token then one of the
            # characters in the token is not in the alphabet. This should be
            # impossible and would be indicative of a bug.
            raise ValueError("Was unable to split token \"%s\" into subtokens." %
                             token)
    return ret


def _count_and_gen_subtokens(token_counts, alphabet, subtoken_dict,
                             max_subtoken_length):
    """Count number of times subtokens appear, and generate new subtokens.

    Args:
      token_counts: dict mapping tokens to the number of times they appear in the
        original files.
      alphabet: list of allowed characters. Used to escape the tokens, which
        guarantees that all tokens can be split into subtokens.
      subtoken_dict: dict mapping subtokens to ids.
      max_subtoken_length: maximum length of subtoken in subtoken_dict.

    Returns:
      A defaultdict mapping subtokens to the number of times they appear in the
      tokens. The dict may contain new subtokens.
    """
    subtoken_counts = collections.defaultdict(int)
    for token, count in six.iteritems(token_counts):
        token = _escape_token(token, alphabet)
        subtokens = _split_token_to_subtokens(token, subtoken_dict,
                                              max_subtoken_length)

        # Generate new subtokens by taking substrings from token.
        start = 0
        for subtoken in subtokens:
            for end in range(start + 1, len(token) + 1):
                new_subtoken = token[start:end]
                subtoken_counts[new_subtoken] += count
            start += len(subtoken)

    return subtoken_counts


def _filter_and_bucket_subtokens(subtoken_counts, min_count):
    """Return a bucketed list of subtokens that are filtered by count.

    Args:
      subtoken_counts: defaultdict mapping subtokens to their counts
      min_count: int count used to filter subtokens

    Returns:
      List of subtoken sets, where subtokens in set i have the same length=i.
    """
    # Create list of buckets, where subtokens in bucket i have length i.
    subtoken_buckets = []
    for subtoken, count in six.iteritems(subtoken_counts):
        if count < min_count:  # Filter out subtokens that don't appear enough
            continue
        while len(subtoken_buckets) <= len(subtoken):
            subtoken_buckets.append(set())
        subtoken_buckets[len(subtoken)].add(subtoken)
    return subtoken_buckets


def _gen_new_subtoken_list(subtoken_counts,
                           min_count,
                           alphabet,
                           reserved_tokens=None):
    """Generate candidate subtokens ordered by count, and new max subtoken length.

    Add subtokens to the candiate list in order of length (longest subtokens
    first). When a subtoken is added, the counts of each of its prefixes are
    decreased. Prefixes that don't appear much outside the subtoken are not added
    to the candidate list.

    For example:
      subtoken being added to candidate list: 'translate'
      subtoken_counts: {'translate':10, 't':40, 'tr':16, 'tra':12, ...}
      min_count: 5

    When 'translate' is added, subtoken_counts is updated to:
      {'translate':0, 't':30, 'tr':6, 'tra': 2, ...}

    The subtoken 'tra' will not be added to the candidate list, because it appears
    twice (less than min_count) outside of 'translate'.

    Args:
      subtoken_counts: defaultdict mapping str subtokens to int counts
      min_count: int minumum count requirement for subtokens
      alphabet: set of characters. Each character is added to the subtoken list to
        guarantee that all tokens can be encoded.
      reserved_tokens: list of tokens that will be added to the beginning of the
        returned subtoken list.

    Returns:
      List of candidate subtokens in decreasing count order, and maximum subtoken
      length
    """
    if reserved_tokens is None:
        reserved_tokens = RESERVED_TOKENS

    # Create a list of (count, subtoken) for each candidate subtoken.
    subtoken_candidates = []

    # Use bucketted list to iterate through subtokens in order of length.
    # subtoken_buckets[i] = set(subtokens), where each subtoken has length i.
    subtoken_buckets = _filter_and_bucket_subtokens(subtoken_counts, min_count)
    max_subtoken_length = len(subtoken_buckets) - 1

    # Go through the list in reverse order to consider longer subtokens first.
    for subtoken_len in range(max_subtoken_length, 0, -1):
        for subtoken in subtoken_buckets[subtoken_len]:
            count = subtoken_counts[subtoken]

            # Possible if this subtoken is a prefix of another token.
            if count < min_count:
                continue

            # Ignore alphabet/reserved tokens, which will be added manually later.
            if subtoken not in alphabet and subtoken not in reserved_tokens:
                subtoken_candidates.append((count, subtoken))

            # Decrement count of the subtoken's prefixes (if a longer subtoken is
            # added, its prefixes lose priority to be added).
            for end in range(1, subtoken_len):
                subtoken_counts[subtoken[:end]] -= count

    # Add alphabet subtokens (guarantees that all strings are encodable).
    subtoken_candidates.extend((subtoken_counts.get(a, 0), a) for a in alphabet)

    # Order subtoken candidates by decreasing count.
    subtoken_list = [t for _, t in sorted(subtoken_candidates, reverse=True)]

    # Add reserved tokens to beginning of the list.
    subtoken_list = reserved_tokens + subtoken_list
    return subtoken_list, max_subtoken_length


def _generate_subtokens(token_counts,
                        alphabet,
                        min_count,
                        num_iterations=4,
                        reserved_tokens=None):
    """Create a list of subtokens in decreasing order of frequency.

    Args:
      token_counts: dict mapping str tokens -> int count
      alphabet: set of characters
      min_count: int minimum number of times a subtoken must appear before it is
        added to the vocabulary.
      num_iterations: int number of iterations to generate new tokens.
      reserved_tokens: list of tokens that will be added to the beginning to the
        returned subtoken list.

    Returns:
      Sorted list of subtokens (most frequent first)
    """
    if reserved_tokens is None:
        reserved_tokens = RESERVED_TOKENS

    # Use alphabet set to create initial list of subtokens
    subtoken_list = reserved_tokens + list(alphabet)
    max_subtoken_length = 1

    # On each iteration, segment all words using the subtokens defined in
    # subtoken_dict, count how often the resulting subtokens appear, and update
    # the dictionary with subtokens w/ high enough counts.
    for i in range(num_iterations):
        logging.info("\tGenerating subtokens: iteration %d", i)
        # Generate new subtoken->id dictionary using the new subtoken list.
        subtoken_dict = _list_to_index_dict(subtoken_list)

        # Create dict mapping subtoken->count, with additional subtokens created
        # from substrings taken from the tokens.
        subtoken_counts = _count_and_gen_subtokens(token_counts, alphabet,
                                                   subtoken_dict,
                                                   max_subtoken_length)

        # Generate new list of subtokens sorted by subtoken count.
        subtoken_list, max_subtoken_length = _gen_new_subtoken_list(
            subtoken_counts, min_count, alphabet, reserved_tokens)

        logging.info("\tVocab size: %d", len(subtoken_list))
    return subtoken_list


def _generate_subtokens_with_target_vocab_size(token_counts,
                                               alphabet,
                                               target_size,
                                               threshold,
                                               min_count=None,
                                               reserved_tokens=None):
    """Generate subtoken vocabulary close to the target size."""
    if reserved_tokens is None:
        reserved_tokens = RESERVED_TOKENS

    if min_count is not None:
        logging.info("Using min_count=%d to generate vocab with target size %d",
                     min_count, target_size)
        return _generate_subtokens(
            token_counts, alphabet, min_count, reserved_tokens=reserved_tokens)

    def bisect(min_val, max_val):
        """Recursive function to binary search for subtoken vocabulary."""
        cur_count = (min_val + max_val) // 2
        logging.info("Binary search: trying min_count=%d (%d %d)", cur_count,
                     min_val, max_val)
        subtoken_list = _generate_subtokens(
            token_counts, alphabet, cur_count, reserved_tokens=reserved_tokens)

        val = len(subtoken_list)
        logging.info("Binary search: min_count=%d resulted in %d tokens", cur_count,
                     val)

        within_threshold = abs(val - target_size) < threshold
        if within_threshold or min_val >= max_val or cur_count < 2:
            return subtoken_list
        if val > target_size:
            other_subtoken_list = bisect(cur_count + 1, max_val)
        else:
            other_subtoken_list = bisect(min_val, cur_count - 1)

        # Return vocabulary dictionary with the closest number of tokens.
        other_val = len(other_subtoken_list)
        if abs(other_val - target_size) < abs(val - target_size):
            return other_subtoken_list
        return subtoken_list

    logging.info("Finding best min_count to get target size of %d", target_size)
    return bisect(_MIN_MIN_COUNT, _MAX_MIN_COUNT)


def _split_string_to_tokens(text, space_wrapper=None):
    """ The pre-tokenization for alphanums.

    Args:
        text: The input text.

    Returns: A list of tokens.
    """
    if not text:
        return []
    ret = []
    token_start = 0
    is_alnum = [c in _ALPHANUMERIC_CHAR_SET for c in text]
    for pos in range(1, len(text)):
        if is_alnum[pos] != is_alnum[pos - 1]:
            token = text[token_start:pos]
            if token != " " or token_start == 0:
                ret.append(token)
            token_start = pos
    final_token = text[token_start:]
    ret.append(final_token)
    if space_wrapper:
        return [i.replace(" ", space_wrapper) for i in ret]
    return ret


def _count_tokens(files, file_byte_limit=1e6):
    """Return token counts of words in the files.

    Samples file_byte_limit bytes from each file, and counts the words that appear
    in the samples. The samples are semi-evenly distributed across the file.

    Args:
      files: List of filepaths
      file_byte_limit: Max number of bytes that will be read from each file.

    Returns:
      Dictionary mapping tokens to the number of times they appear in the sampled
      lines from the files.
    """

    token_counts = collections.defaultdict(int)

    for filepath in files:
        with tf.io.gfile.GFile(filepath, mode="r") as reader:
            file_byte_budget = file_byte_limit
            counter = 0
            lines_to_skip = int(reader.size() / (file_byte_budget * 2))
            for line in reader:
                if counter < lines_to_skip:
                    counter += 1
                else:
                    if file_byte_budget < 0:
                        break
                    line = " ".join(line.strip().split())
                    file_byte_budget -= len(line)
                    counter = 0

                    # Add words to token counts
                    for token in _split_string_to_tokens(line):
                        token_counts[token] += 1
    return token_counts


def _unescape_token(escaped_token):
    """Inverse of _escape_token().

    Args:
        escaped_token: a unicode string

    Returns:
        token: a unicode string
    """

    def match(m):
        if m.group(1) is None:
            return u"_" if m.group(0) == u"\\u" else u"\\"

        try:
            return six.unichr(int(m.group(1)))
        except (ValueError, OverflowError):
            return u"\u3013"  # Unicode for undefined character.

    trimmed = escaped_token[:-1] if escaped_token.endswith("_") \
        else escaped_token
    return _UNESCAPE_REGEX.sub(match, trimmed)


def _escape_token(token, alphabet):
    """Escape away underscores and OOV characters and append '_'.

    This allows the token to be expressed as the concatenation of a list
    of subtokens from the vocabulary. The underscore acts as a sentinel
    which allows us to invertibly concatenate multiple such lists.

    Args:
        token: A unicode string to be escaped.
        alphabet: A set of all characters in the vocabulary's alphabet.

    Returns:
        escaped_token: An escaped unicode string.

    Raises:
        ValueError: If the provided token is not unicode.
    """
    if not isinstance(token, six.text_type):
        raise ValueError("Expected string type for token, got %s" % type(token))

    token = token.replace(u"\\", u"\\\\").replace(u"_", u"\\u")
    ret = [c if c in alphabet and c != u"\n" else r"\%d;" % ord(c) for c in token]  # noqa
    return u"".join(ret) + "_"


@register_tokenizer(["sub_tokenizer", "word_piece", "wordpiece"])
class Subtokenizer(Tokenizer):

    def __init__(self, language, glossaries=None, **kwargs):
        _ = kwargs
        super(Subtokenizer, self).__init__(language=language, glossaries=None)
        self._space_wrapper = "▁"

        self._reserved_tokens = []
        if isinstance(glossaries, list):
            for word in glossaries:
                self._reserved_tokens.append(word)
                self._reserved_tokens.append(word + "_")

        # Initialize the cache to empty.
        self._cache_size = 2 ** 20
        self._cache = [(None, None)] * self._cache_size
        self._built = False

    def init_subtokenizer(self, codes):
        subtoken_list = []
        if isinstance(codes, list):
            subtokens = codes
        else:
            if codes.startswith("http://") or codes.startswith("https://"):
                codes = temp_download(codes)
            with tf.io.gfile.GFile(codes, mode="r") as f:
                subtokens = [line for line in f]
        for subtoken in subtokens:
            subtoken = subtoken.strip()
            if ((subtoken.startswith("'") and subtoken.endswith("'"))
                or (subtoken.startswith('"') and subtoken.endswith('"'))):
                subtoken = subtoken[1:-1]  # Remove surrounding single-quotes
            if subtoken in self._reserved_tokens:
                continue
            if " " in subtoken:
                self._space_wrapper = None
            subtoken_list.append(subtoken)
        self._all_subtoken_strings = subtoken_list + self._reserved_tokens
        # we remember the maximum length of any subtoken to avoid having to
        # check arbitrarily long strings.
        self._max_subtoken_len = max([len(s) for s in self._all_subtoken_strings])
        # Include all characters from all tokens in the alphabet to guarantee
        # any token can be encoded. Include all escaping characters.
        self._alphabet = {c for token in self._all_subtoken_strings for c in token}
        self._alphabet |= _ESCAPE_CHARS
        self._built = True

    def split_string_to_tokens(self, text):
        """ The pre-tokenization for alphanums.

        Args:
            text: The input text.

        Returns: A list of tokens.
        """
        return _split_string_to_tokens(text, self._space_wrapper)

    def join_tokens_to_string(self, tokens):
        """ The post-detokenization for alphanums.

        Args:
            tokens: The input text or a list of token strings.

        Returns: A string.
        """
        s = self._convert_to_list(tokens)
        if self._space_wrapper:
            s = [i.replace(self._space_wrapper, " ") for i in s]
        token_is_alnum = [t[0] in _ALPHANUMERIC_CHAR_SET for t in s]
        ret = []
        for i, token in enumerate(s):
            if i > 0 and token_is_alnum[i - 1] and token_is_alnum[i]:
                ret.append(" ")
            ret.append(token)
        return self._convert_to_str(ret, delimiter="")

    def _escaped_token_to_subtoken_strings(self, escaped_token):
        """ Converts an escaped token string to a list of subtoken strings.
          Note that this algorithm is greedy; it won't produce the "best"
        Args:
            escaped_token: An escaped token string.

        Returns: A list of subtoken strings.
        """
        ret = []
        start = 0
        token_len = len(escaped_token)
        while start < token_len:
            for end in range(min(token_len, start + self._max_subtoken_len), start, -1):
                subtoken = escaped_token[start:end]
                if subtoken in self._all_subtoken_strings:
                    ret.append(subtoken)
                    start = end
                    break
            else:  # Did not break
                # If there is no possible encoding of the escaped token then one
                # characters in the token is not in the alphabet. This should be
                # impossible and would be indicative of a bug.
                assert False, "Token substring not found in subtoken vocabulary"
        return ret

    def _token_to_subtoken_strings(self, token):
        """Converts a token to a list of subtoken stringds.

        Args:
            token: a string.
        Returns:
            a list of subword tokens.
        """
        cache_location = hash(token) % self._cache_size
        cache_key, cache_value = self._cache[cache_location]
        if cache_key == token:
            return cache_value
        escaped_token = _escape_token(token, self._alphabet)
        ret = self._escaped_token_to_subtoken_strings(escaped_token)
        self._cache[cache_location] = (token, ret)
        return ret

    def tokenize(self, text, return_str=False):
        if not self._built:
            raise ValueError("call `init_subtokenizer` at first to initialize the Subtokenizer.")
        words = self.split_string_to_tokens(self._convert_to_str(text))
        ret = []
        for token in words:
            ret.extend(self._token_to_subtoken_strings(token))
        return self._output_wrapper(ret, return_str=return_str)

    def detokenize(self, text, return_str=True):
        processed = False
        if isinstance(text, str):
            if " " in text:
                s = "".join(text.strip().split())
                processed = True
        if not processed:
            s = self._convert_to_str(text, delimiter="")
        ret = []
        for token in s.strip().split("_"):
            if token:
                unescaped_token = _unescape_token(token + "_")
                if unescaped_token:
                    ret.append(unescaped_token)
        ret = self.join_tokens_to_string(ret)
        return self._output_wrapper(ret, return_str=return_str)

    @staticmethod
    def init_from_files(vocab_file,
                        files,
                        target_vocab_size,
                        threshold,
                        min_count=None,
                        language="en",
                        file_byte_limit=1e6,
                        reserved_tokens=None):
        """Create subtoken vocabulary based on files, and save vocab to file.

        Args:
          vocab_file: String name of vocab file to store subtoken vocabulary.
          files: List of file paths that will be used to generate vocabulary.
          target_vocab_size: target vocabulary size to generate.
          threshold: int threshold of vocabulary size to accept.
          language: The language.
          min_count: int minimum count to use for generating the vocabulary. The min
            count is the minimum number of times a subtoken should appear in the
            files before it is added to the vocabulary. If set to none, this value
            is found using binary search.
          file_byte_limit: (Default 1e6) Maximum number of bytes of sample text that
            will be drawn from the files.
          reserved_tokens: List of string tokens that are guaranteed to be at the
            beginning of the subtoken vocabulary list.

        Returns:
          Subtokenizer object
        """
        if reserved_tokens is None:
            reserved_tokens = RESERVED_TOKENS

        if tf.io.gfile.exists(vocab_file):
            logging.info("Vocab file already exists (%s)", vocab_file)
        else:
            logging.info("Begin steps to create subtoken vocabulary...")
            token_counts = _count_tokens(files, file_byte_limit)
            alphabet = _generate_alphabet_dict(token_counts)
            subtoken_list = _generate_subtokens_with_target_vocab_size(
                token_counts, alphabet, target_vocab_size, threshold, min_count,
                reserved_tokens)
            logging.info("Generated vocabulary with %d subtokens.",
                         len(subtoken_list))
            _save_vocab_file(vocab_file, subtoken_list)
        subtokenizer = Subtokenizer(language)
        subtokenizer.init_subtokenizer(vocab_file)
        return subtokenizer
