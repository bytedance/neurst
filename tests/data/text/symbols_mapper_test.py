import os
import tempfile

import pytest

from neurst.data.text.symbols_mapper import SymbolsMapper

word_tokens = ["Hello", "World", "yes", "i", "I"]


def test():
    with pytest.raises(ValueError):
        _ = SymbolsMapper()

    mapper = SymbolsMapper(tokens=word_tokens)
    assert (mapper.map_token_to_id(["Hello", "world", "man"],
                                   return_str=False, with_bos=True, with_eos=True)
            == [6, 0, 5, 5, 7])
    assert (mapper.map_token_to_id("Hello world man",
                                   return_str=True, with_bos=True, with_eos=True)
            == "6 0 5 5 7")
    assert (mapper.map_id_to_token([6, 0, 5, 5, 7], return_str=False)
            == ["Hello", "<UNK>", "<UNK>"])
    assert (mapper.map_id_to_token("6 0 5 5 7 1 2 3 4", return_str=True)
            == "Hello <UNK> <UNK>")

    mapper = SymbolsMapper(tokens=word_tokens, lowercase=True)
    assert (mapper.map_token_to_id(["Hello", "world", "man"],
                                   return_str=False, with_bos=False, with_eos=True)
            == [0, 1, 4, 6])
    assert (mapper.map_token_to_id("Hello world man",
                                   return_str=True, with_bos=False, with_eos=True)
            == "0 1 4 6")
    assert (mapper.map_id_to_token([0, 1, 4, 6], return_str=False)
            == ["hello", "world", "<UNK>"])
    assert (mapper.map_id_to_token("0 1 4 6 1 2", return_str=True)
            == "hello world <UNK>")


def test_file():
    vocab_file = tempfile.NamedTemporaryFile(delete=False)
    with open(vocab_file.name, "w") as fw:
        for t in word_tokens:
            fw.write(t + "\t100\n")
    mapper = SymbolsMapper(vocab_path=vocab_file.name)
    assert (mapper.map_token_to_id(["Hello", "world", "man"],
                                   return_str=False, with_bos=True, with_eos=True)
            == [6, 0, 5, 5, 7])
    assert (mapper.map_token_to_id("Hello world man",
                                   return_str=True, with_bos=True, with_eos=True)
            == "6 0 5 5 7")
    assert (mapper.map_id_to_token([6, 0, 5, 5, 7], return_str=False)
            == ["Hello", "<UNK>", "<UNK>"])
    assert (mapper.map_id_to_token("6 0 5 5 7 1 2 3 4", return_str=True)
            == "Hello <UNK> <UNK>")
    os.remove(vocab_file.name)


if __name__ == "__main__":
    test()
    test_file()
