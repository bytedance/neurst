import os
import tempfile

from neurst.data.text.vocab import Vocab

word_tokens = ["Hello", "World", "yes", "i", "I"]


def test():
    vocab = Vocab(word_tokens,
                  extra_tokens=["UNK", "EOS"])
    assert vocab._token_list == ["Hello", "World", "yes", "i", "I", "UNK", "EOS"]
    assert vocab.vocab_size == 7
    assert vocab.map_token_to_id(["Hello", "world", "man"],
                                 unknown_default=100) == [0, 100, 100]
    assert vocab.map_id_to_token([1, 0, 3]) == ["World", "Hello", "i"]

    vocab = Vocab(word_tokens,
                  extra_tokens=["UNK", "EOS"], lowercase=True)
    assert vocab._token_list == ["hello", "world", "yes", "i", "UNK", "EOS"]
    assert vocab.vocab_size == 6
    assert vocab.map_token_to_id(["Hello", "world", "man"],
                                 unknown_default=100) == [0, 1, 100]
    assert vocab.map_id_to_token([1, 0, 3]) == ["world", "hello", "i"]


def test_file():
    vocab_file = tempfile.NamedTemporaryFile(delete=False)
    with open(vocab_file.name, "w") as fw:
        for t in word_tokens:
            fw.write(t + "\t100\n")
    vocab = Vocab.load_from_file(vocab_file.name,
                                 extra_tokens=["UNK", "EOS"])
    assert vocab._token_list == ["Hello", "World", "yes", "i", "I", "UNK", "EOS"]
    assert vocab.vocab_size == 7
    assert vocab.map_token_to_id(["Hello", "world", "man"],
                                 unknown_default=100) == [0, 100, 100]
    assert vocab.map_id_to_token([1, 0, 3]) == ["World", "Hello", "i"]

    vocab = Vocab.load_from_file(vocab_file.name,
                                 extra_tokens=["UNK", "EOS"], lowercase=True)
    assert vocab._token_list == ["hello", "world", "yes", "i", "UNK", "EOS"]
    assert vocab.vocab_size == 6
    assert vocab.map_token_to_id(["Hello", "world", "man", "EOS"],
                                 unknown_default=100) == [0, 1, 100, 5]
    assert vocab.map_id_to_token([1, 0, 3]) == ["world", "hello", "i"]
    os.remove(vocab_file.name)


if __name__ == "__main__":
    test()
    test_file()
