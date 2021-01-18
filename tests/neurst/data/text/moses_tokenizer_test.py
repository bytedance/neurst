#!/usr/bin/env python
# -*- coding: utf-8 -*-
from neurst.data.text.moses_tokenizer import MosesTokenizer


def test():
    # lang, origin, tokenized, detokenized
    samples = [
        ("zh", False, [], "`啊你     好～！   ", "`啊你 好 ～ ！", "`啊你好 ～ ！"),
        ("en", False, [], "Hello p.m. 10, [.", "Hello p.m. 10 , [ .", None),
        ("en", True, [], "Hello p.m. 10, [.", "Hello p.m. 10 , [ .", None),
        ("en", False, ['<wotama>'], 'Hello <wotama> p.m. 10, [.',
         'Hello <wotama> p.m. 10 , [ .', 'Hello <wotama> p.m. 10, [.')

    ]

    for lang, escape, gloss, ori, tok, detok in samples:
        if not detok:
            detok = ori
        tokenizer = MosesTokenizer(language=lang, glossaries=gloss)
        assert tok == tokenizer.tokenize(ori, return_str=True)
        assert detok == tokenizer.detokenize(tok, return_str=True)


if __name__ == '__main__':
    test()
