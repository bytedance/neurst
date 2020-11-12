#!/usr/bin/env python
# -*- coding: utf-8 -*-

from neurst.data.text.jieba_segment import Jieba


def test():
    tok = Jieba()
    assert tok.tokenize("他来到了网易杭研大厦", return_str=True) == "他 来到 了 网易 杭研 大厦"
    assert tok.detokenize("他 来到 了 网易 杭研 大厦", return_str=True) == "他来到了网易杭研大厦"


if __name__ == "__main__":
    test()
