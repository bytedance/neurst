#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import tempfile

import tensorflow as tf

from neurst.data.text.bpe import BPE


def test():
    codes = ["技 术</w>", "发 展</w>"]
    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    with tf.io.gfile.GFile(tmp_file.name, "w") as fw:
        fw.write("version\n")
        fw.write("\n".join(codes) + "\n")
    bpe = BPE(lang="zh",
              glossaries=["迅速", "<-neplhd-hehe>"])
    bpe.init_subtokenizer(tmp_file.name)

    tokens = bpe.tokenize("技术 发展 迅猛", return_str=True)
    assert tokens == "技术 发@@ 展 迅@@ 猛"
    assert bpe.detokenize(tokens) == "技术 发展 迅猛"
    tokens = bpe.tokenize("技术发展迅猛", return_str=True)
    assert tokens == "技@@ 术@@ 发@@ 展@@ 迅@@ 猛"
    assert bpe.detokenize(tokens) == "技术发展迅猛"
    tokens = bpe.tokenize("技术迅速发展迅速 迅速 <-neplhd-hehe>", return_str=True)
    assert tokens == "技术@@ 迅速@@ 发@@ 展@@ 迅速 迅速 <-neplhd-hehe>"
    assert bpe.detokenize(tokens) == "技术迅速发展迅速 迅速 <-neplhd-hehe>"

    os.remove(tmp_file.name)


if __name__ == "__main__":
    test()
