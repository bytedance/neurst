import tempfile

import tensorflow as tf

from neurst.data.data_pipelines.text_data_pipeline import TextDataPipeline


def test():
    vocab_file = tempfile.NamedTemporaryFile(delete=False)
    with tf.io.gfile.GFile(vocab_file.name, "w") as fw:
        for t in ["技术", "迅@@", "猛"]:
            fw.write(t + "\t100\n")
    bpe_codes_file = tempfile.NamedTemporaryFile(delete=False)
    with tf.io.gfile.GFile(bpe_codes_file.name, "w") as fw:
        fw.write("version\n")
        fw.write("\n".join(["技 术</w>", "发 展</w>"]) + "\n")
    text_dp = TextDataPipeline(
        vocab_path=vocab_file.name,
        language="zh",
        subtokenizer="bpe",
        subtokenizer_codes=bpe_codes_file.name)

    assert text_dp.process("技术 发展 迅猛") == [0, 3, 3, 1, 2, 5]
    assert text_dp.recover([0, 3, 1, 2, 5]) == "技术 <UNK> 迅猛"


if __name__ == "__main__":
    test()
