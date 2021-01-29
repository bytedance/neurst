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
import argparse
import os
import re
import tarfile

import tensorflow as tf
from absl import logging

from neurst.data.text.subtokenizer import Subtokenizer
from neurst.utils.misc import download_with_tqdm

_TRAIN_DATA_RESOURCES = [
    {
        "URL": "http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz",
        "TAR": "training-parallel-nc-v12.tgz",
        "SRC": "training/news-commentary-v12.de-en.en",
        "TRG": "training/news-commentary-v12.de-en.de"
    },
    {
        "URL": "http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz",
        "TAR": "training-parallel-commoncrawl.tgz",
        "SRC": "commoncrawl.de-en.en",
        "TRG": "commoncrawl.de-en.de"
    },
    {
        "URL": "http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz",
        "TAR": "training-parallel-europarl-v7.tgz",
        "SRC": "training/europarl-v7.de-en.en",
        "TRG": "training/europarl-v7.de-en.de"
    },
]

_DEV_DATA_RESOURCE = {
    "URL": "http://data.statmt.org/wmt17/translation-task/dev.tgz",
    "TAR": "dev.tgz",
    "SRC": "dev/newstest2013.en",
    "TRG": "dev/newstest2013.de"
}

_TEST_DATA_RESOURCE = {
    "URL": "http://statmt.org/wmt14/test-full.tgz",
    "TAR": "test-full.tgz",
    "SRC": "test-full/newstest2014-deen-src.en.sgm",
    "TRG": "test-full/newstest2014-deen-ref.de.sgm"
}


def _wrapper_xml(text):
    text = text.replace("</seg>", "")
    text = re.sub('<seg id="[0-9]*">', "", text)
    return text


def download_to(output_dir):
    # download training data
    training_srcs = []
    training_trgs = []
    for traindata in _TRAIN_DATA_RESOURCES:
        src = os.path.join(output_dir, traindata["SRC"])
        trg = os.path.join(output_dir, traindata["TRG"])
        training_srcs.append(src)
        training_trgs.append(trg)
        if os.path.exists(src) and os.path.exists(trg):
            continue
        tar_filename = os.path.join(output_dir, traindata["TAR"])
        if not os.path.exists(tar_filename):
            download_with_tqdm(traindata["URL"], tar_filename)
        with tarfile.open(tar_filename, "r:*") as tar:
            for tarinfo in tar:
                if tarinfo.name.endswith(traindata["SRC"]) or tarinfo.name.endswith(traindata["TRG"]):
                    tar.extract(tarinfo, output_dir)

    with tf.io.gfile.GFile(os.path.join(output_dir, "train.en.txt"), "w") as fw_src:
        with tf.io.gfile.GFile(os.path.join(output_dir, "train.de.txt"), "w") as fw_trg:
            for src, trg in zip(training_srcs, training_trgs):
                with tf.io.gfile.GFile(src, "r") as f_src, tf.io.gfile.GFile(trg, "r") as f_trg:
                    for s, t in zip(f_src, f_trg):
                        fw_src.write(" ".join(s.strip().split()) + "\n")
                        fw_trg.write(" ".join(t.strip().split()) + "\n")
    # download dev data
    dev_tar_filename = os.path.join(output_dir, _DEV_DATA_RESOURCE["TAR"])
    dev_src = os.path.join(output_dir, _DEV_DATA_RESOURCE["SRC"])
    dev_trg = os.path.join(output_dir, _DEV_DATA_RESOURCE["TRG"])
    if not (os.path.exists(dev_src) and os.path.exists(dev_trg)):
        if not os.path.exists(dev_tar_filename):
            download_with_tqdm(_DEV_DATA_RESOURCE["URL"], dev_tar_filename)
        with tarfile.open(dev_tar_filename, "r:*") as tar:
            for tarinfo in tar:
                if (tarinfo.name.endswith(_DEV_DATA_RESOURCE["SRC"])
                    or tarinfo.name.endswith(_DEV_DATA_RESOURCE["TRG"])):
                    tar.extract(tarinfo, output_dir)
    with open(os.path.join(output_dir, "newstest2013.en.txt"), "w") as fw_src:
        with open(os.path.join(output_dir, "newstest2013.de.txt"), "w") as fw_trg:
            with tf.io.gfile.GFile(dev_src) as f_src, tf.io.gfile.GFile(dev_trg) as f_trg:
                for s, t in zip(f_src, f_trg):
                    fw_src.write(s.strip() + "\n")
                    fw_trg.write(t.strip() + "\n")

    # download test data
    test_tar_filename = os.path.join(output_dir, _TEST_DATA_RESOURCE["TAR"])
    test_src = os.path.join(output_dir, _TEST_DATA_RESOURCE["SRC"])
    test_trg = os.path.join(output_dir, _TEST_DATA_RESOURCE["TRG"])
    if not (os.path.exists(test_src) and os.path.exists(test_trg)):
        if not os.path.exists(test_tar_filename):
            download_with_tqdm(_TEST_DATA_RESOURCE["URL"], test_tar_filename)
        with tarfile.open(test_tar_filename, "r:*") as tar:
            for tarinfo in tar:
                if (tarinfo.name.endswith(_TEST_DATA_RESOURCE["SRC"])
                    or tarinfo.name.endswith(_TEST_DATA_RESOURCE["TRG"])):
                    tar.extract(tarinfo, output_dir)
    with open(os.path.join(output_dir, "newstest2014.en.txt"), "w") as fw_src:
        with open(os.path.join(output_dir, "newstest2014.de.txt"), "w") as fw_trg:
            with tf.io.gfile.GFile(test_src) as f_src, tf.io.gfile.GFile(test_trg) as f_trg:
                for s, t in zip(f_src, f_trg):
                    if s.startswith("<seg id"):
                        fw_src.write(_wrapper_xml(s.strip()) + "\n")
                        fw_trg.write(_wrapper_xml(t.strip()) + "\n")
    return training_srcs, training_trgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, default=None, required=True)
    parser.add_argument("--learn_wordpiece", action="store_true", default=False)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    training_srcs, training_trgs = download_to(args.output_dir)
    if args.learn_wordpiece:
        logging.info("Learn word piece vocab on {}".format(training_srcs + training_trgs))
        _ = Subtokenizer.init_from_files(os.path.join(args.output_dir, "vocab"),
                                         training_srcs + training_trgs,
                                         target_vocab_size=32768,
                                         threshold=327,
                                         min_count=6)
