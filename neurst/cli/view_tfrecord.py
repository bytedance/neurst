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
import sys

import tensorflow as tf

from neurst.data.dataset_utils import glob_tfrecords


def cli_main():
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and (sys.argv[1] in ["help", "--help", "-h"])):
        print("Usage: ")
        print("    >> python3 -m neurst.cli.view_tfrecord path")
        print("           Show examples and types of TF Record elements.")
        exit()

    print("===================== Examine elements =====================")
    for x in tf.data.TFRecordDataset(glob_tfrecords(sys.argv[1])).take(1):
        example = tf.train.Example()
        example.ParseFromString(x.numpy())
        print(example)
        print("elements: {")
        for name in example.features.feature:
            if len(example.features.feature[name].bytes_list.value) > 0:
                print(f"    \"{name}\": bytes (str)")
            elif len(example.features.feature[name].int64_list.value) > 0:
                print(f"    \"{name}\": int64")
            elif len(example.features.feature[name].float_list.value) > 0:
                print(f"    \"{name}\": float32")
        print("}")


if __name__ == "__main__":
    cli_main()
