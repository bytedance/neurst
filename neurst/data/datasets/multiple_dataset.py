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
from neurst.data.datasets import build_dataset, register_dataset
from neurst.data.datasets.dataset import Dataset
from neurst.utils.flags_core import Flag


@register_dataset("multi_dataset")
class MultipleDataset(Dataset):

    def __init__(self, args):
        """ Initializes the multiple dataset.

        Args:
            args: containing `multiple_dataset`, which is like
                {
                    "data0": { "dataset.class": "", "dataset.params": ""},
                    "data1": { "dataset.class": "", "dataset.params": ""},
                    ......
                ]
        """
        super(MultipleDataset, self).__init__()
        self._datasets = {name: build_dataset(dsargs)
                          for name, dsargs in args["multiple_datasets"].items()}
        self._sample_weights = dict()
        if args["sample_weights"]:
            assert isinstance(args["sample_weights"], dict)
        else:
            args["sample_weights"] = {}
        sum = 0.
        for name in self._datasets:
            self._sample_weights[name] = args["sample_weights"].get(name, 1.)
            sum += self._sample_weights[name]
        for name in self._datasets:
            self._sample_weights[name] /= sum

    @staticmethod
    def class_or_method_args():
        return [
            Flag("multiple_datasets", dtype=Flag.TYPE.STRING,
                 help="A dict of dataset class and parameters, "
                      "where the key is the dataset name and "
                      "the value is a dict of arguments for one dataset."),
            Flag("sample_weights", dtype=Flag.TYPE.FLOAT,
                 help="A dict of weights for averaging metrics, where the key "
                      "is the dataset name. 1.0 for each by default.")
        ]

    @property
    def status(self):
        raise NotImplementedError

    @property
    def sample_weights(self):
        return self._sample_weights

    @property
    def datasets(self):
        return self._datasets

    def build(self, *args, **kwargs):
        raise NotImplementedError("Call each dataset's build function instead.")

    def build_iterator(self, *args, **kwargs):
        raise NotImplementedError("Call each dataset's build function instead.")
