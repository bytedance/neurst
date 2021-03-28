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
import random
from abc import ABCMeta, abstractmethod

import numpy
import six
import yaml

from neurst.utils.flags_core import Flag


@six.add_metaclass(ABCMeta)
class DataSampler(object):
    REGISTRY_NAME = "data_sampler"

    def __init__(self, args):
        if isinstance(args["sample_sizes"], str):
            args["sample_sizes"] = yaml.load(args["sample_sizes"], Loader=yaml.FullLoader)
        assert isinstance(args["sample_sizes"], dict) and len(args["sample_sizes"]) > 0, (
            "Unknown `sample_sizes`={} with type {}".format(args["sample_sizes"], type(args["sample_sizes"])))
        self._sample_ratios = self.get_sample_ratios(args["sample_sizes"])
        total = sum(self._sample_ratios.values())
        self._normalized_sample_weights = {k: float(v) / total for k, v in self._sample_ratios.items()}
        self._sample_items = []
        self._sample_boundaries = []
        for k, v in self._sample_ratios.items():
            self._sample_items.append(k)
            if len(self._sample_boundaries) == 0:
                self._sample_boundaries.append(float(v) / total)
            else:
                self._sample_boundaries.append(self._sample_boundaries[-1] + float(v) / total)
        self._sample_boundaries = numpy.array(self._sample_boundaries)

    @staticmethod
    def class_or_method_args():
        return [
            Flag("sample_sizes", dtype=Flag.TYPE.STRING,
                 help="A dict. The key is the item name to be sampled, "
                      "while the value is the corresponding proportion.")
        ]

    @property
    def normalized_sample_weights(self):
        return self._normalized_sample_weights

    @abstractmethod
    def get_sample_ratios(self, sample_sizes) -> dict:
        raise NotImplementedError

    def __call__(self):
        ratio = random.random()
        for idx in range(len(self._sample_boundaries) - 1, -1, -1):
            if ratio > self._sample_boundaries[idx]:
                return self._sample_items[idx + 1]
        return self._sample_items[0]
