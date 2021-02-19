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
from neurst.data.datasets.data_sampler import DataSampler, register_data_sampler
from neurst.utils.flags_core import Flag


@register_data_sampler("temperature")
class TemperatureSampler(DataSampler):

    def __init__(self, args):
        self._temperature = args["temperature"]
        super(TemperatureSampler, self).__init__(args)

    @staticmethod
    def class_or_method_args():
        this_flags = super(TemperatureSampler, TemperatureSampler).class_or_method_args()
        this_flags.append(
            Flag("temperature", dtype=Flag.TYPE.FLOAT, default=5,
                 help="The temperature for sampling."))
        return this_flags

    def get_sample_ratios(self, sample_sizes) -> dict:
        total_size = sum(sample_sizes.values())
        return {k: (v / total_size) ** (1. / self._temperature)
                for k, v in sample_sizes.items()}
