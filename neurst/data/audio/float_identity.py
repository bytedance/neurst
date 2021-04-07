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
import numpy

from neurst.data.audio import FeatureExtractor, register_feature_extractor


@register_feature_extractor
class FloatIdentity(FeatureExtractor):

    def __init__(self, args):
        _ = args

    def seconds(self, feature):
        # by default: sample rate=16000
        return len(feature) / 16000.

    @property
    def feature_dim(self):
        return 1

    def __call__(self, signal, rate):
        if isinstance(signal[0], (float, numpy.float32, numpy.float64)):
            return numpy.array(signal)
        return numpy.array(signal) / 32768.
