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
from neurst.utils.flags_core import Flag

try:
    from python_speech_features import logfbank
except ImportError:
    pass


@register_feature_extractor("fbank")
class LogMelFbank(FeatureExtractor):

    def __init__(self, args):
        self._nfilt = args["nfilt"]
        self._winlen = args["winlen"]
        self._winstep = args["winstep"]
        try:
            from python_speech_features import logfbank
            _ = logfbank
        except ImportError:
            raise ImportError('Please install python_speech_features with: pip3 install python_speech_features')

    @staticmethod
    def class_or_method_args():
        return [
            Flag("nfilt", dtype=Flag.TYPE.INTEGER, default=80,
                 help="The number of frames in the filterbank."),
            Flag("winlen", dtype=Flag.TYPE.FLOAT, default=0.025,
                 help="The length of the analysis window in seconds. Default is 0.025s."),
            Flag("winstep", dtype=Flag.TYPE.FLOAT, default=0.01,
                 help="The step between successive windows in seconds. Default is 0.01s.")
        ]

    @property
    def feature_dim(self):
        return self._nfilt

    def seconds(self, feature):
        return (numpy.shape(feature)[0] - 1.) * self._winstep + self._winlen

    def __call__(self, signal, rate):
        inp = logfbank(signal, samplerate=rate, nfilt=self._nfilt,
                       winlen=self._winlen, winstep=self._winstep).astype(numpy.float32)
        inp = (inp - numpy.mean(inp)) / numpy.std(inp)
        return inp
