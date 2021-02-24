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
from abc import ABCMeta, abstractmethod

import six


@six.add_metaclass(ABCMeta)
class FeatureExtractor(object):
    """ Abstract feature extractor for extracting audio features. """
    REGISTRY_NAME = "feature_extractor"

    @property
    @abstractmethod
    def feature_dim(self):
        """ Returns the dimension of the feature. """
        raise NotImplementedError

    @abstractmethod
    def seconds(self, feature):
        """ Returns the time seconds of this sample. """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, signal, rate):
        raise NotImplementedError

    @staticmethod
    def class_or_method_args():
        return []
