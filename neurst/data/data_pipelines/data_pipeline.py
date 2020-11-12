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

from neurst.utils.configurable import extract_constructor_params


@six.add_metaclass(ABCMeta)
class DataPipeline(object):
    REGISTRY_NAME = "data_pipeline"

    def __init__(self, **kwargs):
        self._params = extract_constructor_params(locals(), verbose=False)

    def get_config(self) -> dict:
        return self._params

    @property
    @abstractmethod
    def meta(self) -> dict:
        """ The meta data. """
        return {}

    @abstractmethod
    def recover(self, input):
        """ Recovers one data sample. """
        raise NotImplementedError

    @abstractmethod
    def process(self, input, is_processed=False):
        """ Processes one data sample. """
        raise NotImplementedError
