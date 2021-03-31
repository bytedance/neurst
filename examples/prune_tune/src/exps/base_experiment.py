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
class BaseExperiment(object):
    REGISTRY_NAME = "entry"

    def __init__(self, strategy, model, task, custom_dataset, model_dir):
        """ Initializes the basic experiment for training, evaluation, etc. """
        self._strategy = strategy
        self._model = model
        self._model_dir = model_dir
        self._task = task
        self._custom_dataset = custom_dataset

    @property
    def strategy(self):
        return self._strategy

    @property
    def model(self):
        return self._model

    @property
    def task(self):
        return self._task

    @property
    def custom_dataset(self):
        return self._custom_dataset

    @property
    def model_dir(self):
        return self._model_dir

    @abstractmethod
    def run(self):
        """ Running the method. """
        raise NotImplementedError
