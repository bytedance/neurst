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

from neurst.data.datasets.dataset import Dataset


@six.add_metaclass(ABCMeta)
class TextGenDataset(Dataset):
    """ The abstract dataset for text generation, which must implement `get_targets` function. """

    def __init__(self, trg_lang=None):
        self._targets = None
        self._raw_targets = None
        self._trg_lang = trg_lang
        super(TextGenDataset, self).__init__()

    @property
    def trg_lang(self):
        return self._trg_lang

    @property
    @abstractmethod
    def status(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def build_iterator(self, map_func=None, shard_id=0, total_shards=1):
        """ Returns the iterator of the dataset.

        Args:
            map_func: A function mapping a dataset element to another dataset element.
            shard_id: Generator yields on the `shard_id`-th shard of the whole dataset.
            total_shards: The number of total shards.
        """
        raise NotImplementedError

    @property
    def targets(self):
        """ Returns a list of target texts. """
        return self._targets
