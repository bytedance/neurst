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
class Metric(object):
    REGISTRY_NAME = "metric"

    def __init__(self, *args, **kwargs):
        self._flag = self.__class__.__name__

    @property
    def flag(self):
        return self._flag

    @flag.setter
    def flag(self, flag_name):
        """ Sets the flag metric name if the result of `__call__` is a dict. """
        self._flag = flag_name

    def set_groundtruth(self, groundtruth):
        raise NotImplementedError

    def greater_or_eq(self, result1, result2):
        """ Compare the two metric value result and return True if v1>=v2. """
        return self.get_value(result1) >= self.get_value(result2)

    def get_value(self, result):
        """ Gets a float value from the metric result (if is a dict). """
        if isinstance(result, dict) and self._flag in result:
            return result[self._flag]
        return result

    def __call__(self, hypothesis, groundtruth=None) -> dict:
        """ Returns a dict of metric values. """
        res = self.call(hypothesis, groundtruth=groundtruth)
        if not isinstance(res, dict):
            res = {self.flag: res}
        return res

    @abstractmethod
    def call(self, hypothesis, groundtruth=None):
        """ Returns the metric value (float) or a dict of metric values. """
        raise NotImplementedError


class MetricWrapper(Metric):
    """ A wrapper class for easy-use of metric. """

    def __init__(self, flag, greater_is_better=True):
        super(MetricWrapper, self).__init__()
        self._flag = flag
        self._greater_is_better = greater_is_better

    def call(self, hypothesis, groundtruth=None):
        raise NotImplementedError("No need to call `__call__` in MetricWrapper.")

    def greater_or_eq(self, result1, result2):
        if self._greater_is_better:
            return super(MetricWrapper, self).greater_or_eq(result1, result2)
        return not super(MetricWrapper, self).greater_or_eq(result1, result2)
