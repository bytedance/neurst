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
import yaml

from neurst.tasks import register_task
from neurst.tasks.translation import Translation
from neurst.utils.flags_core import Flag


@register_task
class WaitkTranslation(Translation):
    """ Defines the translation task. """

    def __init__(self, args):
        super(WaitkTranslation, self).__init__(args)
        self._wait_k = args["wait_k"]
        if isinstance(self._wait_k, str):
            self._wait_k = yaml.load(self._wait_k, Loader=yaml.FullLoader)
        assert self._wait_k, "Must provide wait_k as the decode lagging."
        assert isinstance(self._wait_k, list) or isinstance(self._wait_k, int), (
            f"Value error: {self._wait_k}")

    def get_config(self):
        cfg = super(WaitkTranslation, self).get_config()
        cfg["wait_k"] = self._wait_k
        return cfg

    @staticmethod
    def class_or_method_args():
        this_args = super(WaitkTranslation, WaitkTranslation).class_or_method_args()
        this_args.extend([
            Flag("wait_k", dtype=Flag.TYPE.STRING, default=None,
                 help="The lagging k.")
        ])
        return this_args

    def build_model(self, args, name=None, **kwargs):
        return super(WaitkTranslation, self).build_model(args, name=name,
                                                         waitk_lagging=self._wait_k, **kwargs)
