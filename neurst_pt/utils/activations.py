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
import torch.nn.functional as F


def get_activation(activ):
    if callable(activ):
        return activ
    if activ is None:
        return lambda x: x
    if activ == "tanh":
        return F.tanh
    elif activ == "relu":
        return F.relu
    elif activ == "gelu":
        return F.gelu
    elif activ == "glu":
        return lambda x: F.glu(x, -1)
    else:
        raise ValueError("Unknown activation: {}".format(activ))
