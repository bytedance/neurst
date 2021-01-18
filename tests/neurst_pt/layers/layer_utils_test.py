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
from neurst.layers.layer_utils import lower_triangle_attention_bias
from neurst.utils.misc import assert_equal_numpy
from neurst_pt.layers.layer_utils import lower_triangle_attention_bias as pt_lower_triangle_attention_bias


def test_lower_triangle_attention_bias():
    assert_equal_numpy(lower_triangle_attention_bias(5).numpy(),
                       pt_lower_triangle_attention_bias(5).detach().numpy())


if __name__ == "__main__":
    test_lower_triangle_attention_bias()
