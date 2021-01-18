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
import torch


def input_length_to_nonpadding(lengths, max_len, dtype=None):
    """ Creates a bias tensor according to the non-padding tensor for cross entropy.

    Args:
        length: A Tensor with shape [batch_size, ], indicating the true length.
        max_len: A scalar tensor indicating the maximum length.

    Returns:
        A float tensor with shape [batch_size, max_len],
        indicating the padding positions, where 0.0 for padding and
        1.0 for non-padding.
    """
    row_vector = torch.arange(0, max_len)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = (row_vector < matrix).to(dtype or torch.float)
    return mask  # 1.0 for non-padding


def input_length_to_padding(lengths, max_len, dtype=None):
    """ Creates a bias tensor according to the padding tensor for attention.

    Args:
        length: A Tensor with shape [batch_size, ], indicating the true length.
        max_len: A scalar tensor indicating the maximum length.

    Returns:
        A float tensor with shape [batch_size, max_len],
        indicating the padding positions, where 1.0 for padding and
        0.0 for non-padding.
    """
    return 1. - input_length_to_nonpadding(lengths, max_len, dtype)
