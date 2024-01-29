# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from hqq_utils import QTensor, QTensorMetaInfo

in_feats = 3
out_feats = 4

val = torch.randn(out_feats, in_feats)
scale = torch.randn(out_feats)
zero = torch.randint(1, 10, (out_feats,))
q_tensor_meta = QTensorMetaInfo(nbits=4, group_size=64, shape=(out_feats, in_feats), axis=0, packing=False)
q_tensor = QTensor(val, scale, zero, q_tensor_meta)
print(q_tensor)
# q_tensor.to(torch.device("cuda:0"))
# print(q_tensor)


val = torch.randn(out_feats, in_feats)
scale = q_tensor
zero = q_tensor
q_tensor_meta = QTensorMetaInfo(nbits=4, group_size=64, shape=(out_feats, in_feats), axis=0, packing=False)
q_tensor_2 = QTensor(val, scale, zero, q_tensor_meta)
print(q_tensor_2)
q_tensor_2.to(torch.device("cuda:0"))
print(q_tensor_2)


def check_cuda():
    if torch.cuda.is_available():
        print("[check_cuda] cuda is available")
    else:
        print("[check_cuda] cuda is not available")


check_cuda()

# in_feats = 3
# out_feats = 4

# lin = torch.nn.Linear(in_feats, out_feats)
# lin.to(torch.device("cuda:0"))
# print(lin.weight.device)
