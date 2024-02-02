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

from neural_compressor.torch.algorithms.weight_only.hqq.qtensor import QTensor, QTensorMetaInfo


class TestQTensor:
    def test_q_tensor(self):
        in_feats = 3
        out_feats = 4

        val = torch.randn(out_feats, in_feats)
        scale = torch.randn(out_feats)
        zero = torch.randint(1, 10, (out_feats,))
        q_tensor_meta = QTensorMetaInfo(nbits=4, group_size=64, shape=(out_feats, in_feats), axis=0, packing=False)
        q_tensor = QTensor(val, scale, zero, q_tensor_meta)
        print(q_tensor)

    def test_qtensor_meta_info(self):
        in_feats = 3
        out_feats = 4
        meta_config = QTensorMetaInfo(nbits=4, group_size=64, shape=(out_feats, in_feats), axis=0, packing=False)
        print(meta_config)
        print(meta_config.to_dict)
        assert meta_config.to_dict() == {
            "nbits": 4,
            "group_size": 64,
            "shape": (out_feats, in_feats),
            "axis": 0,
            "packing": False,
        }
