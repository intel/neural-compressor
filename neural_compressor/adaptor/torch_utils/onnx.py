#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Onnx Quantization Torch Utils."""
from neural_compressor.utils.utility import LazyImport
from collections import UserDict

ortq = LazyImport('onnxruntime.quantization')


class DataReader(ortq.CalibrationDataReader):
    """DataReader Class for onnx static quantization."""
    def __init__(self, dataloader, sample_size=100):
        """Init a DataReader object."""
        import math
        self.dataloader = dataloader
        self.batch_size = dataloader.batch_size
        self.batch_num = math.ceil(sample_size/self.batch_size)
        self.datasize = self.batch_num * self.batch_size

        self.data = []
        try:
            for i, (input, label) in enumerate(self.dataloader):
                if i * self.batch_size >= self.datasize:
                    break
                if isinstance(input, dict) or isinstance(input, UserDict):
                    batch = {k: v.detach().cpu().numpy() for k, v in input.items()}
                elif isinstance(input, list) or isinstance(input, tuple):
                    batch = {'input': [v.detach().cpu().numpy() for v in input]}
                else:
                    batch = {'input': input.detach().cpu().numpy()}
                self.data.append(batch)
            self.data = iter(self.data)
        except:
            for i, input in enumerate(self.dataloader):
                if i * self.batch_size >= self.datasize:
                    break
                if isinstance(input, dict) or isinstance(input, UserDict):
                    batch = {k: v.detach().cpu().numpy() for k, v in input.items()}
                elif isinstance(input, list) or isinstance(input, tuple):
                    batch = {'input': [v.detach().cpu().numpy() for v in input]}
                else:
                    batch = {'input': input.detach().cpu().numpy()}
                self.data.append(batch)
            self.data = iter(self.data)

    def get_next(self):
        """Get the next data."""
        return next(self.data, None)
