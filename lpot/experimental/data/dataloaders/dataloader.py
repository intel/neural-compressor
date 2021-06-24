#!/usr/bin/env python
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

from .tensorflow_dataloader import TensorflowDataLoader
from .mxnet_dataloader import MXNetDataLoader
from .pytorch_dataloader import PyTorchDataLoader
from .onnxrt_dataloader import ONNXRTDataLoader

DATALOADERS = {"tensorflow": TensorflowDataLoader,
               "mxnet": MXNetDataLoader,
               "pytorch": PyTorchDataLoader,
               "pytorch_ipex": PyTorchDataLoader,
               "pytorch_fx": PyTorchDataLoader,
               "onnxrt_qlinearops": ONNXRTDataLoader,
               "onnxrt_integerops": ONNXRTDataLoader}

