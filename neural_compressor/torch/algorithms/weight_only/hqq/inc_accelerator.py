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

# The design copied from
# https://github.com/microsoft/DeepSpeed/blob/master/accelerator/abstract_accelerator.py.

# To keep it simply, only add the APIs we need.

import os
from abc import ABC, abstractmethod

import torch


class INC_Accelerator(ABC):
    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def device_name(self, device_indx) -> str:
        pass

    @abstractmethod
    def set_device(self, device_index):
        pass

    @abstractmethod
    def current_device(self):
        pass

    @abstractmethod
    def current_device_name(self):
        pass

    @abstractmethod
    def device(self, device_index=None):
        pass

    @abstractmethod
    def empty_cache(self):
        pass


class CPU_Accelerator(INC_Accelerator):
    def __init__(self) -> None:
        self._name = "cpu"

    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True

    def device_name(self, device_indx) -> str:
        return "cpu"

    def set_device(self, device_index):
        pass

    def current_device(self):
        return 0

    def current_device_name(self):
        return "cpu"

    def device(self, device_index=None):
        pass

    def empty_cache(self):
        pass


class CUDA_Accelerator(INC_Accelerator):
    def __init__(self) -> None:
        self._name = "cuda"

    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def device_name(self, device_indx) -> str:
        if device_indx is None:
            return "cuda"
        return f"cuda:{device_indx}"

    def set_device(self, device_index):
        return torch.cuda.set_device(device_index)

    def current_device(self):
        return torch.cuda.current_device()

    def current_device_name(self):
        return "cuda:{}".format(torch.cuda.current_device())

    def device(self, device_index=None):
        torch.cuda.device(device_index)

    def empty_cache(self):
        torch.cuda.empty_cache()


cuda_accelerator = CUDA_Accelerator()
cpu_accelerator = CPU_Accelerator()

accelerator_mapping = {
    "cuda": cuda_accelerator,
    "cpu": cpu_accelerator,
}


def auto_detect_accelerator() -> INC_Accelerator:
    FORCE_DEVICE = os.environ.get("FORCE_DEVICE", None)
    if FORCE_DEVICE and FORCE_DEVICE in accelerator_mapping:
        print(f"!!! Force use {FORCE_DEVICE} accelerator.")
        return accelerator_mapping[FORCE_DEVICE]
    if torch.cuda.is_available():
        print("!!! Use cuda accelerator.")
        return cuda_accelerator
    return cpu_accelerator


# Force use cpu accelerator even if cuda is available.
# FORCE_DEVICE = "cpu" python ...
