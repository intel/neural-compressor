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

# For mapping revert patched module to origin module

helper_mods = {}


def helper_mod_register(name):
    def decorator(mod):
        helper_mods[name] = mod
        return mod

    return decorator


@helper_mod_register(name="Matmul")
class Matmul(torch.nn.Module):
    def __init__(self, patched_mod, *args, **kwargs):
        super().__init__()
        self.__dict__.update(patched_mod.__dict__)
        self.extra_repr = patched_mod.extra_repr_org


@helper_mod_register(name="Linear")
class Linear(torch.nn.Module):
    def __init__(self, patched_mod, *args, **kwargs):
        super().__init__()
        self.__dict__.update(patched_mod.__dict__)
        self.extra_repr = patched_mod.extra_repr_org


@helper_mod_register(name="FalconLinear")
class FalconLinear(torch.nn.Module):
    def __init__(self, patched_mod, *args, **kwargs):
        super().__init__()
        self.__dict__.update(patched_mod.__dict__)
        self.extra_repr = patched_mod.extra_repr_org


@helper_mod_register(name="KVCache")
class KVCache(torch.nn.Module):
    def __init__(self, patched_mod, *args, **kwargs):
        super().__init__()
        self.allocate = patched_mod.org_allocate
        self.get_shape = patched_mod.get_shape
        self.forward = patched_mod.forward
        self.update = patched_mod.update


@helper_mod_register(name="Conv2d")
class Conv2d(torch.nn.Module):
    def __init__(self, patched_mod, *args, **kwargs):
        super().__init__()
        self.__dict__.update(patched_mod.__dict__)
        self.extra_repr = patched_mod.extra_repr_org


@helper_mod_register(name="LoRACompatibleLinear")
class LoRACompatibleLinear(torch.nn.Module):
    def __init__(self, patched_mod, *args, **kwargs):
        super().__init__()
        self.__dict__.update(patched_mod.__dict__)
        self.extra_repr = patched_mod.extra_repr_org


@helper_mod_register(name="LoRACompatibleConv")
class LoRACompatibleConv(torch.nn.Module):
    def __init__(self, patched_mod, *args, **kwargs):
        super().__init__()
        self.__dict__.update(patched_mod.__dict__)
        self.extra_repr = patched_mod.extra_repr_org


@helper_mod_register(name="Softmax")
class Softmax(torch.nn.Module):
    def __init__(self, patched_mod, *args, **kwargs):
        super().__init__()
        self.__dict__.update(patched_mod.__dict__)
        self.extra_repr = patched_mod.extra_repr_org


@helper_mod_register(name="LinearLayer")
class LinearLayer(torch.nn.Module):
    def __init__(self, patched_mod, *args, **kwargs):
        super().__init__()
        self.__dict__.update(patched_mod.__dict__)
        self.extra_repr = patched_mod.extra_repr_org


@helper_mod_register(name="LinearAllreduce")
class LinearAllreduce(torch.nn.Module):
    def __init__(self, patched_mod, *args, **kwargs):
        super().__init__()
        self.__dict__.update(patched_mod.__dict__)
        self.extra_repr = patched_mod.extra_repr_org


@helper_mod_register(name="ScopedLinearAllReduce")
class ScopedLinearAllReduce(torch.nn.Module):
    def __init__(self, patched_mod, *args, **kwargs):
        super().__init__()
        self.__dict__.update(patched_mod.__dict__)
        self.extra_repr = patched_mod.extra_repr_org


@helper_mod_register(name="LmHeadLinearAllreduce")
class LmHeadLinearAllreduce(torch.nn.Module):
    def __init__(self, patched_mod, *args, **kwargs):
        super().__init__()
        self.__dict__.update(patched_mod.__dict__)
        self.extra_repr = patched_mod.extra_repr_org


@helper_mod_register(name="ModuleFusedSDPA")
class ModuleFusedSDPA(torch.nn.Module):
    def __init__(self, patched_mod, *args, **kwargs):
        super().__init__()
        self.__dict__.update(patched_mod.__dict__)
        self.extra_repr = patched_mod.extra_repr_org
