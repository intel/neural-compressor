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

"""Base class and helper functions for registering observers."""

from typing import Dict, Optional, Any
from abc import abstractmethod, ABC
import os
import torch
from neural_compressor.torch.algorithms.fp8_quant.model_configs import (
    ModuleConfig,
    OBSERVER_TYPES,
    OBSERVER_PARAMS,
    IMOD_DICT,
)
from neural_compressor.torch.algorithms.fp8_quant.utils.logger import logger
from neural_compressor.torch.algorithms.fp8_quant._quant_common.quant_config import get_hqt_config

__all__ = [
    "ObserverBase",
    "register_observer",
]


class ObserverBase(ABC):
    def __init__(self, name, mod, d_shape=None, params=None, device="hpu"):
        self.name = name
        self.mod = mod
        self.state = None
        self.used = False
        self.device = device

    @abstractmethod
    def measure(self, x):
        raise NotImplementedError("`measure` is not implemented")

    def is_used(self):
        return self.used

    @abstractmethod
    def update_state(self, x):
        raise NotImplementedError("`update_state` is not implemented")


def register_observer(observer_type):
    """Decorate and register all observer subclasses.

    Args:
        observer_type (str): The observer registration name.
    """
    def decorator(cls):
        if observer_type in OBSERVER_TYPES:
            logger.info("Overwrite the existing observer {}.".format(observer_type))
        OBSERVER_TYPES[observer_type] = cls
        return cls

    return decorator


def register_module_config_for_observer(module_name, inputs_param=(None,), outputs_param=(None,), weight_param=None, **kwargs):
    """Decorate and register module config for specific observer.

    Args:
        module_name (str): The module registration name.
        inputs_param (tuple): The parameter of inputs. It is a tuple of dicts: (input1 param, input2 param, ...).
        outputs_param (tuple): The parameter of outputs. It is a tuple of dicts: (output1 param, output2 param, ...).
        weight_param (dict): The parameter of weight, can be used by some modules like linear.
    """
    def decorator(cls):
        if cls not in OBSERVER_TYPES.values():
            raise ValueError(
                f"Please register observer first and then register module_config like below: \n"
                f"@register_module_config_for_observer(module_name=module_name, inputs_param=..., outputs_param=..., ...) \n"
                f"@register_observer(observer_type=observer_type) \n"
                f"class CustomizedObserver(ObserverBase): \n"
                f"... \n"
            )
        observer_type = list(OBSERVER_TYPES.keys())[list(OBSERVER_TYPES.values()).index(cls)]
        if observer_type in OBSERVER_PARAMS and module_name in OBSERVER_PARAMS[observer_type]:
            logger.info("Overwrite the existing param of {} for {} observer.".format(module_name, observer_type))
        if weight_param is not None:
            kwargs.update({"weight": weight_param})
        OBSERVER_PARAMS.setdefault(observer_type, {})[module_name] = ModuleConfig(
            inputs=inputs_param, outputs=outputs_param, params=kwargs,
        )
        return cls

    return decorator



@register_observer(observer_type="maxabs")
class MaxAbsObserver(ObserverBase):

    def __init__(self, name, mod, d_shape=None, params=None, device="hpu"):
        super().__init__(name=name, mod=mod, device=device)
        self.first = True
        config = get_hqt_config(mod).cfg
        self.state = torch.zeros((1, 1), device=self.device, dtype=config["hp_dtype"])

    def update_state(self, x):
        self.state.copy_(torch.maximum(torch.max(torch.abs(x)).detach(), self.state))

    def measure(self, x):
        self.update_state(x)
        self.used = True


@register_module_config_for_observer(module_name="linear", inputs_param=({"dim": -1},), outputs_param=({"dim": -1},), weight_param={"dim": 0})
@register_module_config_for_observer(module_name="matmul", inputs_param=({"dim": -1}, {"dim": -2},), outputs_param=({"dim": -1},))
@register_observer(observer_type="maxabs_per_channel")
class MaxAbsPerChannelObserver(ObserverBase):
    def __init__(self, name, mod, d_shape=None, params=None, device="hpu"):
        super().__init__(name=name, mod=mod, device=device)
        self.first = True
        self.dim = params["dim"] if (params is not None) and ("dim" in params) else -1
        if d_shape is not None:
            p = list(range(len(d_shape)))
            self.dim = self.dim if self.dim >= 0 else len(d_shape) + self.dim
            p[-1] = self.dim
            p[self.dim] = len(d_shape) - 1
            self.p = p
            self.state = self.init_state_from_shape(d_shape)

    def init_state(self, x):
        device = x.device
        Nch = x.shape[self.dim]
        self.Nch = Nch
        state = torch.zeros((Nch, 1), device=device, dtype=torch.float32)
        self.shape = list(x.shape)
        return state

    def init_state_from_shape(self, x_shape, device="hpu"):
        device = device
        Nch = x_shape[self.dim]
        self.Nch = Nch
        state = torch.zeros((Nch, 1), device=device, dtype=torch.float32)
        self.first = False
        return state

    def update_state(self, x):
        self.state.copy_(
            torch.maximum(
                torch.max(
                    torch.abs(x.permute(self.p).reshape([-1, self.Nch])),
                    dim=0,
                    keepdim=True,
                )[0].t(),
                self.state,
            )
        )

    def measure(self, x):
        if self.first:
            self.state = self.init_state(x)
            self.first = False
        self.update_state(x)
        self.used = True


@register_observer(observer_type="save")
class SaveObserver(ObserverBase):
    def __init__(self, name, mod, d_shape=None, params=None, device="hpu"):
        super().__init__(name=name, mod=mod, device=device)
        self.first = True
        self.cnt = -1
        config = get_hqt_config(mod).cfg
        self.folder_name = os.path.join(config["dump_stats_base_path"], "tensors")
        os.makedirs(self.folder_name, exist_ok=True)
        self.file_base_name = os.path.join(self.folder_name, IMOD_DICT[mod] + "_" + name + "_iter")
        self.state = self.init_state_from_shape(d_shape)

    def init_state(self, x):
        device = x.device
        state = torch.zeros((1, 1), device=device, dtype=torch.float32)
        self.shape = list(x.shape)
        return state

    def init_state_from_shape(self, x_shape, device="hpu"):
        state = torch.zeros((1, 1), device=device, dtype=torch.float32)
        self.first = False
        return state

    def update_state(self, x):
        self.cnt += 1
        torch.save(x, self.file_base_name + str(self.cnt) + ".pt")

    def measure(self, x):
        self.update_state(x)
        self.used = True


@register_observer(observer_type="shape")
class ShapeObserver(ObserverBase):
    def __init__(self, name, mod, d_shape=None, params=None, device="hpu"):
        super().__init__(name=name, mod=mod, device=device)

    def init_state(self, x):
        device = x.device
        Ndim = len(x.shape)
        self.Ndim = Ndim
        state = torch.tensor(x.shape, device=device, dtype=torch.int32).reshape((1, Ndim))
        return state

    def init_state_from_shape(self, x_shape, device="hpu"):
        logger.info("ShapeObserver doesn't support init_state_from_shape")
        return

    def update_state(self, x):
        logger.info("ShapeObserver doesn't support update_state")
        return

    def measure(self, x):
        self.state = self.init_state(x)

    def is_used(self):
        return self.state is not None