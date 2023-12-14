# Copyright (c) 2023 Intel Corporation
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

import os
from typing import Any, Optional

import torch
from torch.types import _dtype


class autocast:
    r"""Instances of :class:`autocast` serve as context managers or decorators that
    allow regions of your script to run in mixed precision.

    In these regions, ops run in an op-specific dtype chosen by autocast
    to improve performance while maintaining accuracy.

    When entering an autocast-enabled region, Tensors may be any type.
    You should not call ``half()`` or ``bfloat16()`` on your model(s) or inputs when using autocasting.

    :class:`autocast` should wrap only the forward pass(es) of your network, including the loss
    computation(s).  Backward passes under autocast are not recommended.
    Backward ops run in the same type that autocast used for corresponding forward ops.

        # Enables autocasting for the inference pass
        with torch.autocast(device_type="hpu", dtype=torch.float8_e4m3fn):
            output = model(input)

    :class:`autocast` can also be used as a decorator, e.g., on the ``forward`` method of your model::

        class AutocastModel(nn.Module):
            ...
            @torch.autocast(device_type="cuda")
            def forward(self, input):
                ...

    The autocast state is thread-local.  If you want it enabled in a new thread, the context manager or decorator
    must be invoked in that thread.  This affects :class:`torch.nn.DataParallel` and
    :class:`torch.nn.parallel.DistributedDataParallel` when used with more than one GPU per process
    (see :ref:`Working with Multiple GPUs<amp-multigpu>`).

    Args:
        device_type(str, required):  Device type to use. Possible values are: 'cuda', 'cpu', 'xpu' and 'hpu'.
                                     The type is the same as the `type` attribute of a :class:`torch.device`.
                                     Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
        enabled(bool, optional):  Whether autocasting should be enabled in the region.
            Default: ``True``
        dtype(torch_dtype, optional):  Whether to use torch.float16 or torch.bfloat16.
        cache_enabled(bool, optional):  Whether the weight cache inside autocast should be enabled.
            Default: ``True``
    """

    def __init__(
        self,
        device_type: str,
        dtype: Optional[_dtype] = None,
        enabled: bool = True,
        cache_enabled: Optional[bool] = None,
    ):
        self.device = device_type
        if dtype is not None:
            self.fast_dtype = dtype
        if cache_enabled is not None:
            self._cache_enabled = cache_enabled
        if not (device_type == "hpu" and dtype in [torch.float8_e4m3fn, torch.float8_e5m2]):
            self._autocast = torch.autocast(device_type, dtype, enabled, cache_enabled)

    def __enter__(self) -> None:
        if self.device == "hpu" and self.fast_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            from neural_compressor.torch.amp.fp8.functions import replace_func

            # This function will replace F.linear and torch.matmul with the fp8 one
            replace_func(self.fast_dtype)
        else:
            self._autocast.__enter__()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.device == "hpu" and self.fast_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            from neural_compressor.torch.amp.fp8.functions import recover_func

            # This function will recover F.linear and torch.matmul with the original one
            recover_func()
        else:
            self._autocast.__exit__(exc_type, exc_value, traceback)
