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

from dataclasses import asdict, dataclass
from typing import Tuple, Union

import torch

__all__ = [
    "QTensor",
    "QTensorMetaInfo",
]


@dataclass
class QTensorMetaInfo:
    nbits: int
    group_size: int
    shape: Tuple
    axis: int
    packing: bool

    def to_dict(self):
        return asdict(self)


class QTensor:
    val: torch.Tensor
    scale: Union[torch.Tensor, "QTensor"] = None
    zero: Union[torch.Tensor, "QTensor"] = None
    meta_info: QTensorMetaInfo = None
    """
    val: torch.Tensor
    scale:
        val: torch.Tensor
        scale: torch.Tensor
        zero: torch.Tensor
    zero:
        torch.Tensor
    """

    def __init__(self, val, scale=None, zero=None, meta_info=None):
        self.val = val
        self.scale = scale
        self.zero = zero
        self.meta_info = meta_info

    def is_scale_quantized(self) -> bool:
        return isinstance(self.scale, QTensor)

    def is_zero_quantized(self) -> bool:
        return isinstance(self.zero, QTensor)

    def _get_scale_repr(self) -> str:
        if not self.is_scale_quantized():
            if self.scale is not None:
                return (
                    f"scale_shape={self.scale.shape}, "
                    f"scale_dtype={self.scale.dtype}, "
                    f"scale_device={self.scale.device}\n"
                )
            else:
                return "scale is None\n"
        else:
            return self.scale.__repr__() + "\n"

    def _get_zero_repr(self) -> str:
        if not self.is_zero_quantized():
            if self.zero is not None:
                return (
                    f"zero_shape={self.zero.shape}, "
                    f"zero_dtype={self.zero.dtype}, "
                    f"zero_device={self.zero.device}\n"
                )
            else:
                return "zero is None\n"
        else:
            return self.zero.__repr__() + "\n"

    def __repr__(self) -> str:
        # TODO: refine it later
        return (
            f"QTensor(\n"
            f"val_shape={self.val.shape}, val_dtype={self.val.dtype}, val_device={self.val.device}\n"
            f"scale_quantized={self.is_scale_quantized()},\n"
            f"zero_quantized={self.is_zero_quantized()},\n"
            f"zero=({self._get_zero_repr()})"
            f"scale=({self._get_scale_repr()})"
            f"meta_info={self.meta_info}\n)"
        )

    def to(self, *args, **kwargs):
        self.val = self.val.to(*args, **kwargs)
        self.scale = self.scale.to(*args, **kwargs)
        self.zero = self.zero.to(*args, **kwargs)
        return self

    def half(self):
        # TODO: refine it later
        if self.val.dtype == torch.float32:
            self.val = self.val.half()
        if self.scale is not None:
            self.scale = self.scale.half()
        if self.zero is not None:
            self.zero = self.zero.half()
        return self
