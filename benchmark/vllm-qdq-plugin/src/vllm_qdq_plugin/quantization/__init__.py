# SPDX-License-Identifier: Apache-2.0
"""AutoRound NVFP4 quantization methods backed by QDQ and Marlin."""

from .patch import apply_patches

__all__ = ["apply_patches"]
