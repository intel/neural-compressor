# SPDX-License-Identifier: Apache-2.0
"""QDQ (Quant-Dequant) implementations for different data types."""

from .mxfp4 import mxfp4_qdq
from .mxfp8 import mxfp8_qdq

__all__ = ["mxfp4_qdq", "mxfp8_qdq"]
