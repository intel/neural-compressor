#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Copyright (c) 2025 Intel Corporation
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
"""Compatibility shim for the legacy QAT ``QuantLinear``.

The implementation moved to
:mod:`neural_compressor.torch.quantization.fake_quant.modules`. Importing the
old path keeps downstream code working; new code should use
``neural_compressor.torch.quantization.QuantLinear`` (re-exported here) and
``FakeQuantConfig`` instead of ``prepare_qat(mapping=...)``.
"""

from __future__ import annotations

from neural_compressor.torch.quantization.fake_quant.modules import QuantLinear

__all__ = ["QuantLinear"]
