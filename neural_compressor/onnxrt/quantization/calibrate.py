#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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

import abc

from onnxruntime.quantization import CalibrationDataReader as ORTCalibrationDataReader

__all__ = ["CalibrationDataReader"]


class CalibrationDataReader(ORTCalibrationDataReader):
    """Get data for calibration.

    We define our CalibrationDataReader based on the class in below link:
    https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/calibrate.py#L139
    """

    @abc.abstractmethod
    def rewind(self):
        """Regenerate data."""
        raise NotImplementedError
