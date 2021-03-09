# -*- coding: utf-8 -*-
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
"""Tuning class."""

from typing import Optional

from lpot.ux.utils.json_serializer import JsonSerializer

DIGITS = 4


class Metric(JsonSerializer):
    """Metric represents data which is sent from Tuning and Benchmark."""

    # TODO: Split into Accuracy, Performance if necessary for Benchmark

    def __init__(self) -> None:
        """Initialize configuration Dataset class."""
        super().__init__()
        self._acc_fp32: Optional[float] = None
        self._acc_int8: Optional[float] = None
        self._perf_latency_fp32: Optional[float] = None
        self._perf_latency_int8: Optional[float] = None
        self._perf_throughput_int8: Optional[float] = None
        self._perf_throughput_fp32: Optional[float] = None
        self._tuning_time: Optional[float] = None
        self._size_fp32: Optional[float] = None
        self._size_int8: Optional[float] = None

    @property
    def acc_fp32(self) -> Optional[float]:
        """Accuracy for float32."""
        return self._acc_fp32

    @acc_fp32.setter
    def acc_fp32(self, value: str) -> None:
        """Set accuracy fp32 from value."""
        float_value = float(value)
        if float_value > 1:
            float_value /= 100
        self._acc_fp32 = round(float_value, DIGITS)

    @property
    def acc_int8(self) -> Optional[float]:
        """Accuracy for int8."""
        return self._acc_int8

    @acc_int8.setter
    def acc_int8(self, value: str) -> None:
        """Set accuracy int8 from value."""
        float_value = float(value)
        if float_value > 1:
            float_value /= 100
        self._acc_int8 = round(float_value, DIGITS)

    @property
    def perf_latency_fp32(self) -> Optional[float]:
        """Latency for fp32."""
        return self._perf_latency_fp32

    @perf_latency_fp32.setter
    def perf_latency_fp32(self, value: str) -> None:
        """Set latency for fp32."""
        self._perf_latency_fp32 = round(float(value), DIGITS)
        if not self.perf_throughput_fp32:
            self.perf_throughput_fp32 = self.calculate_throughput(
                self._perf_latency_fp32,
            )

    @property
    def perf_latency_int8(self) -> Optional[float]:
        """Latency for int8."""
        return self._perf_latency_int8

    @perf_latency_int8.setter
    def perf_latency_int8(self, value: str) -> None:
        """Set latency for int8."""
        self._perf_latency_int8 = round(float(value), DIGITS)
        if not self.perf_throughput_int8:
            self.perf_throughput_int8 = self.calculate_throughput(
                self._perf_latency_int8,
            )

    @property
    def perf_throughput_int8(self) -> Optional[float]:
        """Throughput for int8 model."""
        return self._perf_throughput_int8

    @perf_throughput_int8.setter
    def perf_throughput_int8(self, value: str) -> None:
        """Set throughput from value for int8 model."""
        self._perf_throughput_int8 = round(float(value), DIGITS)

    @property
    def perf_throughput_fp32(self) -> Optional[float]:
        """Throughput for fp32 model."""
        return self._perf_throughput_fp32

    @perf_throughput_fp32.setter
    def perf_throughput_fp32(self, value: str) -> None:
        """Set throughput from value for fp32 model."""
        self._perf_throughput_fp32 = round(float(value), DIGITS)

    def insert_data(self, attribute: str, value: str) -> None:
        """Set attribute value."""
        self.__setattr__(attribute, value)

    @staticmethod
    def calculate_throughput(value: float) -> float:
        """
        Calculate throughput based on latency.

        Right now 1000 represents number of images in dataset.
        TODO: change 1000 to the batch size when Benchmark is ready
        """
        return 1000 / value

    @property
    def size_fp32(self) -> Optional[float]:
        """Model size for float32."""
        return self._size_fp32

    @size_fp32.setter
    def size_fp32(self, value: str) -> None:
        """Set model size fp32 from value."""
        self._size_fp32 = float(value)

    @property
    def size_int8(self) -> Optional[float]:
        """Model size for int8."""
        return self._size_int8

    @size_int8.setter
    def size_int8(self, value: str) -> None:
        """Set model size int8 from value."""
        self._size_int8 = float(value)

    @property
    def tuning_time(self) -> Optional[float]:
        """Tuning time."""
        return self.tuning_time

    @tuning_time.setter
    def tuning_time(self, value: str) -> None:
        """Set tuning_time value."""
        self.tuning_time = float(value)
