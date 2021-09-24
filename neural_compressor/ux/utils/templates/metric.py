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
"""Metric class."""

from typing import Optional

from neural_compressor.ux.utils.json_serializer import JsonSerializer

DIGITS = 4


class Metric(JsonSerializer):
    """Metric represents data which is sent from Optimization and Benchmark."""

    # TODO: Split into Accuracy, Performance if necessary for Benchmark

    def __init__(self) -> None:
        """Initialize configuration Dataset class."""
        super().__init__()
        self._acc_input_model: Optional[float] = None
        self._acc_optimized_model: Optional[float] = None
        self._perf_latency_input_model: Optional[float] = None
        self._perf_latency_optimized_model: Optional[float] = None
        self._perf_throughput_optimized_model: Optional[float] = None
        self._perf_throughput_input_model: Optional[float] = None
        self._optimization_time: Optional[float] = None
        self._size_input_model: Optional[float] = None
        self._size_optimized_model: Optional[float] = None
        self._path_optimized_model: Optional[str] = None

    @property
    def acc_input_model(self) -> Optional[float]:
        """Accuracy for float32."""
        return self._acc_input_model

    @acc_input_model.setter
    def acc_input_model(self, value: str) -> None:
        """Set input model accuracy from value."""
        float_value = float(value)
        if float_value > 1:
            float_value /= 100
        self._acc_input_model = round(float_value, DIGITS)

    @property
    def acc_optimized_model(self) -> Optional[float]:
        """Accuracy for optimized model."""
        return self._acc_optimized_model

    @acc_optimized_model.setter
    def acc_optimized_model(self, value: str) -> None:
        """Set optimized model accuracy from value."""
        float_value = float(value)
        if float_value > 1:
            float_value /= 100
        self._acc_optimized_model = round(float_value, DIGITS)

    @property
    def perf_latency_input_model(self) -> Optional[float]:
        """Latency for input model."""
        return self._perf_latency_input_model

    @perf_latency_input_model.setter
    def perf_latency_input_model(self, value: str) -> None:
        """Set latency for input model."""
        self._perf_latency_input_model = round(float(value), DIGITS)
        if not self.perf_throughput_input_model:
            self.perf_throughput_input_model = self.calculate_throughput(
                self._perf_latency_input_model,
            )

    @property
    def perf_latency_optimized_model(self) -> Optional[float]:
        """Latency for optimized model."""
        return self._perf_latency_optimized_model

    @perf_latency_optimized_model.setter
    def perf_latency_optimized_model(self, value: str) -> None:
        """Set latency for optimized model."""
        self._perf_latency_optimized_model = round(float(value), DIGITS)
        if not self.perf_throughput_optimized_model:
            self.perf_throughput_optimized_model = self.calculate_throughput(
                self._perf_latency_optimized_model,
            )

    @property
    def perf_throughput_optimized_model(self) -> Optional[float]:
        """Throughput for optimized model."""
        return self._perf_throughput_optimized_model

    @perf_throughput_optimized_model.setter
    def perf_throughput_optimized_model(self, value: str) -> None:
        """Set throughput from value for optimized model."""
        self._perf_throughput_optimized_model = round(float(value), DIGITS)

    @property
    def perf_throughput_input_model(self) -> Optional[float]:
        """Throughput for input model."""
        return self._perf_throughput_input_model

    @perf_throughput_input_model.setter
    def perf_throughput_input_model(self, value: str) -> None:
        """Set throughput from value for input model."""
        self._perf_throughput_input_model = round(float(value), DIGITS)

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
    def size_input_model(self) -> Optional[float]:
        """Model size for float32."""
        return self._size_input_model

    @size_input_model.setter
    def size_input_model(self, value: str) -> None:
        """Set input model size from value."""
        self._size_input_model = float(value)

    @property
    def size_optimized_model(self) -> Optional[float]:
        """Get optimized model size."""
        return self._size_optimized_model

    @size_optimized_model.setter
    def size_optimized_model(self, value: str) -> None:
        """Set optimized  model size optimized from value."""
        self._size_optimized_model = float(value)

    @property
    def path_optimized_model(self) -> Optional[str]:
        """Get optimized model path."""
        return self._path_optimized_model

    @path_optimized_model.setter
    def path_optimized_model(self, value: str) -> None:
        """Set optimized model path."""
        self._path_optimized_model = value

    @property
    def optimization_time(self) -> Optional[float]:
        """Optimization time."""
        return self.optimization_time

    @optimization_time.setter
    def optimization_time(self, value: str) -> None:
        """Set optimization_time value."""
        self.optimization_time = float(value)
