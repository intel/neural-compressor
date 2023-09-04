#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
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
"""Built-in COCO filter."""

from neural_compressor.utils.utility import LazyImport

from .filter import Filter, filter_registry

tf = LazyImport("tensorflow")


@filter_registry(filter_type="LabelBalanceCOCORecord", framework="tensorflow, tensorflow_itex")
class LabelBalanceCOCORecordFilter(Filter):  # pragma: no cover
    """The label balance filter for COCO Record."""

    def __init__(self, size=1):
        """Initialize the attribute of class."""
        self.size = size

    def __call__(self, image, label):
        """Execute the filter.

        Args:
            image: Not used.
            label: label of a sample.
        """
        return tf.math.equal(len(label[0]), self.size)


@filter_registry(
    filter_type="LabelBalanceCOCORaw",
    framework="tensorflow, \
                 tensorflow_itex, pytorch, mxnet, onnxrt_qlinearops, onnxrt_integerops",
)
class LabelBalanceCOCORawFilter(Filter):  # pragma: no cover
    """The label balance filter for COCO raw data."""

    def __init__(self, size=1):
        """Initialize the attribute of class."""
        self.size = size

    def __call__(self, image, label):
        """Execute the filter.

        Args:
            image: Not used.
            label: label of a sample.
        """
        return len(label) == self.size
