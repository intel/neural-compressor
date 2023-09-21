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
"""The base filter class for all frameworks."""

from abc import abstractmethod

from neural_compressor.utils.utility import singleton


@singleton
class TensorflowFilters(object):  # pragma: no cover
    """The base filter class for Tensorflow framework."""

    def __init__(self):
        """Initialize the attribute of the class."""
        self.filters = {}
        self.filters.update(TENSORFLOW_FILTERS)


@singleton
class ONNXRTQLFilters(object):  # pragma: no cover
    """The base filter class for ONNXRT framework QLinear mode."""

    def __init__(self):
        """Initialize the attribute of the class."""
        self.filters = {}
        self.filters.update(ONNXRT_QL_FILTERS)


@singleton
class ONNXRTITFilters(object):  # pragma: no cover
    """The base filter class for ONNXRT framework IT mode."""

    def __init__(self):
        """Initialize the attribute of the class."""
        self.filters = {}
        self.filters.update(ONNXRT_IT_FILTERS)


@singleton
class PyTorchFilters(object):  # pragma: no cover
    """The base filter class for PyTorch framework."""

    def __init__(self):
        """Initialize the attribute of the class."""
        self.filters = {}
        self.filters.update(PYTORCH_FILTERS)


@singleton
class MXNetFilters(object):  # pragma: no cover
    """The base filter class for MXNet framework."""

    def __init__(self):
        """Initialize the attribute of the class."""
        self.filters = {}
        self.filters.update(MXNET_FILTERS)


TENSORFLOW_FILTERS = {}
TENSORFLOW_ITEX_FILTERS = {}
ONNXRT_IT_FILTERS = {}
ONNXRT_QL_FILTERS = {}
PYTORCH_FILTERS = {}
MXNET_FILTERS = {}

framework_filters = {
    "tensorflow": TensorflowFilters,
    "tensorflow_itex": TensorflowFilters,
    "pytorch": PyTorchFilters,
    "pytorch_ipex": PyTorchFilters,
    "pytorch_fx": PyTorchFilters,
    "mxnet": MXNetFilters,
    "onnxrt_qlinearops": ONNXRTQLFilters,
    "onnxrt_qdq": ONNXRTQLFilters,
    "onnxruntime": ONNXRTQLFilters,
    "onnxrt_integerops": ONNXRTITFilters,
}

registry_filters = {
    "tensorflow": TENSORFLOW_FILTERS,
    "tensorflow_itex": TENSORFLOW_ITEX_FILTERS,
    "pytorch": PYTORCH_FILTERS,
    "pytorch_ipex": PYTORCH_FILTERS,
    "pytorch_fx": PYTORCH_FILTERS,
    "mxnet": MXNET_FILTERS,
    "onnxrt_integerops": ONNXRT_IT_FILTERS,
    "onnxrt_qdq": ONNXRT_QL_FILTERS,
    "onnxruntime": ONNXRT_QL_FILTERS,
    "onnxrt_qlinearops": ONNXRT_QL_FILTERS,
}


class FILTERS(object):  # pragma: no cover
    """The filter register for all frameworks.

    Args:
        framework (str): frameworks in ["tensorflow", "tensorflow_itex", "mxnet",
                                        "onnxrt_qdq", "pytorch", "pytorch_ipex",
                                        "pytorch_fx", "onnxrt_integerops", "keras",
                                        "onnxrt_qlinearops", "onnxruntime"].
    """

    def __init__(self, framework):
        """Initialize the attribute of class."""
        assert framework in [
            "tensorflow",
            "tensorflow_itex",
            "keras",
            "mxnet",
            "onnxrt_qdq",
            "pytorch",
            "pytorch_ipex",
            "pytorch_fx",
            "onnxrt_integerops",
            "onnxrt_qlinearops",
            "onnxruntime",
        ], "framework support tensorflow pytorch mxnet onnxrt"
        self.filters = framework_filters[framework]().filters
        self.framework = framework

    def __getitem__(self, filter_type):
        """Magic method.

        x[i] is roughly equivalent to type(x).__getitem__(x, index)
        """
        assert filter_type in self.filters.keys(), "filter support {}".format(self.filters.keys())
        return self.filters[filter_type]


def filter_registry(filter_type, framework):  # pragma: no cover
    """Register all transform subclasses.

    Args:
        filter_type (str): fILTER registration name.
        framework (str): support 4 framework including 'tensorflow', 'pytorch', 'mxnet', 'onnxrt'.
        cls (class): The class of register.

    Returns:
        cls: The class of register.
    """

    def decorator_transform(cls):
        """Decorate a class."""
        for single_framework in [fwk.strip() for fwk in framework.split(",")]:
            assert single_framework in [
                "tensorflow",
                "tensorflow_itex",
                "pytorch",
                "pytorch_ipex",
                "pytorch_fx",
                "mxnet",
                "onnxrt_integerops",
                "onnxrt_qdq",
                "onnxrt_qlinearops",
                "onnxruntime",
            ], "The framework support tensorflow mxnet pytorch onnxrt"
            if filter_type in registry_filters[single_framework].keys():
                raise ValueError("Cannot have two transforms with the same name")
            registry_filters[single_framework][filter_type] = cls
        return cls

    return decorator_transform


class Filter(object):  # pragma: no cover
    """The base class for transform.

    __call__ method is needed when write user specific transform.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Execute the filter."""
        raise NotImplementedError
