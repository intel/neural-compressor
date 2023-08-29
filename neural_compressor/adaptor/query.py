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
import logging
from abc import abstractmethod


class QueryBackendCapability:
    """Base class that defines Query Interface.

    Each adaption layer should implement the inherited class for specific backend on their own.
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_version(self):
        """Get the current backend's version string."""
        raise NotImplementedError

    @abstractmethod
    def get_precisions(self):
        """Get the supported low precisions, e.g ['int8', 'bf16']"""
        raise NotImplementedError

    @abstractmethod
    def get_op_types(self):
        """Get the op types for specific backend per low precision.

        e.g {'2.3.0': {'int8': ['Conv2D', 'MatMuL']}}
        """
        raise NotImplementedError

    @abstractmethod
    def get_fuse_patterns(self):
        """Get the fusion patterns for specified op type for every specific precision."""
        raise NotImplementedError

    @abstractmethod
    def set_quantization_config(self, q_config):
        """Set the quantization config to backend.

        Args:
            q_config (yaml content?): set the organized quantization configuration to backend.
        """
        raise NotImplementedError

    @abstractmethod
    def get_quantization_capability(self):
        """Get the quantization capability of low precision op types.

        e.g, granularity, scheme and etc.
        """
        raise NotImplementedError

    @abstractmethod
    def get_mixed_precision_combination(self, unsupported_precisions):
        """Get the valid precision combination base on hardware and user' config.

        e.g['fp32', 'bf16', 'int8']
        """
        raise NotImplementedError
