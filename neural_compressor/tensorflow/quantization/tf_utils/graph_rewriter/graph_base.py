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
"""Graph Rewrite Base Class."""

import logging
from abc import abstractmethod


class GraphRewriterBase:
    """Graph Rewrite Base class.

    We abstract this base class and define the interface only.

    Args:
        object (model): the input model to be converted.
    """

    def __init__(self, model):
        """Initialization."""
        self.model = model
        self.logger = logging.getLogger("neural_compressor")

    @abstractmethod
    def do_transformation(self):
        """Base Interface that need to be implemented by each sub class."""
        raise NotImplementedError
