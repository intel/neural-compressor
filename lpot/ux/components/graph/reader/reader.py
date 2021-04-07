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
"""Default Model Reader."""

from ..graph import Graph


class Reader:
    """Empty model reader."""

    def __init__(self) -> None:
        """Construct the reader."""
        pass

    def read(self, model_path: str) -> Graph:
        """Read a graph."""
        raise NotImplementedError

    def ensure_model_readable(self, model_path: str) -> None:
        """Throw Exception if can't read model."""
        raise NotImplementedError
