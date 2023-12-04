# -*- coding: utf-8 -*-
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
"""Generic profiler."""


class Profiler:
    """Profiler class."""

    def __init__(self, input_graph: str) -> None:
        """Profiler constructor.

        Args:
            input_graph: path to model.

        Returns:
            None
        """
        self.input_graph = input_graph

    def profile_model(self, *args, **kwargs) -> None:
        """Profile model."""
        raise NotImplementedError
