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
"""Model domain class."""
from neural_compressor.ux.utils.json_serializer import JsonSerializer


class Domain(JsonSerializer):
    """Model domain definition."""

    def __init__(self, domain: str = "", domain_flavour: str = "") -> None:
        """Object construction."""
        super().__init__()

        if not domain and domain_flavour:
            raise ValueError("Domain must be set when given flavour")

        self.domain = domain
        self.domain_flavour = domain_flavour
