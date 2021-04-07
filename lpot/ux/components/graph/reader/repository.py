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
"""Graph reader repository."""

from typing import Dict

from lpot.ux.utils.exceptions import ClientErrorException
from lpot.ux.utils.utils import get_framework_from_path

from .reader import Reader


class GraphReaderRepository:
    """Model Reader repository."""

    def __init__(self) -> None:
        """Construct repository."""
        self._framework_readers: Dict[str, type] = {}

        self._add_framework_readers()

    def find(self, model_path: str) -> Reader:
        """Find Graph Model Reader for given model."""
        framework = get_framework_from_path(model_path)
        if framework is None:
            raise ClientErrorException(
                f"Models of {model_path} type are not yet supported.",
            )

        reader_name = self._framework_readers.get(framework)
        if reader_name is None:
            raise ClientErrorException(
                f"Models from {framework} framework are not yet supported.",
            )

        return reader_name()

    def _add_framework_readers(self) -> None:
        """Try to add framework readers."""
        try:
            from .tensorflow import TensorflowReader

            self._framework_readers["tensorflow"] = TensorflowReader
        except Exception:
            pass
