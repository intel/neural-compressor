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
"""File browser."""

import os
from typing import Any, Dict, List

from lpot.ux.utils.exceptions import (
    AccessDeniedException,
    ClientErrorException,
    NotFoundException,
)
from lpot.ux.utils.utils import is_hidden, is_model_file


def get_directory_entries(
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Get directory entries."""
    try:
        path = os.path.abspath(data["path"][0])
        contents = get_non_hidden_directory_entries(path)

        show_files = should_show_files(data)
        if not show_files:
            contents = list(filter(lambda e: "file" != e["type"], contents))

        show_only_models = should_show_only_model_files(data)
        if show_only_models:
            contents = list(
                filter(
                    lambda e: "directory" == e["type"] or is_model_file(e["name"]),
                    contents,
                ),
            )

        return {
            "path": path,
            "contents": sort_entries(contents),
        }
    except PermissionError as err:
        raise AccessDeniedException(err)
    except FileNotFoundError as err:
        raise NotFoundException(err)
    except NotADirectoryError as err:
        raise ClientErrorException(err)


def get_non_hidden_directory_entries(path: str) -> List:
    """Build a list of entries for path."""
    entries = []

    with os.scandir(path) as it:
        for entry in it:
            if is_hidden(entry.path):
                continue
            if entry.is_dir():
                entries.append(create_dir_entry(entry))
            if entry.is_file():
                entries.append(create_file_entry(entry))

    return entries


def create_dir_entry(entry: os.DirEntry) -> Dict:
    """Build a Directory entry."""
    return create_entry(entry.path, True)


def create_file_entry(entry: os.DirEntry) -> Dict:
    """Build a File entry."""
    return create_entry(entry.path, False)


def create_entry(path: str, is_directory: bool) -> Dict:
    """Build an entry."""
    entry_type = "directory" if is_directory else "file"
    return {
        "name": path,
        "type": entry_type,
    }


def sort_entries(entries: List) -> List:
    """Sort entries, directories first, files second."""
    entries.sort(key=lambda e: f"{e['type']} {e['name']}")
    return entries


def should_show_files(data: Dict[str, Any]) -> bool:
    """Decide if files be returned."""
    return get_setting_value("files", True, data)


def should_show_only_model_files(data: Dict[str, Any]) -> bool:
    """Decide if files be returned."""
    return get_setting_value("models_only", False, data)


def get_setting_value(setting: str, default: bool, data: Dict[str, Any]) -> bool:
    """Get bool value from parameters."""
    try:
        not_default = repr(not default).lower()
        tested_value = data[setting][0].lower()

        # opposite of default MUST be provided explicit to return not default
        return not default if not_default == tested_value else default
    except KeyError:
        return default
