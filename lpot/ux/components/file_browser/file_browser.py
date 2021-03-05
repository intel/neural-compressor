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
from lpot.ux.utils.utils import (
    is_dataset_file,
    is_hidden,
    is_model_file,
    verify_file_path,
)


def get_directory_entries(
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Get directory entries."""
    try:
        path = get_requested_path(data)
        verify_file_path(path)
        contents = get_non_hidden_directory_entries(path)

        contents = filter_requested_entries(contents, get_filter_value(data))

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


def get_requested_path(data: Dict[str, Any]) -> str:
    """Get name of requested filter."""
    try:
        path = data["path"][0]
    except KeyError:
        path = "."

    return os.path.abspath(path)


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


def get_filter_value(data: Dict[str, Any]) -> str:
    """Get name of requested filter."""
    try:
        return data["filter"][0].lower()
    except KeyError:
        return ""


def filter_requested_entries(entries: List, filter_name: str) -> List:
    """Filter list of entries using provided filter."""
    filter_map = {
        "models": is_model_or_directory_entry,
        "datasets": is_dataset_or_directory_entry,
        "directories": is_directory_entry,
    }

    requested_filter = filter_map.get(filter_name)
    if requested_filter is None:
        return entries

    return list(filter(requested_filter, entries))


def is_directory_entry(entry: Dict) -> bool:
    """Return if given entry is for directory."""
    return "directory" == entry["type"]


def is_model_or_directory_entry(entry: Dict) -> bool:
    """Return if given entry should be shown on model list."""
    return is_model_file(entry["name"]) or is_directory_entry(entry)


def is_dataset_or_directory_entry(entry: Dict) -> bool:
    """Return if given entry should be shown on dataset list."""
    return is_dataset_file(entry["name"]) or is_directory_entry(entry)
