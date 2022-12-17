# -*- coding: utf-8 -*-
# Copyright (c) 2022 Intel Corporation
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
"""Defines data structure of request object."""
from dataclasses import dataclass
from enum import Enum
from subprocess import Popen
from threading import Event
from typing import Any, Callable, Iterable, Optional


class _RequestType(Enum):
    """Define possible types of request object."""

    SCHEDULE = "add new job to queue"
    ABORT = "abort process of specified job if exists else kill it right after it spawns"
    DELETE_JOB = "delete job from job dict"
    ADD_PROCESS_HANDLE = "add process handle to job instance"


@dataclass(frozen=True)
class _Request:
    """Data structure to store information about job managment requests."""

    type: _RequestType
    job_id: str
    target: Optional[Callable] = None
    args: Optional[Iterable[Any]] = None
    request_id: Optional[str] = None
    process_handle: Optional[Popen] = None
    event: Optional[Event] = None
