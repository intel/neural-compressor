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
"""Package jobs_management contains all packages required to manage jobs."""
from neural_compressor.ux.components.jobs_management.jobs_control_queue import _JobsControlQueue
from neural_compressor.ux.components.jobs_management.jobs_manager import _Job, _JobsManager

jobs_control_queue = _JobsControlQueue()
jobs_manager = _JobsManager(jobs_control_queue)
parse_job_id = _Job.parse_job_id
