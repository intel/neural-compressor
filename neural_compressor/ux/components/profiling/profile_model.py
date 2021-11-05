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
"""Generic profiling script."""

import argparse
from typing import Any, Optional

from neural_compressor.ux.components.profiling.factory import ProfilerFactory
from neural_compressor.ux.components.profiling.profiler import Profiler


def parse_args() -> Any:
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload-id",
        type=str,
        required=False,
        help="Workload ID.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=False,
        help="Path to model.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    profiler: Optional[Profiler] = ProfilerFactory().get_profiler(
        args.workload_id,
        args.model_path,
    )
    if profiler is None:
        raise Exception("Could not find profiler for workload.")
    profiler.profile_model()
