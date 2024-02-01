# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

SPR_BASE_VERSIONS = (
    "2.11.0202242",
    "2.11.0202250",
    "2.11.0202317",
    "2.11.0202323",
    "2.14.0202335",
    "2.14.dev202335",
    "2.15.0202341",
)

DEFAULT_SQ_ALPHA_ARGS = {
    "alpha_min": 0.0,
    "alpha_max": 1.0,
    "alpha_step": 0.1,
    "shared_criterion": "mean",
    "do_blockwise": False,
}
