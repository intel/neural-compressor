# Copyright (c) 2023 Intel Corporation
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
"""The utils of common module."""

from neural_compressor.common.utils.constants import *
from neural_compressor.common.utils.logger import *
from neural_compressor.common.utils.save_load import save_config_mapping, load_config_mapping

# ! Put the following `utility` import after the `logger` import as `utility` used `logger`
from neural_compressor.common.utils.utility import *


# FIXME: (Yi) REMOVE BELOW CODE
import os

DEEPSEEK_EXPERTS = 256
VLLM_EP_SIZE = int(os.getenv("VLLM_EP_SIZE", None))
NUM_EXPERTS_PER_EP_RANK = DEEPSEEK_EXPERTS // VLLM_EP_SIZE  # 32
NUM_EXPERTS_GROUPS = 8
NUM_EXPERTS_PER_GROUP_PER_RANK = NUM_EXPERTS_PER_EP_RANK // NUM_EXPERTS_GROUPS # 4
FUSED_MOE_EXPERTS = NUM_EXPERTS_PER_GROUP_PER_RANK  # 4


import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
