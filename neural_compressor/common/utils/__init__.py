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

# DEEPSEEK_EXPERTS = 256
# VLLM_TP_SIZE = int(os.getenv("VLLM_TP_SIZE", "8"))
# VLLM_EP_SIZE = int(os.getenv("VLLM_EP_SIZE", VLLM_TP_SIZE))
# NUM_EXPERTS_PER_EP_RANK = DEEPSEEK_EXPERTS // VLLM_EP_SIZE  # 32
# VLLM_MOE_N_SLICE = int(os.getenv("VLLM_MOE_N_SLICE", 8))
# NUM_EXPERTS_PER_GROUP_PER_RANK = NUM_EXPERTS_PER_EP_RANK // VLLM_MOE_N_SLICE # 4
# FUSED_MOE_EXPERTS = NUM_EXPERTS_PER_GROUP_PER_RANK  # 4

# logger.warning_once(
#     (
#         f"INC uses VLLM_TP_SIZE={VLLM_TP_SIZE},\n"
#         f"VLLM_EP_SIZE={VLLM_EP_SIZE},\n"
#         f"NUM_EXPERTS_PER_EP_RANK={NUM_EXPERTS_PER_EP_RANK},\n"
#         f"VLLM_MOE_N_SLICE={VLLM_MOE_N_SLICE},\n"
#         f"NUM_EXPERTS_PER_GROUP_PER_RANK={NUM_EXPERTS_PER_GROUP_PER_RANK},\n"
#         f"FUSED_MOE_EXPERTS={FUSED_MOE_EXPERTS}"
#     )
# )

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
