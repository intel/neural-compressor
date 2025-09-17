# Copyright (c) 2025 Intel Corporation
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


# File contains logic for importing vLLM distributed functions, such as we import them only when needed and possible,
# and protect from import errors in case vLLM is not installed.

try:
    from vllm.distributed import (
        tensor_model_parallel_all_gather,
        tensor_model_parallel_all_reduce,
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
except ImportError:
    tensor_model_parallel_all_gather = None
    tensor_model_parallel_all_reduce = None
    get_tensor_model_parallel_rank = None
    get_tensor_model_parallel_world_size = None

def get_vllm_column_parallel_collective_func():
    assert tensor_model_parallel_all_gather is not None, "Couldn't import vllm function tensor_model_parallel_all_gather"
    return tensor_model_parallel_all_gather

def get_vllm_row_parallel_collective_func():
    assert tensor_model_parallel_all_reduce is not None, "Couldn't import vllm function tensor_model_parallel_all_reduce"
    return tensor_model_parallel_all_reduce

def get_vllm_tensor_model_parallel_rank_func():
    assert get_tensor_model_parallel_rank is not None, "Couldn't import vllm function get_tensor_model_parallel_rank"
    return get_tensor_model_parallel_rank

def get_vllm_tensor_model_parallel_world_size_func():
    assert get_tensor_model_parallel_world_size is not None, "Couldn't import vllm function get_tensor_model_parallel_world_size"
    return get_tensor_model_parallel_world_size