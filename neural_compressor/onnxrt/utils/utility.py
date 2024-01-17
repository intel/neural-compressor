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

from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import onnx
from packaging.version import Version

from neural_compressor.common.logger import Logger

logger = Logger().get_logger()

ONNXRT116_VERSION = Version("1.16.0")
ONNXRT1161_VERSION = Version("1.16.1")

# Dictionary to store a mapping between algorithm names and corresponding algo implementation(function)
algos_mapping: Dict[str, Callable] = {}

# All constants for onnxrt
WHITE_MODULE_LIST = ["MatMul", "Conv"]

MAXIMUM_PROTOBUF = 2147483648

dtype_mapping = {
    "fp32": 1,
    "float32": 1,
    "uint8": 2,
    "int8": 3,
    "uint16": 4,
    "int16": 5,
    "int32": 6,
    "int64": 7,
    "string": 8,
    "bool": 9,
    "fp16": 10,
    "float16": 10,
    "double": 11,
    "uint32": 12,
    "uint64": 13,
    "complex64": 14,
    "complex128": 15,
    "bf16": 16,
    "bfloat16": 16,
}


def find_by_name(name, item_list):
    """Helper function to find item by name in a list."""
    items = []
    for item in item_list:
        assert hasattr(item, "name"), "{} should have a 'name' attribute defined".format(item)  # pragma: no cover
        if item.name == name:
            items.append(item)
    if len(items) > 0:
        return items[0]
    else:
        return None


def simple_progress_bar(total, i):
    """Progress bar for cases where tqdm can't be used."""
    progress = i / total
    bar_length = 20
    bar = "#" * int(bar_length * progress)
    spaces = " " * (bar_length - len(bar))
    percentage = progress * 100
    print(f"\rProgress: [{bar}{spaces}] {percentage:.2f}%", end="")


def register_algo(name):
    """Decorator function to register algorithms in the algos_mapping dictionary.

    Usage example:
        @register_algo(name=example_algo)
        def example_algo(model: Union[onnx.ModelProto, Path, str],
                         quant_config: RTNWeightQuantConfig) -> onnx.ModelProto:
            ...

    Args:
        name (str): The name under which the algorithm function will be registered.

    Returns:
        decorator: The decorator function to be used with algorithm functions.
    """

    def decorator(algo_func):
        algos_mapping[name] = algo_func
        return algo_func

    return decorator


def get_model_info(
    model: Union[onnx.ModelProto, Path, str], white_op_type_list: List[Callable]
) -> List[Tuple[str, Callable]]:
    if not isinstance(model, onnx.ModelProto):
        model = onnx.load(model)
    filter_result = []
    filter_result_set = set()
    for node in model.graph.node:
        if node.op_type in white_op_type_list:
            pair = (node.name, node.op_type)
            if pair not in filter_result_set:
                filter_result_set.add(pair)
                filter_result.append(pair)
    logger.debug(f"Get model info: {filter_result}")
    return filter_result
