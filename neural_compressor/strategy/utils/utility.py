#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
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

"""Tuning utility."""
import os
import pickle
from collections import OrderedDict
from typing import List, Optional, Any

import prettytable

from neural_compressor.utils import logger
from neural_compressor.utils.utility import print_table, dump_table


class OrderedDefaultDict(OrderedDict):
    """Ordered default dict."""

    def __missing__(self, key):
        """Initialize value for the missing key."""
        self[key] = value = OrderedDefaultDict()
        return value


def extract_data_type(data_type: str) -> str:
    """Extract data type and signed from data type.

    Args:
        data_type: The original data type such as uint8, int8.

    Returns:
        (signed or unsigned, data type without signed)
    """
    return ('signed', data_type) if data_type[0] != 'u' else ('unsigned', data_type[1:])


def reverted_data_type(signed_flag: str, data_type: str) -> str:
    """Revert the data type."""
    return data_type if signed_flag == 'signed' else 'u' + data_type


def get_adaptor_name(adaptor):
    """Get adaptor name.

    Args:
        adaptor: adaptor instance.
    """
    adaptor_name = type(adaptor).__name__.lower()
    adaptor_name_lst = ['onnx', 'tensorflow', 'pytorch']
    for name in adaptor_name_lst:
        if adaptor_name.startswith(name):
            return name
    return ""


class OpEntry:
    """OP entry class."""

    def __init__(self, op_name: str, mse: float, activation_min: float, activation_max: float):
        """Initialize OP entry."""
        self.op_name: str = op_name
        self.mse: float = mse
        self.activation_min: float = activation_min
        self.activation_max: float = activation_max


def print_op_list(workload_location: str):
    """Print OP table."""
    minmax_file_path = os.path.join(workload_location, "inspect_saved", "dequan_min_max.pkl")
    input_model_tensors = get_tensors_info(
        workload_location,
        model_type="input",
    )["activation"][0]
    optimized_model_tensors = get_tensors_info(
        workload_location,
        model_type="optimized",
    )["activation"][0]
    op_list = get_op_list(minmax_file_path, input_model_tensors, optimized_model_tensors)
    sorted_op_list = sorted(op_list, key=lambda x: x.mse, reverse=True)
    if len(op_list) <= 0:
        return
    print_table(
        title="Activations summary",
        column_mapping={
            "OP name": "op_name",
            "MSE": "mse",
            "Activation min": "activation_min",
            "Activation max": "activation_max",
        },
        table_entries=sorted_op_list,
    )

    activations_table_file = os.path.join(
        workload_location,
        "activations_table.csv",
    )
    dump_table(
        filepath=activations_table_file,
        column_mapping={
            "OP name": "op_name",
            "MSE": "mse",
            "Activation min": "activation_min",
            "Activation max": "activation_max",
        },
        table_entries=sorted_op_list,
        file_type="csv",
    )


def get_tensors_info(workload_location, model_type: str = "optimized") -> dict:
    """Get information about tensors."""
    tensors_filenames = {
        "input": os.path.join("fp32", "inspect_result.pkl"),
        "optimized": os.path.join("quan", "inspect_result.pkl"),
    }

    tensors_filename = tensors_filenames.get(model_type, None)
    if tensors_filename is None:
        raise Exception(f"Could not find tensors data for {model_type} model.")
    tensors_path = os.path.join(
        workload_location,
        "inspect_saved",
        tensors_filename,
    )
    if not os.path.exists(tensors_path):
        raise Exception("Could not find tensor data for specified optimization.")
    with open(tensors_path, "rb") as tensors_pickle:
        dump_tensor_result = pickle.load(tensors_pickle)
    return dump_tensor_result


def get_op_list(minmax_file_path, input_model_tensors, optimized_model_tensors) -> List[OpEntry]:
    """Get OP list for model."""
    with open(minmax_file_path, "rb") as min_max_file:
        min_max_data: dict = pickle.load(min_max_file)

    op_list: List[OpEntry] = []

    for op_name, min_max in min_max_data.items():
        mse = calculate_mse(op_name, input_model_tensors, optimized_model_tensors)
        if mse is None:
            continue
        min = float(min_max.get("min", None))
        max = float(min_max.get("max", None))
        op_entry = OpEntry(op_name, mse, min, max)
        op_list.append(op_entry)
    return op_list


def calculate_mse(
    op_name: str,
    input_model_tensors: dict,
    optimized_model_tensors: dict,
) -> Optional[float]:
    """Calculate MSE for specified OP."""
    input_model_op_data = input_model_tensors.get(op_name, None)
    optimized_model_op_data = optimized_model_tensors.get(op_name, None)

    if input_model_op_data is None or optimized_model_op_data is None:
        return None

    mse: float = mse_metric_gap(
        next(iter(input_model_op_data.values()))[0],
        next(iter(optimized_model_op_data.values()))[0],
    )

    return mse


def mse_metric_gap(fp32_tensor: Any, dequantize_tensor: Any) -> float:
    """Calculate the euclidean distance between fp32 tensor and int8 dequantize tensor.

    Args:
        fp32_tensor (tensor): The FP32 tensor.
        dequantize_tensor (tensor): The INT8 dequantize tensor.
    """
    import numpy as np

    fp32_max = np.max(fp32_tensor)  # type: ignore
    fp32_min = np.min(fp32_tensor)  # type: ignore
    dequantize_max = np.max(dequantize_tensor)  # type: ignore
    dequantize_min = np.min(dequantize_tensor)  # type: ignore
    fp32_tensor_norm = fp32_tensor
    dequantize_tensor_norm = dequantize_tensor
    if (fp32_max - fp32_min) != 0:
        fp32_tensor_norm = (fp32_tensor - fp32_min) / (fp32_max - fp32_min)

    if (dequantize_max - dequantize_min) != 0:
        dequantize_tensor_norm = (dequantize_tensor - dequantize_min) / (dequantize_max - dequantize_min)

    diff_tensor = fp32_tensor_norm - dequantize_tensor_norm
    euclidean_dist = np.sum(diff_tensor**2)  # type: ignore
    return euclidean_dist / fp32_tensor.size
