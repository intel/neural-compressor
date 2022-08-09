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
"""The diagnosis class."""
import os
import pickle
from typing import Any, List, Optional

from neural_compressor.ux.components.diagnosis.op_details import OpDetails
from neural_compressor.ux.components.diagnosis.op_entry import OpEntry
from neural_compressor.ux.components.optimization.optimization import Optimization
from neural_compressor.ux.utils.exceptions import ClientErrorException, InternalException
from neural_compressor.ux.utils.utils import check_module


class Diagnosis:
    """Diagnosis class."""

    def __init__(self, optimization: Optimization):
        """Initialize diagnosis."""
        self.optimization: Optimization = optimization
        self.config_path = self.optimization.config_path
        self.model_path = self.optimization.output_graph

    def get_tensors_info(self, model_type: str = "optimized") -> dict:
        """Get information about tensors."""
        tensors_filenames = {
            "input": os.path.join("fp32", "inspect_result.pkl"),
            "optimized": os.path.join("quan", "inspect_result.pkl"),
        }

        tensors_filename = tensors_filenames.get(model_type, None)
        if tensors_filename is None:
            raise InternalException(f"Could not find tensors data for {model_type} model.")
        tensors_path = os.path.join(
            self.optimization.workdir,
            tensors_filename,
        )
        if not os.path.exists(tensors_path):
            raise ClientErrorException("Could not find tensor data for specified optimization.")
        with open(tensors_path, "rb") as tensors_pickle:
            dump_tensor_result = pickle.load(tensors_pickle)
        return dump_tensor_result

    def load_quantization_config(self) -> dict:
        """Get config quantization data."""
        config_path = os.path.join(
            self.optimization.workdir,
            "cfg.pkl",
        )
        if not os.path.exists(config_path):
            raise ClientErrorException("Could not find config data for specified optimization.")
        with open(config_path, "rb") as config_pickle:
            config_data = pickle.load(config_pickle)
        return config_data

    def get_op_list(self) -> List[dict]:
        """Get OP list for model."""
        minmax_file_path = os.path.join(
            self.optimization.workdir,
            "dequan_min_max.pkl",
        )
        with open(minmax_file_path, "rb") as min_max_file:
            min_max_data: dict = pickle.load(min_max_file)

        op_list: List[dict] = []
        for op_name, min_max in min_max_data.items():
            mse = self.calculate_mse(op_name)
            if mse is None:
                continue
            min = float(min_max.get("min", None))
            max = float(min_max.get("max", None))
            op_entry = OpEntry(op_name, mse, min, max)
            op_list.append(op_entry.serialize())
        return op_list

    def calculate_mse(self, op_name: str) -> Optional[float]:
        """Calculate MSE for specified OP."""
        input_model_tensors: dict = self.get_tensors_info(model_type="input")["activation"][0]
        optimized_model_tensors: dict = self.get_tensors_info(model_type="optimized")[
            "activation"
        ][0]

        input_model_op_data = input_model_tensors.get(op_name, None)
        optimized_model_op_data = optimized_model_tensors.get(op_name, None)

        if input_model_op_data is None or optimized_model_op_data is None:
            return None

        mse: float = self.mse_metric_gap(
            list(input_model_op_data.values())[0],
            list(optimized_model_op_data.values())[0],
        )

        return mse

    def get_op_details(self, name: str) -> Optional[OpDetails]:
        """Get details of specific OP."""
        config_data = self.load_quantization_config()
        for op_tuple, op_details in config_data["op"].items():
            if op_tuple[0] == name:
                return OpDetails(name, op_details)
        return None

    def get_histogram_data(self, op_name: str, inspect_type: str) -> list:
        """Get data to draw histogram."""
        tensors = self.get_tensors_info(model_type="optimized").get(inspect_type, None)
        if tensors is None:
            raise ClientErrorException(
                f"Could not get tensor information for {inspect_type} type.",
            )

        if inspect_type == "activation":
            tensors = tensors[0]

        op_tensors: Optional[dict] = tensors.get(op_name, None)
        if op_tensors is None:
            raise ClientErrorException(
                f"Could not get tensor information for {op_name} OP.",
            )

        op_histograms = []
        for tensor_name, tensor_data in op_tensors.items():
            tensor_histograms = []
            if tensor_data.ndim < 2:
                continue
            for tensor_channel_data in tensor_data[0]:
                tensor_histograms.append(
                    {
                        "data": tensor_channel_data.flatten().tolist(),
                    },
                )
            op_histograms.append(
                {
                    "name": f"{tensor_name} {inspect_type} histogram",
                    "histograms": tensor_histograms,
                },
            )

        return op_histograms

    @staticmethod
    def mse_metric_gap(fp32_tensor: Any, dequantize_tensor: Any) -> float:
        """
        Calculate the euclidean distance between fp32 tensor and int8 dequantize tensor.

        Args:
            fp32_tensor (tensor): The FP32 tensor.
            dequantize_tensor (tensor): The INT8 dequantize tensor.
        """
        check_module("numpy")
        import numpy as np

        fp32_max = np.max(fp32_tensor)  # type: ignore
        fp32_min = np.min(fp32_tensor)  # type: ignore
        dequantize_max = np.max(dequantize_tensor)  # type: ignore
        dequantize_min = np.min(dequantize_tensor)  # type: ignore
        fp32_tensor = (fp32_tensor - fp32_min) / (fp32_max - fp32_min)
        dequantize_tensor = (dequantize_tensor - dequantize_min) / (
            dequantize_max - dequantize_min
        )
        diff_tensor = fp32_tensor - dequantize_tensor
        euclidean_dist = np.sum(diff_tensor**2)  # type: ignore
        return euclidean_dist / fp32_tensor.size
