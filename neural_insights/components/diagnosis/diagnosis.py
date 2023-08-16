# -*- coding: utf-8 -*-
# Copyright (c) 2023 Intel Corporation
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
from abc import abstractmethod
from typing import Any, List, Optional

from neural_insights.components.diagnosis.op_details import OpDetails
from neural_insights.components.diagnosis.op_entry import OpEntry
from neural_insights.components.diagnosis.weights_details import WeightsDetails
from neural_insights.components.model.model import Model
from neural_insights.components.workload_manager.workload import Workload
from neural_insights.utils.exceptions import ClientErrorException, InternalException
from neural_insights.utils.logger import log
from neural_insights.utils.utils import check_module


class Diagnosis:
    """Diagnosis class."""

    def __init__(self, workload: Workload):
        """Initialize diagnosis."""
        self.workload_location = workload.workload_location
        self.model_path = workload.model_path

    @abstractmethod
    def model(self) -> Model:
        """Neural Insights model instance."""
        raise NotImplementedError

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
            self.workload_location,
            "inspect_saved",
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
            self.workload_location,
            "inspect_saved",
            "cfg.pkl",
        )
        if not os.path.exists(config_path):
            raise ClientErrorException("Could not find config data for specified optimization.")
        with open(config_path, "rb") as config_pickle:
            config_data = pickle.load(config_pickle)
        return config_data

    def get_op_list(self) -> List[dict]:
        """Get OP list for model."""
        check_module("numpy")
        import numpy as np

        op_list: List[dict] = []

        input_model_tensors: dict = self.get_tensors_info(model_type="input")["activation"][0]
        optimized_model_tensors: dict = self.get_tensors_info(model_type="optimized")[
            "activation"
        ][0]

        minmax_file_path = os.path.join(
            self.workload_location,
            "inspect_saved",
            "activation_min_max.pkl",
        )

        try:
            with open(minmax_file_path, "rb") as min_max_file:
                min_max_data: dict = pickle.load(min_max_file)
        except FileNotFoundError:
            log.debug("Could not find minmax file.")
            common_ops = list(
                set(input_model_tensors.keys()) & set(optimized_model_tensors.keys()))
            min_max_data = dict(zip(common_ops, [{"min": None, "max": None}]*len(common_ops)))
            print(min_max_data)

        for op_name, min_max in min_max_data.items():

            mse = self.calculate_mse(op_name, input_model_tensors, optimized_model_tensors)
            if mse is None or np.isnan(mse):
                continue
            min = min_max.get("min", None)
            max = min_max.get("max", None)

            if min is not None:
                min = float(min)
            if max is not None:
                max = float(max)

            op_entry = OpEntry(op_name, mse, min, max)
            op_list.append(op_entry.serialize())
        return op_list

    def get_weights_details(self, inspect_type: str) -> List[WeightsDetails]:
        """Get weights details for model."""
        weights_details = []

        minmax_file_path = os.path.join(
            self.workload_location,
            "inspect_saved",
            "activation_min_max.pkl",
        )
        with open(minmax_file_path, "rb") as min_max_file:
            min_max_data: dict = pickle.load(min_max_file)

        input_model_tensors: dict = self.get_tensors_info(model_type="input")[inspect_type]
        optimized_model_tensors: dict = self.get_tensors_info(model_type="optimized")[
            inspect_type
        ]
        if inspect_type == "activation":
            input_model_tensors = input_model_tensors[0]
            optimized_model_tensors = optimized_model_tensors[0]
        common_ops = list(set(input_model_tensors.keys()) & set(optimized_model_tensors.keys()))
        for op_name in common_ops:

            input_model_op_tensors = input_model_tensors[op_name]
            optimized_model_op_tensors = optimized_model_tensors[op_name]

            if op_name not in min_max_data.keys():
                continue

            if isinstance(input_model_op_tensors, dict):
                for (input_op_name, input_op_values), (optimized_op_name, optimized_op_values) in\
                        zip(input_model_op_tensors.items(), optimized_model_op_tensors.items()):
                    if input_op_values.ndim != 4 or optimized_op_values.ndim != 4:
                        continue

                    weights_entry = WeightsDetails(
                        input_op_name,
                        input_op_values,
                        optimized_op_values,
                    )
                    weights_details.append(weights_entry)
        return weights_details

    def calculate_mse(
        self,
        op_name: str,
        input_model_tensors: dict,
        optimized_model_tensors: dict,
    ) -> Optional[float]:
        """Calculate MSE for specified tensors."""
        input_model_op_data = input_model_tensors.get(op_name, None)
        optimized_model_op_data = optimized_model_tensors.get(op_name, None)

        if input_model_op_data is None or optimized_model_op_data is None:
            return None

        mse: float = self.mse_metric_gap(
            next(iter(input_model_op_data.values()))[0],
            next(iter(optimized_model_op_data.values()))[0],
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
        for tensor_name, tensor_data_raw in op_tensors.items():
            tensor_histograms = []

            if tensor_data_raw.ndim == 1:
                tensor_data = [tensor_data_raw]
            elif tensor_data_raw.ndim == 2:
                tensor_data = tensor_data_raw[0]
            else:
                tensor_data = tensor_data_raw[0]
            for tensor_channel_data in tensor_data:
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
        """Calculate the euclidean distance between fp32 tensor and int8 dequantize tensor.

        Args:
            fp32_tensor (tensor): The FP32 tensor.
            dequantize_tensor (tensor): The INT8 dequantize tensor.
        """
        check_module("numpy")
        import numpy as np

        # Get input tensor min max
        fp32_max = np.max(fp32_tensor)  # type: ignore
        fp32_min = np.min(fp32_tensor)  # type: ignore

        # Get input dequantize tensor min max
        dequantize_max = np.max(dequantize_tensor)  # type: ignore
        dequantize_min = np.min(dequantize_tensor)  # type: ignore

        # Normalize tensor values
        fp32_tensor = (fp32_tensor - fp32_min) / (fp32_max - fp32_min)
        dequantize_tensor = (dequantize_tensor - dequantize_min) / (
            dequantize_max - dequantize_min
        )

        diff_tensor = fp32_tensor - dequantize_tensor
        euclidean_dist = np.sum(diff_tensor**2)  # type: ignore
        return euclidean_dist / fp32_tensor.size

    def get_weights_data(self, op_name: str, channel_normalization=True) -> list:
        """Get weights data for optimized model."""
        from PIL import Image
        check_module("numpy")
        import numpy as np

        tensors = self.get_tensors_info(model_type="optimized").get("weight", None)
        if tensors is None:
            raise ClientErrorException(
                "Could not get tensor information to display activations.",
            )

        op_tensors: Optional[dict] = tensors.get(op_name, None)
        if op_tensors is None:
            raise ClientErrorException(
                f"Could not get tensor information for {op_name} OP.",
            )

        weights = []
        for tensor_name, tensor_data_raw in op_tensors.items():
            if tensor_data_raw.ndim != 4:
                continue
            tensor_data = tensor_data_raw[0]
            shapes_order = self.model.shape_elements_order  # pylint: disable=no-member
            channels_index = shapes_order.index("channels")
            new_order = [channels_index]
            new_order.extend([x for x in range(len(shapes_order)) if x != channels_index])

            tensor_data = np.transpose(tensor_data, new_order)

            if tensor_data.shape[1] != tensor_data.shape[2]:
                continue

            for tensor in tensor_data:
                if channel_normalization:
                    tensor = 255 * (tensor - np.min(tensor)) / (np.max(tensor) - np.min(tensor))
                img = Image.fromarray(tensor)
                img = img.convert("L")
                img.show()
                weights.append(tensor.tolist())
        return weights
