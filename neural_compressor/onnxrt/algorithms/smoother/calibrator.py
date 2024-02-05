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
"""Calibration for smooth quant."""

import sys
import tempfile
from importlib.util import find_spec
from pathlib import Path
from typing import List

import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper
import onnxruntime

from neural_compressor.common import Logger
from neural_compressor.onnxrt.quantization.calibrate import CalibrationDataReader
from neural_compressor.onnxrt.utils.onnx_model import ONNXModel

logger = Logger().get_logger()

__all__ = ["Calibrator"]


class Calibrator:
    """Dump information for smooth quant."""

    def __init__(
        self,
        model: ONNXModel,
        dataloader: CalibrationDataReader,
        iterations: List[int] = [],
        providers: List[str] = ["CPUExecutionProvider"],
        **kwargs,
    ):
        """Initialize a Calibrator to dump information.

        Args:
            model (ONNXModel): ONNXModel object.
            dataloader (CalibrationDataReader): user implemented object to read in and preprocess calibration dataset.
            iterations (List[int], optional): tensor of which iteration will be collected. Defaults to [].
            providers (List[str], optional): execution provider for onnxruntime. Defaults to ["CPUExecutionProvider"].
        """
        self.model_wrapper = model
        self.dataloader = dataloader
        self.augmented_model = None
        self.iterations = iterations
        self.providers = providers

    def _check_is_group_conv(self, node):
        """Check the op is group wised or not(depthwise conv is excluded,return false).

        Args:
            node: The op node

        Returns:
            Bool: group wised True, otherwise False, depthwise False
        """
        name_to_indices = {}
        for index, i in enumerate(self.model_wrapper.initializer()):
            name_to_indices[i.name] = index

        if node.op_type == "Conv":
            group = 1
            for attr in node.attribute:
                if hasattr(attr, "name"):
                    if attr.name == "group":
                        group = attr.i
                        break
            # currently only normal conv and depthwise conv are supported
            if group > 1:  # group conv, need to check depthwise or not
                weight_name = node.input[1]
                weight_shape = numpy_helper.to_array(
                    self.model_wrapper.initializer()[name_to_indices[weight_name]]
                ).shape
                input_channel = weight_shape[1]
                if input_channel != 1:  # TODO: need to double check
                    return True
        return False

    def _get_input_tensor_of_ops(self, op_types: List[str] = ["MatMul", "Gemm", "Conv", "FusedConv"]):
        """Traverse the graph and get all the data tensors flowing into layers of {op_types}.

        Group conv is excluded.
        # TODO: the tensors could be set/filtered in configuration.

        Args:
            op_types (List[str], optional): The op types whose input tensor will be dumped.
                Defaults to ["MatMul", "Gemm", "Conv", "FusedConv"].

        Returns:
            dict: A dict of dumped tensor to node info
        """
        tensors_to_node = {}
        initializers = {i.name: i for i in self.model_wrapper.initializer()}

        for node in self.model_wrapper.nodes():
            if len(op_types) == 0 or node.op_type in op_types:
                if node.op_type in ["Conv", "FusedConv"] and self._check_is_group_conv(node):
                    continue
                # also need to check whether the layer has weight
                if len(node.input) >= 2 and node.input[1] in initializers.keys():
                    tensors_to_node.setdefault(node.input[0], []).append([node.name, node.input, node.output])
        return tensors_to_node

    def _get_max_per_channel(self, datas, percentile):
        """Get the max values per input channel.

        Args:
            datas: The tensors
            percentile: percentile of calibration to remove outliers

        Returns:
            The max values per input channel
        """
        permute_datas = []
        for data in datas:
            if len(data.shape) == 3:  # TODO: mammul batchsize*seq*inchannel, conv:batchsize*inchannle*f*f
                tensor = np.abs(np.reshape(data, (-1, data.shape[-1])))
                permute_datas.append(tensor)
            elif len(data.shape) == 4:
                tensor = np.swapaxes(data, 1, -1)
                tensor = np.abs(np.reshape(tensor, (-1, tensor.shape[-1])))
                permute_datas.append(tensor)
            elif len(data.shape) == 2:
                permute_datas.append(np.abs(data))
            else:
                assert False, "not supported"
        permute_datas = np.stack(permute_datas, axis=0)
        permute_datas = permute_datas.reshape(-1, permute_datas.shape[-1])
        max_per_channels = np.percentile(permute_datas, percentile, axis=0)
        max_per_channels = max_per_channels.astype(np.single)
        return max_per_channels

    def get_intermediate_outputs(self):
        so = onnxruntime.SessionOptions()
        if sys.version_info < (3, 11) and find_spec("onnxruntime_extensions"):  # pragma: no cover
            from onnxruntime_extensions import get_library_path

            so.register_custom_ops_library(get_library_path())

        providers = self.providers if "TensorrtExecutionProvider" not in self.providers else ["CUDAExecutionProvider"]
        if self.model_wrapper.is_large_model:  # pragma: no cover
            with tempfile.TemporaryDirectory(prefix="ort.calib.") as tmp_dir:
                onnx.save_model(
                    self.model_wrapper.model,
                    Path(tmp_dir).joinpath("augment.onnx").as_posix(),
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    convert_attribute=False,
                )
                session = onnxruntime.InferenceSession(
                    Path(tmp_dir).joinpath("augment.onnx").as_posix(), so, providers=providers
                )
                from onnx.external_data_helper import load_external_data_for_model

                load_external_data_for_model(self.model_wrapper.model, Path(tmp_dir).as_posix())
        else:
            session = onnxruntime.InferenceSession(
                self.model_wrapper.model.SerializeToString(), so, providers=providers
            )
        node_output_names = [output.name for output in session.get_outputs()]
        output_dicts = {}
        input_name_to_nodes = self.model_wrapper.input_name_to_nodes()
        output_name_to_node = self.model_wrapper.output_name_to_node()
        name_to_node = {}
        for data_name in node_output_names:
            node = None
            if data_name in output_name_to_node:
                node = output_name_to_node[data_name]
            elif data_name in input_name_to_nodes:
                node = input_name_to_nodes[data_name][0]
            assert node, "{} is neither an input nor an output of nodes in augmented model.".format(data_name)
            name_to_node[data_name] = node.name

        def _collect_data(ort_inputs):
            for output_idx, output in enumerate(session.run(None, ort_inputs)):
                output_dicts.setdefault(node_output_names[output_idx], []).append(output)

        idx = 0
        while True:
            inputs = self.dataloader.get_next()
            if not inputs:
                break
            if self.iterations != []:
                if idx > max(self.iterations):
                    break
                if idx in self.iterations:
                    _collect_data(inputs)
            else:
                _collect_data(inputs)
            idx += 1
        return output_dicts

    def calib_smooth(self, op_types, percentile: float = 99.999):
        """Smooth model calibration.

        Mainly get the max info per channel of input tensors.

        Args:
            op_types (_type_): The op types whose input tensor will be dumped.
            percentile (float, optional): Percentile of calibration to remove outliers.
                Defaults to 99.999.

        Returns:
            max_vals_per_channel: max values per channel of input tensors
            shape_infos: The shape information of input tensors
        """
        logger.info("Start smooth model calibration.")
        # add the input tensors of {op_types} to outputs of the model
        tensors_to_node = self._get_input_tensor_of_ops(op_types)
        self.model_wrapper.add_tensors_to_outputs(tensors_to_node.keys())
        output_dicts = self.get_intermediate_outputs()

        # remove the input tensors of {op_types} to outputs of the model
        self.model_wrapper.remove_tensors_from_outputs(tensors_to_node.keys())
        max_vals_per_channel = {}
        shape_infos = {}

        for key, val in tensors_to_node.items():
            max_val_per_channel = self._get_max_per_channel(output_dicts[key], percentile=percentile)
            max_vals_per_channel[key] = max_val_per_channel
            shape_infos[key] = output_dicts[key][0].shape
            for item in val:
                shape_infos[item[1][1]] = self.model_wrapper.get_initializer(item[1][1]).dims
        return max_vals_per_channel, shape_infos, tensors_to_node
