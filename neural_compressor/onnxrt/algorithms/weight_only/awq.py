# Copyright (c) 2023 MIT HAN Lab
# This source code is licensed under the MIT license
#
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


import copy
import os
from pathlib import Path
from typing import List, Union

import numpy as np
import onnx
import onnxruntime as ort
from packaging.version import Version

from neural_compressor.common import Logger
from neural_compressor.onnxrt.algorithms.weight_only.rtn import rtn_quantize
from neural_compressor.onnxrt.algorithms.weight_only.utility import pad_tensor, prepare_inputs, qdq_tensor
from neural_compressor.onnxrt.quantization.calibrate import CalibrationDataReader
from neural_compressor.onnxrt.quantization.config import AWQConfig
from neural_compressor.onnxrt.utils.onnx_model import ONNXModel
from neural_compressor.onnxrt.utils.utility import ONNXRT116_VERSION, ONNXRT1161_VERSION, dtype_mapping

logger = Logger().get_logger()

__all__ = ["apply_awq_on_model", "awq_quantize"]


def _get_weight_scale(weight, group_size):
    """Get the scale of weight."""
    org_shape = weight.shape
    weight = np.reshape(weight, (-1, group_size)) if group_size != -1 else weight
    scale = np.mean(np.reshape(np.abs(weight) / np.max(np.abs(weight), axis=1, keepdims=True), org_shape), axis=0)
    return scale


def _apply_awq_scale(model, weight_config, absorb_pairs, output_dicts, num_bits, group_size, scheme):
    """Apply scale for salient weight."""
    best_scales = {}
    new_init_tensors = []
    new_added_mul_nodes = []
    replace_input = []
    updated_nodes = []
    base_dir = os.path.dirname(model.model_path) if model.model_path is not None else ""

    for parent, nodes in absorb_pairs.items():
        if any([node.input[0] not in output_dicts for node in nodes]):
            logger.warning(
                "Miss input tensors of nodes {} during AWQ, skip it!".format(
                    ", ".join([node.name for node in nodes if node.input[0] not in output_dicts])
                )
            )
            continue
        inp = np.concatenate(output_dicts[nodes[0].input[0]], axis=0)
        inp_scale = np.mean(np.reshape(np.abs(inp), (-1, inp[0].shape[-1])), axis=0)
        dtype = None
        weight = []
        org_out = []
        for node in nodes:
            if node.name in weight_config and weight_config.get(node.name, "fp32") != "fp32":
                num_bits = weight_config[node.name]["bits"]
                group_size = weight_config[node.name]["group_size"]
                scheme = weight_config[node.name]["scheme"]
                break

        # search scale
        best_error = float("inf")
        best_ratio = -1
        best_scale = None
        n_grid = 20

        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid
            loss = 0
            for node in nodes:
                if weight_config.get(node.name, {}) == "fp32":
                    continue

                weight = onnx.numpy_helper.to_array(model.get_initializer(node.input[1]), base_dir)
                if len(weight.shape) != 2:
                    continue

                org_out = np.matmul(inp, weight)
                org_w_shape = weight.shape
                group_size = group_size if group_size != -1 else org_w_shape[0]

                w_scale = _get_weight_scale(weight.T, weight.shape[0])
                scales = np.clip(np.power(inp_scale, ratio) / np.power(w_scale, (1 - ratio)), 1e-4, None)
                scales = scales / np.sqrt(np.max(scales) * np.min(scales))
                weight = weight.T * scales
                weight = pad_tensor(weight, group_size, (org_w_shape[0] + group_size - 1) // group_size).T

                if (Version(ort.__version__) > ONNXRT1161_VERSION and num_bits == 4) or (
                    Version(ort.__version__) >= ONNXRT116_VERSION and num_bits == 4 and group_size == 32
                ):  # pragma: no cover
                    # MatMulFpQ4 support 4 bits and 32 group_size with ort 1.16.0 and 1.16.1 versions
                    # MatMulNBits supports 4 bits and 2^n group_size with ort > 1.16.1
                    q_weight = qdq_tensor(weight, num_bits, group_size, scheme, "uint") / np.expand_dims(
                        scales, axis=-1
                    )
                else:
                    q_weight = qdq_tensor(weight, num_bits, group_size, scheme, "int") / np.expand_dims(scales, axis=-1)

                q_weight = np.reshape(q_weight, (org_w_shape[1], -1))[:, : org_w_shape[0]]
                out = np.matmul(inp, q_weight.T)
                loss += np.mean(np.power((org_out - out), 2))

            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scale = scales

        for node in nodes:
            weight_config.setdefault(node.name, {}).update({"bits": num_bits})
            weight_config.setdefault(node.name, {}).update({"group_size": group_size})
            weight_config.setdefault(node.name, {}).update({"scheme": scheme})

            init_share_num = model.get_initializer_share_num(node.input[1])
            weight_tensor = model.get_initializer(node.input[1])
            tensor = onnx.numpy_helper.to_array(weight_tensor, base_dir)
            dtype = tensor.dtype
            tensor = tensor.T * best_scale
            tensor = (tensor.T).astype(dtype)

            new_tensor = onnx.helper.make_tensor(
                name=node.input[1] + "_scaled",
                data_type=dtype_mapping[str(dtype)],
                dims=tensor.shape,
                vals=tensor.tobytes(),
                raw=True,
            )
            model.add_initializer(new_tensor)
            node.input[1] = new_tensor.name

            if init_share_num == 1:
                model.remove_initializer(weight_tensor)

        parent = model.get_node(parent)
        if parent.name in updated_nodes:
            continue

        if parent.op_type in ["LayerNormalization", "BatchNormalization", "InstanceNormalization"] and len(
            model.input_name_to_nodes()[nodes[0].input[0]]
        ) == len(nodes):
            for idx in [1, 2]:
                tensor = onnx.numpy_helper.to_array(model.get_initializer(parent.input[idx]), base_dir)
                dtype = tensor.dtype
                new_tensor = tensor / np.reshape(best_scale, (1, -1))
                model.set_initializer(parent.input[idx], new_tensor.astype(dtype), raw=True)
                updated_nodes.append(parent.name)
            output_dicts[parent.output[0]] = output_dicts[parent.output[0]] / np.reshape(best_scale, (1, -1))

        elif (
            parent.op_type in ["SimplifiedLayerNormalization", "MatMul", "Gemm", "Mul"]
            and not all([model.get_initializer(inp) is None for inp in parent.input])
            and len(model.input_name_to_nodes()[nodes[0].input[0]]) == len(nodes)
        ):  # pragma: no cover
            for inp in parent.input:
                if model.get_initializer(inp) is not None:
                    tensor = onnx.numpy_helper.to_array(model.get_initializer(inp), base_dir)
                    dtype = tensor.dtype
                    new_tensor = tensor / np.reshape(best_scale, (1, -1))
                    model.set_initializer(inp, new_tensor.astype(dtype), raw=True)
            updated_nodes.append(parent.name)
            output_dicts[parent.output[0]] = output_dicts[parent.output[0]] / np.reshape(best_scale, (1, -1))

        elif parent.op_type in ["Conv", "FusedConv"] and len(model.input_name_to_nodes()[nodes[0].input[0]]) == len(
            nodes
        ):  # pragma: no cover
            tensor = onnx.numpy_helper.to_array(model.get_initializer(parent.input[2]), base_dir)
            dtype = tensor.dtype
            new_tensor = tensor / np.reshape(best_scale, (1, -1))
            model.set_initializer(parent.input[2], new_tensor.astype(dtype), raw=True)
            updated_nodes.append(parent.name)
            output_dicts[parent.output[0]] = output_dicts[parent.output[0]] / np.reshape(best_scale, (1, -1))

        else:  # pragma: no cover
            # insert mul
            scale_tensor = onnx.helper.make_tensor(
                name=parent.output[0] + "_weight_only_scale",
                data_type=dtype_mapping[str(dtype)],
                dims=best_scale.shape,
                vals=(1.0 / best_scale).flatten().tolist(),
            )
            new_init_tensors.append(scale_tensor)
            mul_output_name = parent.output[0] + "_weight_only_out"
            mul_node = onnx.helper.make_node(
                "Mul",
                inputs=[nodes[0].input[0], scale_tensor.name],
                outputs=[mul_output_name],
                name=nodes[0].input[0] + "_weight_only_mul",
            )
            new_added_mul_nodes.append(mul_node)
            for node in nodes:
                replace_input.append([node, node.input[0], mul_node.output[0]])
            updated_nodes.append(parent.name)
            output_dicts[mul_node.output[0]] = output_dicts[mul_node.input[0]] / np.reshape(best_scale, (1, -1))

    model.add_nodes(new_added_mul_nodes)
    model.add_initializers(new_init_tensors)
    for node, old_input_name, new_input_name in replace_input:
        model.replace_node_input(node, old_input_name, new_input_name)

    return model, output_dicts


def _apply_awq_clip(model, weight_config, absorb_pairs, output_dicts, num_bits, group_size, scheme):
    """Apply clip for weight by checking mse."""
    base_dir = os.path.dirname(model.model_path) if model.model_path is not None else ""
    ratios = {}
    for parent, nodes in absorb_pairs.items():
        if any([node.input[0] not in output_dicts for node in nodes]):
            logger.warning(
                "Miss input tensors of nodes {} during AWQ, skip it!".format(
                    ", ".join([node.name for node in nodes if node.input[0] not in output_dicts])
                )
            )
            continue

        inp = np.concatenate(output_dicts[nodes[0].input[0]], axis=0)

        for node in nodes:
            if node.name in weight_config:
                num_bits = weight_config[node.name]["bits"]
                group_size = weight_config[node.name]["group_size"]
                scheme = weight_config[node.name]["scheme"]

            org_weight = onnx.numpy_helper.to_array(model.get_initializer(node.input[1]), base_dir=base_dir)
            org_w_shape = org_weight.shape  # ic, oc
            group_size = group_size if group_size != -1 else org_w_shape[0]
            org_out = np.matmul(inp, org_weight)  # n_token, oc

            k_blocks = (org_w_shape[0] - 1) // group_size + 1
            org_weight = pad_tensor(org_weight, group_size, k_blocks)

            org_weight = np.transpose(org_weight)

            best_error = float("inf")
            best_ratio = 1
            for i_s in range(10):
                ratio = 1 - i_s / 100
                weight = copy.deepcopy(org_weight)
                if (Version(ort.__version__) > ONNXRT1161_VERSION and num_bits == 4) or (
                    Version(ort.__version__) >= ONNXRT116_VERSION and num_bits == 4 and group_size == 32
                ):  # pragma: no cover
                    # MatMulFpQ4 support 4 bits and 32 group_size with ort 1.16.0 and 1.16.1 versions
                    # MatMulNBits supports 4 bits and 2^n group_size with ort > 1.16.1
                    weight = qdq_tensor(weight, num_bits, group_size, scheme, "uint", ratios.get(node.input[1], 1))
                else:
                    weight = qdq_tensor(weight, num_bits, group_size, scheme, "int", ratios.get(node.input[1], 1))
                weight = np.reshape(weight, (org_w_shape[1], -1))[:, : org_w_shape[0]]
                cur_out = np.matmul(inp, weight.T)
                loss = np.mean(np.power((org_out - cur_out), 2))
                is_best = loss < best_error
                if is_best:
                    best_error = loss
                    best_ratio = ratio
            ratios[node.input[1]] = best_ratio
    return ratios


def awq_quantize(
    model: Union[onnx.ModelProto, ONNXModel, Path, str],
    data_reader: CalibrationDataReader,
    weight_config: dict = {},
    num_bits: int = 4,
    group_size: int = 32,
    scheme: str = "asym",
    enable_auto_scale: bool = True,
    enable_mse_search: bool = True,
    accuracy_level: int = 0,
    providers: List[str] = ["CPUExecutionProvider"],
) -> onnx.ModelProto:
    """Quant the model with Activation-aware Weight quantization(AWQ) method.

    Args:
        model (Union[onnx.ModelProto, ONNXModel, Path, str]): onnx model.
        data_reader (CalibrationDataReader): data_reader for calibration.
        weight_config (dict, optional): quantization config
            For example,
            weight_config = {
                '(fc2, "MatMul")':
                {
                    'weight_dtype': 'int',
                    'weight_bits': 4,
                    'weight_group_size': 32,
                    'weight_sym': True,
                    'accuracy_level': 0
                }
            }. Defaults to {}.
        num_bits (int, optional): number of bits used to represent weights. Defaults to 4.
        group_size (int, optional): size of weight groups. Defaults to 32.
        scheme (str, optional): indicates whether weights are symmetric. Defaults to "asym".
        enable_auto_scale (bool, optional): whether to search for best scales based on activation
            distribution. Defaults to True.
        enable_mse_search (bool, optional): whether to search for the best clip range from range
            [0.91, 1.0, 0.01]. Defaults to True.
        accuracy_level (int, optional): accuracy level. Support 0 (unset),
            1(fp32 compute type of jblas kernel), 2 (fp16 compute type of jblas kernel),
            3 (bf16 compute type of jblas kernel), 4 (int8 compute type of jblas kernel). Defaults to 0.
        providers (list, optional): providers to use. Defaults to ["CPUExecutionProvider"].

    Returns:
        onnx.ModelProto: quantized onnx model.
    """
    if not isinstance(model, ONNXModel):
        model = ONNXModel(model)
    output_dicts = {}
    full_ratio = {}

    if enable_mse_search:
        inputs, so = prepare_inputs(model, data_reader, providers)
        del data_reader

        org_output = copy.deepcopy(model.model.graph.output)
        model.remove_tensors_from_outputs([i.name for i in org_output])

        output_names = []
        for node in model.nodes():
            # check op_type of node is MatMul
            # check dim 1 of input is weight tensor
            # check weight_type is not "fp32"
            if (
                node.op_type in ["MatMul"]
                and model.get_initializer(node.input[1]) is not None
                and weight_config.get((node.name, node.op_type), {}).get("weight_dtype", "fp32") != "fp32"
            ):
                output_names.append(node.input[0])
        output_names = list(set(output_names))
        model.add_tensors_to_outputs(output_names)
        if model.is_large_model:
            onnx.save_model(
                model.model,
                model.model_path + "_augment.onnx",
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                convert_attribute=False,
            )

        session = (
            ort.InferenceSession(model.model.SerializeToString(), so, providers=providers)
            if not model.is_large_model
            else ort.InferenceSession(model.model_path + "_augment.onnx", so, providers=providers)
        )

        for input_name in output_names:
            parent = model.output_name_to_node()[input_name]
            dump_pairs = {parent.name: []}

            for node in model.input_name_to_nodes()[input_name]:
                # check op_type of node is MatMul
                # check dim 1 of input is weight tensor
                # check weight_type is not "fp32"
                if (
                    node.op_type in ["MatMul"]
                    and model.get_initializer(node.input[1]) is not None
                    and weight_config.get((node.name, node.op_type), {}).get("weight_dtype", "fp32") != "fp32"
                ):
                    dump_pairs[parent.name].append(model.get_node(node.name))

            if len(dump_pairs[parent.name]) == 0:
                continue

            output_dicts = {}
            for inp in inputs:
                output = session.run([input_name], inp)
                output_dicts.setdefault(input_name, []).append(output)

            if enable_auto_scale:
                model, output_dicts = _apply_awq_scale(
                    model,
                    weight_config,
                    dump_pairs,
                    output_dicts,
                    num_bits,
                    group_size,
                    scheme,
                )
            if enable_mse_search:
                ratios = _apply_awq_clip(
                    model,
                    weight_config,
                    dump_pairs,
                    output_dicts,
                    num_bits,
                    group_size,
                    scheme,
                )
            del output_dicts
            del dump_pairs
            full_ratio.update(ratios)

        model.remove_tensors_from_outputs(output_names)
        model.model.graph.output.MergeFrom(org_output)
    model = rtn_quantize(model, weight_config, num_bits, group_size, scheme, full_ratio, accuracy_level, providers)
    return model


def apply_awq_on_model(
    model: Union[onnx.ModelProto, ONNXModel, Path, str],
    quant_config: dict,
    calibration_data_reader: CalibrationDataReader,
) -> onnx.ModelProto:
    """Apply Activation-aware Weight quantization(AWQ) on onnx model.

    Args:
        model (Union[onnx.ModelProto, ONNXModel, Path, str]): nnx model.
        quant_config (dict): quantization config.
        calibration_data_reader (CalibrationDataReader): data_reader for calibration.

    Returns:
        onnx.ModelProto: quantized onnx model.
    """
    # set model params
    kwargs = {}
    kwargs = {key: quant_config.pop(key) for key in AWQConfig.model_params_list if key in quant_config}

    # change op config to dict type
    for op_name_type, op_config in quant_config.items():
        if isinstance(op_config, AWQConfig):
            quant_config[op_name_type] = op_config.to_dict()

    return awq_quantize(model, data_reader=calibration_data_reader, weight_config=quant_config, **kwargs)
