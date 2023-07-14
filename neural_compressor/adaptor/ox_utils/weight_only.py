#
# -*- coding: utf-8 -*-
#
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
"""WeightOnly for onnxrt adaptor."""

import sys
import os
import math
import copy
import onnx
import logging
import numpy as np
from onnx import onnx_pb as onnx_proto
from neural_compressor.model.model import BaseModel
from neural_compressor.model.onnx_model import ONNXModel
from onnx import numpy_helper, helper

logger = logging.getLogger("neural_compressor")

def qdq_tensor(data, config, ratio=1.):
    """Quant and dequant tensor per group.

    Args:
        data : input weight
        config (dict): quantization config
        ratio (float, optional): percentile of clip. Defaults to 1.0.

    Returns:
        output: qdq weight
    """
    bit = config["bits"]
    scheme = config["scheme"]
    if scheme == "sym":
        maxq = 2 ** (bit - 1) - 1 if bit != 1 else 0
        minq = -2 ** (bit - 1) if bit != 1 else -1
    elif scheme == "asym":
        maxq = 2 ** bit - 1
        minq = 0

    rmin = np.min(data, axis=0, keepdims=True) * ratio
    rmax = np.max(data, axis=0, keepdims=True) * ratio
    if scheme == "sym":
        max_range = np.maximum(np.abs(rmin), np.abs(rmax))
        scale = np.ones(rmax.shape, dtype="float32")
        scale[max_range > 0] = np.array([float(i) / (maxq - minq) for i in \
            (max_range[max_range > 0] * 2.).flatten().tolist()], dtype="float32")
        zero_point = np.zeros(scale.shape)
    else:
        scale = np.ones(rmax.shape, dtype="float32")
        scale[rmin != rmax] = np.array([float(i) / (maxq - minq) for i in \
            (rmax - rmin)[rmin != rmax].flatten().tolist()], dtype="float32")
        zero_point = ((np.zeros(scale.shape) - rmin) / scale).round()

    return scale * (np.clip((data / scale + zero_point).round(), minq, maxq) - zero_point)

def rtn_quantize(model, tune_cfg, ratios={}):
    """Quant the model with round to nearst method.

    Args:
        model (ModelProto or ONNXModel): onnx model
        tune_cfg (dict): quantization config
                For example, 
                tune_cfg={
                    'fc2':
                        {
                            'bits': 4, 
                            'group_size': 32, 
                            'scheme': 'sym',
                            'algorithm': 'RTN'
                        }
                }
        ratios (dict, optional): percentile of clip. Defaults to {}.

    Returns:
        model: fake quantized ONNXModel
    """
    model = model if isinstance(model, BaseModel) else ONNXModel(model) 
    for node in model.nodes():
        if node.name in tune_cfg and tune_cfg[node.name] != "fp32":
            if model.get_initializer(node.input[1]) is None:
                continue
            weight = numpy_helper.to_array(
                        model.get_initializer(node.input[1]),
                        base_dir=os.path.dirname(model.model_path)).copy()
            dtype = weight.dtype
            config = tune_cfg[node.name]["weight"]

            org_w_shape = weight.shape # ic, oc
            group_size = config["group_size"] if config["group_size"] != -1 else org_w_shape[0]

            if org_w_shape[0] % group_size == 0:
                weight = weight.reshape(group_size, -1)
                weight = qdq_tensor(weight, config, ratios.get(node.input[1], 1))
                weight = weight.reshape(org_w_shape)
            else:
                index = org_w_shape[0] // group_size * group_size
                if index != 0:
                    part_weight = weight[:index, :].reshape(group_size, -1)
                    part_weight = qdq_tensor(part_weight, config, ratios.get(node.input[1], 1))
                    weight[:index, :] = part_weight.reshape(index, -1)
                weight[index:, :] = qdq_tensor(weight[index:, :], config, ratios.get(node.input[1], 1))
            model.set_initializer(node.input[1], weight.astype(dtype), raw=True)
    return model

def get_weight_scale(weight, group_size):
    """Get the scale of weight."""
    org_shape = weight.shape
    weight = np.reshape(weight, (group_size, -1)) if group_size != -1 else weight
    scale = np.mean(
            np.reshape(np.abs(weight) / np.max(np.abs(weight), axis=0, keepdims=True), org_shape),
            axis=1)
    return scale

def apply_awq_scale(model, tune_cfg, absorb_pairs, output_dicts):
    """Apply scale for salient weight."""
    best_scales = {}
    new_init_tensors = []
    new_added_mul_nodes = []
    replace_input = []
    updated_nodes = []
 
    for parent, nodes in absorb_pairs.items():
        if any([node.input[0] not in output_dicts for node in nodes]):
            logger.warning("Miss tensors of node {} during AWQ, skip it!".format(node.name))
            continue
        inp = np.concatenate(output_dicts[nodes[0].input[0]], axis=0)
        inp_scale = np.mean(np.reshape(np.abs(inp), (-1, inp[0].shape[-1])), axis=0)
        weight = []
        org_out = []
        config = tune_cfg[nodes[0].name]
        
        # search scale
        best_error = float("inf")
        best_ratio = -1
        best_scale = None
        n_grid = 20

        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid
            loss = 0
            for node in nodes:
                weight = numpy_helper.to_array(model.get_initializer(node.input[1]),
                                                os.path.dirname(model.model_path))
                w_scale = get_weight_scale(weight, config["weight"]["group_size"])
                org_out = np.matmul(inp, weight)
                scales = np.clip(np.power(inp_scale, ratio) / np.power(w_scale, (1 - ratio)), 1e-4, None)
                scales = np.reshape(scales / np.sqrt(np.max(scales) * np.min(scales)), (-1, 1))

                q_weight = qdq_tensor(weight * scales, config["weight"]) / scales
                out = np.matmul(inp, q_weight)
                loss += np.mean(np.power((org_out - out), 2))

            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scale = scales

        for node in nodes:
            if node.name in tune_cfg and tune_cfg[node.name] != "fp32":
                tensor = numpy_helper.to_array(model.get_initializer(node.input[1]))
                new_tensor = tensor * best_scale
                model.set_initializer(node.input[1], new_tensor.astype(tensor.dtype), raw=True)
                output_dicts[node.input[0]] = output_dicts[node.input[0]] / np.reshape(best_scale, (1, -1))

        parent = model.get_node(parent)
        if parent.name in updated_nodes:
            continue

        if parent.op_type in ["LayerNormalization", "BatchNormalization", "InstanceNormalization"] and \
                all([node.name in tune_cfg and tune_cfg[node.name] != "fp32" for node in nodes]):  # pragma: no cover
            for idx in [1, 2]:
                tensor = numpy_helper.to_array(model.get_initializer(parent.input[idx]),
                                                os.path.dirname(model.model_path))
                new_tensor = tensor / np.reshape(best_scale, (1, -1))
                model.set_initializer(parent.input[idx], new_tensor.astype(tensor.dtype), raw=True)
                updated_nodes.append(parent.name)
            output_dicts[parent.output[0]] = output_dicts[parent.output[0]] / np.reshape(best_scale, (1, -1))

        elif parent.op_type in ["SimplifiedLayerNormalization", "MatMul", "Gemm", "Mul"] and \
                not all([model.get_initializer(inp) is None for inp in parent.input]) and \
                all([node.name in tune_cfg and tune_cfg[node.name] != "fp32" for node in nodes]):
            for inp in parent.input:
                if model.get_initializer(inp) is not None:
                    tensor = numpy_helper.to_array(model.get_initializer(inp),
                                                    os.path.dirname(model.model_path))
                    new_tensor = tensor / np.reshape(best_scale, (1, -1))
                    model.set_initializer(inp, new_tensor.astype(tensor.dtype), raw=True)
            updated_nodes.append(parent.name)
            output_dicts[parent.output[0]] = output_dicts[parent.output[0]] / np.reshape(best_scale, (1, -1))

        elif parent.op_type in ["Conv", "FusedConv"] and \
                all([node.name in tune_cfg and tune_cfg[node.name] != "fp32" for node in nodes]):  # pragma: no cover
            tensor = numpy_helper.to_array(model.get_initializer(parent.input[2]),
                                            os.path.dirname(model.model_path))
            new_tensor = tensor / np.reshape(best_scale, (1, -1))
            model.set_initializer(parent.input[2], new_tensor.astype(tensor.dtype), raw=True)
            updated_nodes.append(parent.name)
            output_dicts[parent.output[0]] = output_dicts[parent.output[0]] / np.reshape(best_scale, (1, -1))

        else:  # pragma: no cover
            # insert mul
            q_nodes = [node for node in nodes if node.name in tune_cfg and tune_cfg[node.name] != "fp32"]
            if len(q_nodes) > 0:
                scale_tensor = helper.make_tensor(
                    name=parent.output[0] + "_weight_only_scale",
                    data_type=onnx_proto.TensorProto.FLOAT,
                    dims=best_scale.shape,
                    vals=(1. / best_scale).flatten().tolist())
                new_init_tensors.append(scale_tensor)
                mul_output_name = parent.output[0] + "_weight_only_out"
                mul_node = helper.make_node(
                    "Mul",
                    inputs=[q_nodes[0].input[0], scale_tensor.name],
                    outputs=[mul_output_name],
                    name=q_nodes[0].input[0] + "_weight_only_mul"
                )
                new_added_mul_nodes.append(mul_node)
                for node in q_nodes:
                    replace_input.append([node, node.input[0], mul_node.output[0]])
                updated_nodes.append(parent.name)
                output_dicts[mul_node.output[0]] = output_dicts[mul_node.input[0]] / np.reshape(best_scale, (1, -1))
 
    model.add_nodes(new_added_mul_nodes)
    model.add_initializers(new_init_tensors)
    for node, old_input_name, new_input_name in replace_input:
        model.replace_node_input(node, old_input_name, new_input_name)

    return model, output_dicts

def apply_awq_clip(model, tune_cfg, absorb_pairs, output_dicts):
    """Apply clip for weight by checking mse."""
    ratios = {}
    for parent, nodes in absorb_pairs.items():
        if any([node.input[0] not in output_dicts for node in nodes]):
            logger.warning("Miss tensors of node {} during AWQ, skip it!".format(node.name))
            continue

        inp = np.concatenate(output_dicts[nodes[0].input[0]], axis=0)

        for node in nodes:
            if node.name in tune_cfg and tune_cfg[node.name] != "fp32":
 
                group_size = tune_cfg[node.name]["weight"]["group_size"]
                config = tune_cfg[node.name]
                weight = numpy_helper.to_array(
                            model.get_initializer(node.input[1]),
                            base_dir=os.path.dirname(model.model_path))
                org_w_shape = weight.shape # ic, oc
                org_out = np.matmul(inp, weight) # n_token, oc

                best_error = float("inf")
                best_ratio = 1
                for i_s in range(10):
                    ratio = 1 - i_s / 100
                    q_weight = qdq_tensor(weight, config["weight"], ratio)
                    cur_out = np.matmul(inp, q_weight)
                    loss = np.mean(np.power((org_out - cur_out), 2))
                    is_best = loss < best_error
                    if is_best:
                        best_error = loss
                        best_ratio = ratio
                ratios[node.input[1]] = best_ratio
    model = rtn_quantize(model, tune_cfg, ratios)        
    return model

def awq_quantize(model,
                 tune_cfg,
                 dataloader,
                 n_samples=128,
                 auto_scale=True,
                 mse_range=True,
                 n_blocks=5
                 ):
    """Quant the model with Activation-aware Weight quantization(AWQ) method.

    Args:
        model (ModelProto or ONNXModel): onnx model
        tune_cfg (dict): quantization config
                For example, 
                tune_cfg={
                    'fc2':
                        {
                            'bits': 4, 
                            'group_size': 32, 
                            'scheme': 'sym',
                            'algorithm': 'AWQ'
                        }
                }
        n_samples: calibration sample number.
        auto_scale (bool, optional): whether enable scale for salient weight. Defaults to True.
        mse_range (bool, optional):  whether enable clip for weight by checking mse. Defaults to True.
        n_blocks (int, optional): split model into block number to avoid OOM.

    Returns:
        model: fake quantized ONNXModel
    """
    model = model if isinstance(model, BaseModel) else ONNXModel(model)
    white_nodes = []
    output_dicts = {}

    if mse_range or mse_range:
        import onnxruntime
        from importlib.util import find_spec
        from neural_compressor.adaptor.ox_utils.util import to_numpy
        
        so = onnxruntime.SessionOptions()
        if sys.version_info < (3, 10) and find_spec('onnxruntime_extensions'):  # pragma: no cover
            from onnxruntime_extensions import get_library_path
            so.register_custom_ops_library(get_library_path())
        if model.is_large_model:
            onnx.save_model(model.model,
                            model.model_path + '_augment.onnx',
                            save_as_external_data=True,
                            all_tensors_to_one_file=True,
                            convert_attribute=False)

        session = onnxruntime.InferenceSession(
                    model.model.SerializeToString(),
                    so,
                    providers=["CPUExecutionProvider"]) if not model.is_large_model else \
                  onnxruntime.InferenceSession(
                    model.model_path,
                    so,
                    providers=["CPUExecutionProvider"])
        inputs_names = [i.name for i in session.get_inputs()]
        del session

        org_output = model.output()
        model.remove_tensors_from_outputs(org_output)
        block_num = 0
        absorb_pairs = model.get_absorb_pairs(["MatMul", "Attention"])
        dump_pairs = {}
        inputs = []
        for i, data in enumerate(dataloader):
            if ((i + 1) * dataloader.batch_size) >= n_samples:
                break
            if len(inputs_names) != 1 or isinstance(data[0], dict):
                assert len(data[0]) == len(inputs_names), "Input number mismatch, " \
                        "require {} but get {}".format(len(inputs_names), len(data[0]))
                
            if isinstance(data[0], dict):
                inputs.append(dict([(name, to_numpy(inp_data)) for name, inp_data in data[0].items()]))
            else:
                inputs.append(dict([(name, to_numpy(inp)) for name, inp in zip(inputs_names, data[0])]))
            del dataloader

        num_block = math.ceil(len(absorb_pairs) / n_blocks)
        for idx, parent in absorb_pairs:
            if (idx + 1) % num_block == 0 or (idx + 1) == len(absorb_pairs):
                dump_pairs.update(parent, absorb_pairs[parent])
                output_dicts = {}
                dump_tensor = list(set([i.input[0] for nodes in dump_pairs.values() for i in nodes]))
                model.add_tensors_to_outputs(dump_tensor)

                if model.is_large_model:
                    onnx.save_model(model.model,
                                    model.model_path + '_augment.onnx',
                                    save_as_external_data=True,
                                    all_tensors_to_one_file=True,
                                    convert_attribute=False)

                session = onnxruntime.InferenceSession(
                            model.model.SerializeToString(),
                            so,
                            providers=["CPUExecutionProvider"]) if not model.is_large_model else \
                          onnxruntime.InferenceSession(
                            model.model_path,
                            so,
                            providers=["CPUExecutionProvider"])

                for inp in inputs:
                    for output_idx, output in enumerate(session.run(None, inp)):
                        output_dicts.setdefault(dump_tensor[output_idx], []).append(output)

                model.remove_tensors_from_outputs(dump_tensor)
                if auto_scale:
                    model, output_dicts = apply_awq_scale(model, tune_cfg, dump_pairs, output_dicts)
                if mse_range:
                    model = apply_awq_clip(model, tune_cfg, dump_pairs, output_dicts)
                del output_dicts
                dump_pairs = {}
            else:
                dump_pairs.update(parent, absorb_pairs[parent])

        model.add_tensors_to_outputs(org_output)
    return model
