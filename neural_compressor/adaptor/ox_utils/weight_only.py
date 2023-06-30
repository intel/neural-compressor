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

import os
import copy
import onnx
import logging
import numpy as np
from onnx import onnx_pb as onnx_proto
from neural_compressor.model.model import BaseModel
from neural_compressor.model.onnx_model import ONNXModel
from neural_compressor.adaptor.ox_utils.util import find_by_name, \
                    quantize_data, _get_qrange_for_qType, is_B_transposed
from onnx import numpy_helper, helper

logger = logging.getLogger("neural_compressor")

int8_values = list([range(-128, 128, 1)])
int4_values = list([range(-8, 8, 1)])
int3_values = list([range(-4, 4, 1)])

values = {
    8: int8_values,
    4: int4_values,
    3: int3_values
}

def rescale_tensor(data, config):
    bit = config["weight"]["bits"][0]
    scheme = config["weight"]["scheme"][0]
    if scheme == "sym":
        maxq = 2 ** (bit - 1) - 1 if bit != 1 else 0
        minq = -2 ** (bit - 1) if bit != 1 else -1
    elif scheme == "asym":
        maxq = 2 ** bit - 1
        minq = 0

    rmin = np.min(data, axis=-1, keepdims=True)
    rmax = np.max(data, axis=-1, keepdims=True)
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

    return np.clip((data / scale + zero_point).round(), minq, maxq), scale, zero_point


def rtn_quantize(model, tune_cfg):
    model = model if isinstance(model, BaseModel) else ONNXModel(model) 
    for node in model.nodes():
        if node.name in tune_cfg and tune_cfg[node.name]["weight"]["dtype"] != "fp32" and \
                                                tune_cfg[node.name]["bits"] in values:
            if model.get_initializer(node.input[1]) is None:
                continue
            weight = numpy_helper.to_array(
                        model.get_initializer(node.input[1]),
                        base_dir=os.path.dirname(model.model_path) if model.model_path is not None else "").copy()

            config = tune_cfg[node.name]["weight"]

            org_w_shape = weight.shape

            if org_w_shape[1] % config["weight"]["group_size"][0] == 0:
                weight = weight.reshape(-1, config["weight"]["group_size"][0])
                weight, scale, zp = rescale_tensor(weight, config)
                pos = np.abs(np.subtract.outer(values[config["bits"]], weight)).argmin(axis=1)
                weight = np.array(values[config["bits"]], dtype="float32")[0][pos]
                weight = scale * (weight - zp)
                weight = weight.reshape(org_w_shape)
            else:
                index = org_w_shape[1] // config["weight"]["group_size"][0] * config["weight"]["group_size"][0]
                if index != 0:
                    part_weight = weight[:, :index].reshape(-1, config["weight"]["group_size"][0])
                    part_weight, scale, zp = rescale_tensor(part_weight, config)
                    pos = np.abs(np.subtract.outer(values[config["bits"]], part_weight)).argmin(axis=1)
                    part_weight = np.array(values[config["bits"]], dtype="float32")[0][pos]
                    part_weight = scale * (part_weight - zp)
                    weight[:, :index] = part_weight.reshape(-1, index)
                weight[:, index:], scale, zp = rescale_tensor(weight[:, index:], config)
                pos = np.abs(np.subtract.outer(values[config["bits"]], weight[:, index:])).argmin(axis=1)
                weight[:, index:] = np.array(values[config["bits"]], dtype="float32")[0][pos]
                weight[:, index:] = scale * (weight[:, index:] - zp)

            model.set_initializer(node.input[1], weight)
    return model

def get_weight_scale(weight, group_size):
    org_shape = weight.shape
    weight = np.reshape(weight, (-1, group_size))
    scale = np.mean(np.reshape(np.abs(weight) / np.max(np.abs(weight), axis=1, keepdims=True), org_shape), axis=0)
    return scale

def apply_awq_scale(model, tune_cfg, attentions, qkvs, output_dicts):
    best_scales = {}

    for qkv in qkvs:
        if not all([i in tune_cfg and tune_cfg[i]["weight"]["dtype"] != "fp32" for i in qkv]):
            continue
        config = tune_cfg[qkv[0]]
        q = model.get_node(qkv[0])
        k = model.get_node(qkv[1])
        v = model.get_node(qkv[2])

        q_weight = numpy_helper.to_array(
                        model.get_initializer(q.input[1]),
                        base_dir=os.path.dirname(model.model_path) if model.model_path is not None else "")
        k_weight = numpy_helper.to_array(
                        model.get_initializer(k.input[1]),
                        base_dir=os.path.dirname(model.model_path) if model.model_path is not None else "")
        v_weight = numpy_helper.to_array(
                        model.get_initializer(v.input[1]),
                        base_dir=os.path.dirname(model.model_path) if model.model_path is not None else "")

        weights = np.concatenate((q_weight, k_weight, v_weight), axis=0)
        w_shape = q_weight.shape
        inp = output_dicts[q.input[0]]
        org_out = np.concatenate((output_dicts[q.output[0]][0],
                                  output_dicts[k.output[0]][0],
                                  output_dicts[v.output[0]][0]), axis=0)
        w_scale = get_weight_scale(weights, config["weight"]["group_size"][0])
        inp_scale = np.mean(np.reshape(np.abs(inp[0]), (-1, inp[0].shape[-1])), axis=0)
 
        # search scale
        best_error = float("inf")
        best_ratio = -1
        best_scale = None
        n_grid = 20

        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid
            scales = np.clip(np.power(inp_scale, ratio) / np.power(w_scale, (1 - ratio)), 1e-4, None)
            scales = scales / np.sqrt(np.max(scales) * np.min(scales))
            new_weights = weights * scales
            q_weights, scale, zp = rescale_tensor(new_weights, config)
            q_weights = scale * (q_weights - zp)
            out = np.concatenate((np.matmul(inp[0], q_weights[:w_shape[0], :]),
                                  np.matmul(inp[0], q_weights[w_shape[0]:w_shape[0]*2, :]),
                                  np.matmul(inp[0], q_weights[w_shape[0]*2:, :])), axis=0)
            loss = np.mean(np.power((org_out - out), 2))

            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scale = scales
        best_scales[q.name] = best_scale
        best_scales[k.name] = best_scale
        best_scales[v.name] = best_scale

        for node in [q, k, v]:
            for inp in node.input:
                if model.get_initializer(inp) is not None:
                    tensor = model.get_initializer(inp)
                    new_tensor = numpy_helper.to_array(tensor, os.path.dirname(model.model_path)) * best_scale if \
                        model.model_path is not None else numpy_helper.to_array(tensor) * best_scale
                    model.set_initializer(inp, new_tensor)
            output_dicts[node.input[0]] = output_dicts[node.input[0]] / best_scale

    for node_name in attentions:
        if node_name in tune_cfg and tune_cfg[node_name]["weight"]["dtype"] != "fp32":
            node = model.get_node(node_name)
            config = tune_cfg[node_name]
 
            if node.input[0] not in output_dicts or node.output[0] not in output_dicts:
                logger.warning("Miss tensors of node {} during AWQ, skip it!".format(node.name))
                
            att_weight = numpy_helper.to_array(
                        model.get_initializer(node.input[1]),
                        base_dir=os.path.dirname(model.model_path) if model.model_path is not None else "")

            width = int(att_weight.shape[-1] / 3)
            height = att_weight.shape[0]
            weight = np.concatenate((att_weight[:, :width], att_weight[:, width: 2*width], att_weight[:, 2*width:]), axis=0)
            inp = output_dicts[node.input[0]]
            org_out = np.matmul(inp[0], att_weight)
            w_scale = get_weight_scale(weight, config["weight"]["group_size"][0])
            inp_scale = np.mean(np.reshape(np.abs(inp[0]), (-1, inp[0].shape[-1])), axis=0)
            
            # search scale
            best_error = float("inf")
            best_ratio = -1
            best_scale = None
            n_grid = 20

            for ratio in range(n_grid):
                ratio = ratio * 1 / n_grid
                scales = np.clip(np.power(inp_scale, ratio) / np.power(w_scale, (1 - ratio)), 1e-4, None)
                scales = scales / np.sqrt(np.max(scales) * np.min(scales))
                new_weight = weight * scales
                q_weight, scale, zp = rescale_tensor(new_weight, config)
                q_weight = scale * (q_weight - zp)
                q_weight = np.concatenate((q_weight[:height, :],
                                           q_weight[height:2*height, :],
                                           q_weight[2*height:, :]), axis=1)
                out = np.matmul(inp[0], q_weight)
                loss = np.mean(np.power((org_out - out), 2))

                is_best = loss < best_error
                if is_best:
                    best_error = loss
                    best_ratio = ratio
                    best_scale = scales

            for inp in node.input:
                if model.get_initializer(inp) is not None:
                    tensor = model.get_initializer(inp)
                    new_tensor = \
                        numpy_helper.to_array(tensor, os.path.dirname(model.model_path)) * np.tile(best_scale, 3) if \
                        model.model_path is not None else numpy_helper.to_array(tensor) * np.tile(best_scale, 3)
                    model.set_initializer(inp, new_tensor)
            output_dicts[node.input[0]] = output_dicts[node.input[0]] / best_scale
            best_scales[node.name] = best_scale

    new_init_tensors = []
    new_added_mul_nodes = []
    replace_input = []
    updated_nodes = []
    for k, scale in best_scales.items():
        node = model.get_node(k)
        parent = model.get_parent(node, 0)
        if parent is None or parent.name in updated_nodes:
            continue

        if parent.op_type in ["LayerNormalization", "BatchNormalization", "InstanceNormalization"]:
            for idx in [1, 2]:
                tensor = model.get_initializer(node.input[idx])
                new_tensor = numpy_helper.to_array(tensor, os.path.dirname(model.model_path)) / scale if \
                    model.model_path is not None else numpy_helper.to_array(tensor) / scale
                model.set_initializer(node.input[idx], new_tensor)
                updated_nodes.append(parent.name)

        elif parent.op_type in ["SimplifiedLayerNormalization", "MatMul", "Gemm", "Mul"] and \
            not all([model.get_initializer(inp) is None for inp in node.input]):
            for inp in node.input:
                if model.get_initializer(inp) is not None:
                    tensor = model.get_initializer(inp)
                    new_tensor = numpy_helper.to_array(tensor, os.path.dirname(model.model_path)) / scale if \
                        model.model_path is not None else numpy_helper.to_array(tensor) / scale
                    model.set_initializer(inp, new_tensor)
            updated_nodes.append(parent.name)

        elif parent.op_type in ["Conv", "FusedConv"]:
            tensor = model.get_initializer(node.input[2])
            new_tensor = numpy_helper.to_array(tensor, os.path.dirname(model.model_path)) / scale if \
                model.model_path is not None else numpy_helper.to_array(tensor) / scale
            model.set_initializer(node.input[2], new_tensor)
            updated_nodes.append(parent.name)

        else:
            # insert mul
            scale_tensor = helper.make_tensor(
                name=parent.output[0] + "_weight_only_scale",
                data_type=onnx_proto.TensorProto.FLOAT,
                dims=scale.shape,
                vals=scale.flatten().tolist())
            new_init_tensors.append(scale_tensor)
            mul_output_name = parent.output[0] + "_weight_only_out"
            mul_node = helper.make_node(
                "Mul",
                inputs=[node.input[0], scale_tensor.name],
                outputs=[mul_output_name],
                name=node.input[0] + "_weight_only_mul"
            )
            new_added_mul_nodes.append(mul_node)
            replace_input.append([node, node.input[0], mul_node.output[0]])
            updated_nodes.append(parent.name)
 
    model.add_nodes(new_added_mul_nodes)
    model.add_initializers(new_init_tensors)
    for node, old_input_name, new_input_name in replace_input:
        model.replace_node_input(node, old_input_name, new_input_name)

    return model, output_dicts

def apply_awq_clip(model, tune_cfg, attentions, qkvs, output_dicts):
    for qkv in qkvs:
        if not all([i in tune_cfg and tune_cfg[i]["weight"]["dtype"] != "fp32" for i in qkv]):
            continue
        config = tune_cfg[qkv[0]] 
        group_size = config["weight"]["group_size"][0]

        q = model.get_node(qkv[0])
        k = model.get_node(qkv[1])
        v = model.get_node(qkv[2])

        q_weight = numpy_helper.to_array(
                        model.get_initializer(q.input[1]),
                        base_dir=os.path.dirname(model.model_path) if model.model_path is not None else "")
        k_weight = numpy_helper.to_array(
                        model.get_initializer(k.input[1]),
                        base_dir=os.path.dirname(model.model_path) if model.model_path is not None else "")
        v_weight = numpy_helper.to_array(
                        model.get_initializer(v.input[1]),
                        base_dir=os.path.dirname(model.model_path) if model.model_path is not None else "")

        weight = np.concatenate((q_weight, k_weight, v_weight), axis=0)
        org_w_shape = q_weight.shape
        inp = np.concatenate(output_dicts[q.input[0]][0], axis=0)
 
        org_out = np.concatenate((output_dicts[q.output[0]][0],
                                  output_dicts[k.output[0]][0],
                                  output_dicts[v.output[0]][0]), axis=0)

        inp = inp.reshape(1, inp.shape[0], -1, group_size) # 1, n_token, n_group, goup_size
        weight = weight.reshape(weight.shape[0], 1, -1, group_size) # co, 1, n_group, group_size
        org_out = np.sum(inp * weight, axis=-1) # co, n_token, 1
        org_max_val = np.max(np.abs(weight), axis=-1, keepdims=True)  # co, 1, n_group, 1
        min_errs = np.ones(org_max_val.shape) * 1e9  # co, 1, n_group, 1
        best_max_val = np.max(np.abs(weight), axis=-1, keepdims=True)

        for i_s in range(10):
            max_val = org_max_val * (1 - i_s / 20)
            min_val = - max_val
            cur_w = np.clip(weight, min_val, max_val)
            q_weight, scale, zp = rescale_tensor(weight, config)
            q_weight = scale * (q_weight - zp)
            cur_out = np.sum(inp * q_weight, axis=-1)
            err = np.reshape(np.mean(np.power(cur_out - org_out, 2), axis=1), min_errs.shape)
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]

        best_max_val = best_max_val.squeeze(1)
        weight = weight.reshape(*best_max_val.shape[:2], -1)
        weight = np.clip(weight, -best_max_val, best_max_val)
        model.set_initializer(q.input[1], weight[:org_w_shape[1], :].reshape(org_w_shape))
        model.set_initializer(k.input[1], weight[org_w_shape[1]:2*org_w_shape[1], :].reshape(org_w_shape))
        model.set_initializer(v.input[1], weight[2*org_w_shape[1]:, :].reshape(org_w_shape))
 
    for node_name in attentions:
        if node_name in tune_cfg and tune_cfg[node_name]["weight"]["dtype"] != "fp32":
            node = model.get_node(node_name)
 
            if node.input[0] not in output_dicts or node.output[0] not in output_dicts:
                logger.warning("Miss tensors of node {} during AWQ, skip it!".format(node.name))
                
            att_weight = numpy_helper.to_array(
                        model.get_initializer(node.input[1]),
                        base_dir=os.path.dirname(model.model_path) if model.model_path is not None else "")

            org_w_shape = att_weight.shape
            width = int(att_weight.shape[-1] / 3)
            weight = np.concatenate((att_weight[:, :width],
                                     att_weight[:, width: 2*width],
                                     att_weight[:, 2*width:]), axis=0)
            inp = np.concatenate(output_dicts[node.input[0]][0], axis=0)
            org_out = output_dicts[node.output[0]]

            inp = inp.reshape(1, inp.shape[0], -1, group_size) # 1, n_token, n_group, goup_size
            weight = weight.reshape(weight.shape[0], 1, -1, group_size) # co, 1, n_group, group_size
            org_out = np.sum(inp * weight, axis=-1) # co, n_token, 1
            org_max_val = np.max(np.abs(weight), axis=-1, keepdims=True)  # co, 1, n_group, 1
            min_errs = np.ones(org_max_val.shape) * 1e9  # co, 1, n_group, 1
            best_max_val = np.max(np.abs(weight), axis=-1, keepdims=True)

            for i_s in range(10):
                max_val = org_max_val * (1 - i_s / 20)
                min_val = - max_val
                cur_w = np.clip(weight, min_val, max_val)
                q_weight, scale, zp = rescale_tensor(weight, config)
                q_weight = scale * (q_weight - zp)
                cur_out = np.sum(inp * q_weight, axis=-1)
                err = np.reshape(np.mean(np.power(cur_out - org_out, 2), axis=1), min_errs.shape)
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]

            best_max_val = best_max_val.squeeze(1)
            weight = weight.reshape(*best_max_val.shape[:2], -1)
            weight = np.clip(weight, -best_max_val, best_max_val)
            model.set_initializer(node.input[1], weight.reshape(org_w_shape))
            
    return model

def awq_quantize(model, tune_cfg, dataloader):
    from neural_compressor.adaptor.ox_utils.calibration import ONNXRTAugment

    model = model if isinstance(model, BaseModel) else ONNXModel(model)

    attentions = [i.name for i in model.nodes() if i.op_type == "Attention"]
    qkvs = model.find_qkv_in_attention(find_all=True)

    if len(qkvs) == 0 and len(attentions) == 0:
        return model

    white_nodes = []
    for qkv in qkvs:
        white_nodes.extend([i for i in qkv if i in tune_cfg])
    white_nodes.extend([i for i in attentions if i in tune_cfg])

    augment = ONNXRTAugment(model,
                            dataloader,
                            [],
                            white_nodes=white_nodes,
                            iterations=list(range(0, 0)))

    augment.augment_graph(activation_only=False, weight_only=False)

    _, output_dicts = augment.get_intermediate_outputs()
    model, output_dicts = apply_awq_scale(model, tune_cfg[, attentions, qkvs, output_dicts)
    model = apply_awq_clip(model, tune_cfg, attentions, qkvs, output_dicts)
    return model
