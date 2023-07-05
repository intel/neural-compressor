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

def qdq_tensor(data, config, ratio=1):
    bit = config["bits"]
    scheme = config["scheme"]
    if scheme == "sym":
        maxq = 2 ** (bit - 1) - 1 if bit != 1 else 0
        minq = -2 ** (bit - 1) if bit != 1 else -1
    elif scheme == "asym":
        maxq = 2 ** bit - 1
        minq = 0

    rmin = np.min(data, axis=-1, keepdims=True) * ratio
    rmax = np.max(data, axis=-1, keepdims=True) * ratio
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
    model = model if isinstance(model, BaseModel) else ONNXModel(model) 
    for node in model.nodes():
        if node.name in tune_cfg and tune_cfg[node.name]["weight"]["dtype"] != "fp32":
            if model.get_initializer(node.input[1]) is None:
                continue
            weight = numpy_helper.to_array(
                        model.get_initializer(node.input[1]),
                        base_dir=os.path.dirname(model.model_path) if model.model_path is not None else "").copy()

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

            model.set_initializer(node.input[1], weight)
    return model

def get_weight_scale(weight, group_size):
    org_shape = weight.shape
    weight = np.reshape(weight, (group_size, -1)) if group_size != -1 else weight
    scale = np.mean(np.reshape(np.abs(weight) / np.max(np.abs(weight), axis=0, keepdims=True), org_shape), axis=1)
    return scale

def apply_awq_scale(model, tune_cfg, absorb_pairs, output_dicts):
    best_scales = {}
    new_init_tensors = []
    new_added_mul_nodes = []
    replace_input = []
    updated_nodes = []
 
    for parent, nodes in absorb_pairs.items():
        if any([node.input[0] not in output_dicts for node in nodes]) or \
            any([node.output[0] not in output_dicts for node in nodes]):
            logger.warning("Miss tensors of node {} during AWQ, skip it!".format(node.name))
            continue
        print(parent)
        inp = output_dicts[nodes[0].input[0]]
        inp_scale = np.mean(np.reshape(np.abs(inp[0]), (-1, inp[0].shape[-1])), axis=0)
        weight = []
        org_out = []
        config = tune_cfg[nodes[0].name]
        for node in nodes:
            weight.append(numpy_helper.to_array(
                                model.get_initializer(node.input[1]),
                                base_dir=os.path.dirname(model.model_path) if model.model_path is not None else ""))
            org_out.append(output_dicts[node.output[0]])

        weight = np.concatenate(weight, axis=1)
        #org_out = np.concatenate(org_out, axis=1)
        w_scale = get_weight_scale(weight, config["weight"]["group_size"])
        org_out = np.matmul(inp, weight)
        
        # search scale
        best_error = float("inf")
        best_ratio = -1
        best_scale = None
        n_grid = 20

        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid
            scales = np.clip(np.power(inp_scale, ratio) / np.power(w_scale, (1 - ratio)), 1e-4, None)
            scales = np.reshape(scales / np.sqrt(np.max(scales) * np.min(scales)), (-1, 1))

            q_weight = qdq_tensor(weight * scales, config["weight"]) / scales
            out = np.matmul(inp, q_weight)
            loss = np.mean(np.power((org_out - out), 2))

            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scale = scales

        for node in nodes:
            tensor = model.get_initializer(node.input[1])
            new_tensor = numpy_helper.to_array(tensor, os.path.dirname(model.model_path)) * best_scale
            model.set_initializer(node.input[1], new_tensor)
            output_dicts[node.input[0]] = output_dicts[node.input[0]] / np.reshape(best_scale, (1, -1))

        parent = model.get_node(parent)
        if parent.name in updated_nodes:
            continue

        if parent.op_type in ["LayerNormalization", "BatchNormalization", "InstanceNormalization"]:
            for idx in [1, 2]:
                tensor = model.get_initializer(parent.input[idx])
                new_tensor = numpy_helper.to_array(tensor, os.path.dirname(model.model_path)) / best_scale if \
                    model.model_path is not None else numpy_helper.to_array(tensor) / best_scale
                model.get_initializer(parent.input[idx]).CopyFrom(new_tensor)
                #model.set_initializer(parent.input[idx], new_tensor)
                updated_nodes.append(parent.name)

        elif parent.op_type in ["SimplifiedLayerNormalization", "MatMul", "Gemm", "Mul"] and \
            not all([model.get_initializer(inp) is None for inp in parent.input]):
            for inp in parent.input:
                if model.get_initializer(inp) is not None:
                    tensor = model.get_initializer(inp)
                    new_tensor = numpy_helper.to_array(tensor, os.path.dirname(model.model_path)) / np.reshape(best_scale, (1, -1)) if \
                        model.model_path is not None else numpy_helper.to_array(tensor) / np.reshape(best_scale, (1, -1))
                    model.get_initializer(inp).CopyFrom(numpy_helper.from_array(new_tensor, inp))
                    #model.set_initializer(inp, new_tensor)
            updated_nodes.append(parent.name)

        elif parent.op_type in ["Conv", "FusedConv"]:
            tensor = model.get_initializer(parent.input[2])
            new_tensor = numpy_helper.to_array(tensor, os.path.dirname(model.model_path)) / best_scale if \
                model.model_path is not None else numpy_helper.to_array(tensor) / best_scale
            model.set_initializer(parent.input[2], new_tensor)
            updated_nodes.append(parent.name)

        else:
            # insert mul
            scale_tensor = helper.make_tensor(
                name=parent.output[0] + "_weight_only_scale",
                data_type=onnx_proto.TensorProto.FLOAT,
                dims=best_scale.shape,
                vals=best_scale.flatten().tolist())
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
            output_dicts[mul_node.output[0]] = output_dicts[mul_node.input[0]]
        break
 
    model.add_nodes(new_added_mul_nodes)
    model.add_initializers(new_init_tensors)
    for node, old_input_name, new_input_name in replace_input:
        model.replace_node_input(node, old_input_name, new_input_name)

    return model, output_dicts

def apply_awq_clip(model, tune_cfg, absorb_pairs, output_dicts):
    import copy 
    for parent, nodes in absorb_pairs.items():
        if any([node.input[0] not in output_dicts for node in nodes]) or \
            any([node.output[0] not in output_dicts for node in nodes]):
            logger.warning("Miss tensors of node {} during AWQ, skip it!".format(node.name))
            continue

        inp = np.concatenate(output_dicts[nodes[0].input[0]][0], axis=0)

        for node in nodes:
            if node.name in tune_cfg and tune_cfg[node_name]["weight"]["dtype"] != "fp32":
                node = model.get_node(node_name)
 
                print(node.name)
                group_size = tune_cfg[node_name]["weight"]["group_size"]
                config = tune_cfg[node_name]
                att_weight = numpy_helper.to_array(
                            model.get_initializer(node.input[1]),
                            base_dir=os.path.dirname(model.model_path) if model.model_path is not None else "")
                org_w_shape = att_weight.shape # ic, oc
                width = int(att_weight.shape[-1] / 3)
                v_weight = att_weight[:, 2*width:]
                weight = copy.deepcopy(v_weight)
                org_out = output_dicts[node.output[0]]

                inp = inp.reshape(1, inp.shape[0], -1, group_size) # 1, n_token, n_group, goup_size
                weight = weight.reshape(group_size, -1, 1, weight.shape[-1]) # group_size, n_group, 1, co
                org_out = np.sum(inp * weight.T, axis=-1) # co, n_token, n_group
                org_max_val = np.max(np.abs(weight), axis=0, keepdims=True)  # 1, n_group, 1, co
                min_errs = np.ones(org_max_val.shape) * 1e9  # 1, n_group, 1, co
                best_max_val = np.max(np.abs(weight), axis=0, keepdims=True)

                for i_s in range(10):
                    max_val = org_max_val * (1 - i_s / 100)
                    min_val = - max_val
                    cur_w = np.clip(weight, min_val, max_val)
                    q_weight = qdq_tensor(weight, config["weight"])
                    cur_out = np.sum(inp * q_weight.T, axis=-1)
                    err = np.reshape(np.mean(np.power(cur_out - org_out, 2), axis=1), min_errs.shape)
                    cur_best_idx = err < min_errs
                    min_errs[cur_best_idx] = err[cur_best_idx]
                    best_max_val[cur_best_idx] = max_val[cur_best_idx]

                best_max_val = best_max_val.squeeze(-2)
                weight = weight.reshape(-1, *best_max_val.shape[1:])
                weight = np.clip(weight, -best_max_val, best_max_val)
                model.set_initializer(node.input[1], weight.reshape(org_w_shape))
                print((att_weight==weight.reshape(org_w_shape)).all())
            
    return model

def awq_quantize(model, tune_cfg, dataloader):
    from neural_compressor.adaptor.ox_utils.calibration import ONNXRTAugment

    model = model if isinstance(model, BaseModel) else ONNXModel(model)

    white_nodes = []
    absorb_pairs = model.get_absorb_pairs(["MatMul", "Attention"])
    for parent, nodes in absorb_pairs.items():
        white_nodes.extend([i.name for i in nodes])

    augment = ONNXRTAugment(model,
                            dataloader,
                            [],
                            white_nodes=white_nodes,
                            iterations=list(range(0, 0)))

    augment.augment_graph(activation_only=False, weight_only=False)

    _, output_dicts = augment.get_intermediate_outputs()
    model, output_dicts = apply_awq_scale(model, tune_cfg, absorb_pairs, output_dicts)
    #model = apply_awq_clip(model, tune_cfg, absorb_pairs, output_dicts)
    return model
