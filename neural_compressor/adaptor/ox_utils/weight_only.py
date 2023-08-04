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
from neural_compressor.utils.utility import LazyImport
from neural_compressor.model.model import BaseModel
from neural_compressor.model.onnx_model import ONNXModel
from onnx import numpy_helper, helper

ort = LazyImport("onnxruntime")
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
    bit = config.get("bits", 8)
    scheme = config.get("scheme", "asym")
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
        if node.op_type in ["MatMul", "Attention"] and model.get_initializer(node.input[1]) is not None:
            weight = numpy_helper.to_array(
                        model.get_initializer(node.input[1]),
                        base_dir=os.path.dirname(model.model_path)).copy()
            dtype = weight.dtype
            config = tune_cfg[node.name].get("weight", {})

            org_w_shape = weight.shape # ic, oc
            group_size = config.get("group_size", -1) if config.get("group_size", -1) != -1 else org_w_shape[0]

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
        config = tune_cfg.get(nodes[0].name, {})
        
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
                w_scale = get_weight_scale(weight, config.get("weight", {}).get("group_size", -1))
                org_out = np.matmul(inp, weight)
                scales = np.clip(np.power(inp_scale, ratio) / np.power(w_scale, (1 - ratio)), 1e-4, None)
                scales = np.reshape(scales / np.sqrt(np.max(scales) * np.min(scales)), (-1, 1))

                q_weight = qdq_tensor(weight * scales, config.get("weight", {})) / scales
                out = np.matmul(inp, q_weight)
                loss += np.mean(np.power((org_out - out), 2))

            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scale = scales

        for node in nodes:
            tensor = numpy_helper.to_array(model.get_initializer(node.input[1]),
                                                os.path.dirname(model.model_path))
            new_tensor = tensor * best_scale
            model.set_initializer(node.input[1], new_tensor.astype(tensor.dtype), raw=True)
            output_dicts[node.input[0]] = output_dicts[node.input[0]] / np.reshape(best_scale, (1, -1))

        parent = model.get_node(parent)
        if parent.name in updated_nodes:
            continue

        if parent.op_type in ["LayerNormalization", "BatchNormalization", "InstanceNormalization"]:  # pragma: no cover
            for idx in [1, 2]:
                tensor = numpy_helper.to_array(model.get_initializer(parent.input[idx]),
                                                os.path.dirname(model.model_path))
                new_tensor = tensor / np.reshape(best_scale, (1, -1))
                model.set_initializer(parent.input[idx], new_tensor.astype(tensor.dtype), raw=True)
                updated_nodes.append(parent.name)
            output_dicts[parent.output[0]] = output_dicts[parent.output[0]] / np.reshape(best_scale, (1, -1))

        elif parent.op_type in ["SimplifiedLayerNormalization", "MatMul", "Gemm", "Mul"] and \
                not all([model.get_initializer(inp) is None for inp in parent.input]):
            for inp in parent.input:
                if model.get_initializer(inp) is not None:
                    tensor = numpy_helper.to_array(model.get_initializer(inp),
                                                    os.path.dirname(model.model_path))
                    new_tensor = tensor / np.reshape(best_scale, (1, -1))
                    model.set_initializer(inp, new_tensor.astype(tensor.dtype), raw=True)
            updated_nodes.append(parent.name)
            output_dicts[parent.output[0]] = output_dicts[parent.output[0]] / np.reshape(best_scale, (1, -1))

        elif parent.op_type in ["Conv", "FusedConv"]:  # pragma: no cover
            tensor = numpy_helper.to_array(model.get_initializer(parent.input[2]),
                                            os.path.dirname(model.model_path))
            new_tensor = tensor / np.reshape(best_scale, (1, -1))
            model.set_initializer(parent.input[2], new_tensor.astype(tensor.dtype), raw=True)
            updated_nodes.append(parent.name)
            output_dicts[parent.output[0]] = output_dicts[parent.output[0]] / np.reshape(best_scale, (1, -1))

        else:  # pragma: no cover
            # insert mul
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
            for node in nodes:
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
            config = tune_cfg.get(node.name, {})
            org_weight = numpy_helper.to_array(
                                model.get_initializer(node.input[1]),
                                base_dir=os.path.dirname(model.model_path))
            org_w_shape = org_weight.shape # ic, oc
            group_size = config.get("group_size", -1) if config.get("group_size", -1) != -1 else org_w_shape[0]
            org_out = np.matmul(inp, org_weight) # n_token, oc

            best_error = float("inf")
            best_ratio = 1
            for i_s in range(10):
                ratio = 1 - i_s / 100
                weight = copy.deepcopy(org_weight)
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
                    
                cur_out = np.matmul(inp, weight)
                loss = np.mean(np.power((org_out - cur_out), 2))
                is_best = loss < best_error
                if is_best:
                    best_error = loss
                    best_ratio = ratio
            ratios[node.input[1]] = best_ratio
    model = rtn_quantize(model, tune_cfg, ratios)
    return model

def prepare_inputs(model, n_samples, dataloader):
    """Prepare inputs for weight only quantization.

    Args:
        model (ModelProto or ONNXModel): onnx model
        n_samples (int, optional): calibration sample number.
        dataloader (object): dataloader for calibration.

    Returns:
        inputs: prepared inputs.
        so: session options
    """
    from importlib.util import find_spec
    from neural_compressor.adaptor.ox_utils.util import to_numpy
    
    so = ort.SessionOptions()
    if sys.version_info < (3, 10) and find_spec('onnxruntime_extensions'):  # pragma: no cover
        from onnxruntime_extensions import get_library_path
        so.register_custom_ops_library(get_library_path())
    if model.is_large_model:
        onnx.save_model(model.model,
                        model.model_path + '_augment.onnx',
                        save_as_external_data=True,
                        all_tensors_to_one_file=True,
                        convert_attribute=False)

    session = ort.InferenceSession(
                model.model.SerializeToString(),
                so,
                providers=ort.get_available_providers()) if not model.is_large_model else \
              ort.InferenceSession(
                model.model_path + '_augment.onnx',
                so,
                providers=ort.get_available_providers())
    inputs_names = [i.name for i in session.get_inputs()]
    del session

    inputs = []
    for i, data in enumerate(dataloader):
        if ((i + 1) * dataloader.batch_size) > n_samples:
            break
        if len(inputs_names) != 1 or isinstance(data[0], dict):
            assert len(data[0]) == len(inputs_names), "Input number mismatch, " \
                    "require {} but get {}".format(len(inputs_names), len(data[0]))
            
        if isinstance(data[0], dict):
            inputs.append(dict([(name, to_numpy(inp_data)) for name, inp_data in data[0].items()]))
        else:
            inputs.append(dict([(name, to_numpy(inp)) for name, inp in zip(inputs_names, data[0])]))
    return inputs, so

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
        dataloader (object): dataloader for calibration.
        n_samples (int, optional): calibration sample number.
        auto_scale (bool, optional): whether enable scale for salient weight. Defaults to True.
        mse_range (bool, optional):  whether enable clip for weight by checking mse. Defaults to True.
        n_blocks (int, optional): split model into block number to avoid OOM.

    Returns:
        model: fake quantized ONNXModel
    """
    model = model if isinstance(model, BaseModel) else ONNXModel(model)
    output_dicts = {}

    if mse_range or mse_range:
        absorb_pairs = model.get_absorb_pairs(["MatMul", "Attention"])

        inputs, so = prepare_inputs(model, n_samples, dataloader)
        del dataloader

        org_output = copy.deepcopy(model.model.graph.output)
        model.remove_tensors_from_outputs([i.name for i in org_output])
        num_block = math.ceil(len(absorb_pairs) / n_blocks)
        dump_pairs = {}
        for idx, parent in enumerate(absorb_pairs):
            if (idx + 1) % num_block == 0 or (idx + 1) == len(absorb_pairs):
                dump_pairs[parent] = absorb_pairs[parent]
                output_dicts = {}
                dump_tensor = list(set([i.input[0] for nodes in dump_pairs.values() for i in nodes]))
                model.add_tensors_to_outputs(dump_tensor)

                if model.is_large_model:
                    onnx.save_model(model.model,
                                    model.model_path + '_augment.onnx',
                                    save_as_external_data=True,
                                    all_tensors_to_one_file=True,
                                    convert_attribute=False)

                session = ort.InferenceSession(
                            model.model.SerializeToString(),
                            so,
                            providers=ort.get_available_providers()) if not model.is_large_model else \
                          ort.InferenceSession(
                            model.model_path + '_augment.onnx',
                            so,
                            providers=ort.get_available_providers())

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
                dump_pairs[parent] = absorb_pairs[parent]

        model.model.graph.output.MergeFrom(org_output)
    return model

def gptq(Ws, Hs, config, blocksize=128, percdamp=.01, actorder=False, mse=False, perchannel=True):
    """Quant the model with Activation-aware Weight quantization(AWQ) method.

    Args:
        Ws (list): list of weight.
        Hs (list): list of Hessian matrix.
        config (dict): quantizaion config.
        blocksize (int, optional): blocksize to quantize weight.
        percdamp (float, optional): percent of the average Hessian diagonal to use for dampening.
        actorder (bool, optional): whether rearrange Hessian matrix considering the diag's value.
        mse (bool, optional): whether get scale and zero point with mse error.
        perchannel (bool, optional): whether quantize weight per-channel.

    Returns:
        Qs: fake quantized weights
    """
    Qs = []
    group_size = config.get("weight", {}).get("group_size", -1)
    bits = config.get("weight", {}).get("bits", 8)
    scheme = config.get("weight", {}).get("scheme", "asym")
    maxq = 2 ** bits - 1
    grid=100
    maxshrink=.8
    norm=2.4

    def find_params(weight):
        org_shape = weight.shape
        # find zp, scale
        if not perchannel:
            weight = np.expand_dims(weight.flatten(), axis=1)
        tmp = np.zeros(weight.shape[1])
        xmin = np.minimum(np.min(weight, axis=0), tmp)
        xmax = np.maximum(np.max(weight, axis=0), tmp)
        if scheme == "sym":
            xmax = np.maximum(np.abs(xmin), xmax)
            tmp = xmin < 0
            if np.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        scale = (xmax - xmin) / maxq
        if scheme == "sym":
            zero = np.ones(scale.shape) * (maxq + 1) / 2
        else:
            zero = np.round(-xmin / scale)
        if mse:
            best = np.ones([weight.shape[1]]) * float("inf")
            for i in range(int(maxshrink * grid)):
                p = 1 - i / grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / maxq
                zero1 = np.round(-xmin1 / scale1) if scheme != "sym" else zero
                q = np.clip(np.round(weight / scale1) + zero1, 0, maxq)
                q -= weight
                q = np.power(np.abs(q), norm)
                err = np.sum(q, 0)
                tmp = err < best
                if np.any(tmp):
                    best[tmp] = err[tmp]
                    scale[tmp] = scale1[tmp]
                    zero[tmp] = zero1[tmp]
        if not perchannel:
            tmp = org_shape[1]
            scale = np.repeat(scale, tmp)
            zero = np.repeat(zero, tmp)
        shape = [-1] + [1] * (len(org_shape) - 1)
        scale = np.reshape(scale, shape)
        zero = np.reshape(zero, shape)
        return scale, zero

    for W, H in zip(Ws, Hs):
        dtype = W.dtype
        shape = W.shape
        scale, zp = find_params(W)
        dead = np.diag(H) == 0
        H[dead, dead] = 1
        W[dead, :] = 0 # such channel makes no contribution to quantization computation

        # rearrange considering the diag's value
        if actorder:
            perm = np.argsort(np.diag(H))[::-1]
            W = W[perm, :]
            H = H[perm, :][:, perm]
        Losses = np.zeros(W.shape)
        Q = np.zeros(W.shape)
        damp = percdamp * np.mean(np.diag(H))
        diag = np.arange(shape[0])
        H[diag, diag] += damp # add a average value of 
        H = np.linalg.cholesky(np.linalg.inv(H)).T
        Hinv = H
        for i1 in range(0, shape[0], blocksize):
            i2 = min(i1 + blocksize, shape[0])
            count = i2 - i1

            W1 = copy.deepcopy(W[i1:i2, :])
            Q1 = np.zeros(W1.shape)
            Err1 = np.zeros(W1.shape)
            Losses1 = np.zeros(W1.shape)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count): # within a block, channel wise
                w = W1[i, :]
                d = Hinv1[i, i]

                if group_size != -1:
                    if (i1 + i) % group_size == 0:
                        scale, zp = find_params(W[(i1 + i):(i1 + i + group_size), :])

                q = (scale * (np.clip(np.round(np.expand_dims(w, axis=1) / scale) + zp, 0, maxq) - zp)).flatten()
                Q1[i, :] = q
                Losses1[i, :] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[i:, :] -= np.matmul(np.expand_dims(Hinv1[i:, i], axis=1), np.expand_dims(err1, axis=0))
                Err1[i, :] = err1

            Q[i1:i2, :] = Q1
            Losses[i1:i2, :] = Losses1 / 2

            W[i2:, :] -= np.matmul(Hinv[i2:, i1:i2], Err1)

        if actorder:
            invperm = np.argsort(perm)
            Q = Q[invperm, :]

        Qs.append(np.reshape(Q, W.shape).astype(dtype))
    del Ws
    return Qs

def gptq_quantize(model,
                  tune_cfg,
                  dataloader,
                  n_samples=128,
                  blocksize=128,
                  percdamp=.01,
                  actorder=False,
                  mse=False,
                  perchannel=True
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
                            'algorithm': 'GPTQ'
                        }
                }
        dataloader (object): dataloader for calibration.
        n_samples (int, optional): calibration sample number.
        blocksize (int, optional): blocksize to quantize weight.
        percdamp (float, optional): percent of the average Hessian diagonal to use for dampening.
        actorder (bool, optional): whether rearrange Hessian matrix considering the diag's value.
        mse (bool, optional): whether get scale and zero point with mse error.
        perchannel (bool, optional): whether quantize weight per-channel.

    Returns:
        model: fake quantized ONNXModel
    """
    model = model if isinstance(model, BaseModel) else ONNXModel(model)
    output_dicts = {}
    absorb_pairs = model.get_absorb_pairs(["MatMul", "Attention"])

    inputs, so = prepare_inputs(model, n_samples, dataloader)
    del dataloader
    org_output = copy.deepcopy(model.model.graph.output)
    model.remove_tensors_from_outputs([i.name for i in org_output])
    for parent, nodes in absorb_pairs.items():
        dump_tensor = list(set([i.input[0] for i in nodes]))
        model.add_tensors_to_outputs(dump_tensor)

        if model.is_large_model:
            onnx.save_model(model.model,
                            model.model_path + '_augment.onnx',
                            save_as_external_data=True,
                            all_tensors_to_one_file=True,
                            convert_attribute=False)

        session = ort.InferenceSession(
                    model.model.SerializeToString(),
                    so,
                    providers=ort.get_available_providers()) if not model.is_large_model else \
                  ort.InferenceSession(
                    model.model_path + '_augment.onnx',
                    so,
                    providers=ort.get_available_providers())

        weights = [copy.deepcopy(numpy_helper.to_array(model.get_initializer(node.input[1]),
                    os.path.dirname(model.model_path))) for node in nodes]
        Hs = [np.zeros((i.shape[0], i.shape[0])) for i in weights]
        nsamples = 0
        for inp in inputs:
            output_dicts = {}
            for output_idx, output in enumerate(session.run(None, inp)):
                output_dicts.setdefault(dump_tensor[output_idx], []).append(output)

            inp = output_dicts[nodes[0].input[0]][0]
            tmp = inp.shape[0]
            if len(inp.shape) == 3:
                inp = np.reshape(inp, (-1, inp.shape[-1]))
            Hs = [i * (nsamples / (nsamples + tmp)) for i in Hs]
            nsamples += tmp
            inp = np.sqrt(2 / nsamples) * inp
            Hs = [i + np.matmul(inp.T, inp) for i in Hs]

        model.remove_tensors_from_outputs(dump_tensor)
        weights = gptq(weights,
                       Hs,
                       tune_cfg.get(nodes[0].name, {}),
                       blocksize=blocksize,
                       percdamp=percdamp,
                       actorder=actorder,
                       mse=mse,
                       perchannel=perchannel
                      )
        for name, weight in zip([i.input[1] for i in nodes], weights):
            model.set_initializer(name, weight, raw=True)
    model.model.graph.output.MergeFrom(org_output)
    return model
