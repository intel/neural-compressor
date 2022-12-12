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

"""Helper functions to export model from PyTorch/TensorFlow to ONNX."""

import os
import numpy as np
from collections import UserDict
from neural_compressor.adaptor.torch_utils.util import input2tuple
from neural_compressor.utils import logger
from neural_compressor.utils.utility import LazyImport

torch = LazyImport('torch')
onnx = LazyImport('onnx')
ort = LazyImport('onnxruntime')
ortq = LazyImport('onnxruntime.quantization')


def update_weight_bias(
    int8_model,
    fp32_onnx_path,
):
    """Update wegiht and bias of FP32 ONNX model with QAT INT8 PyTorch model .

    Args:
        int8_model (torch.nn.module): int8 model.
        fp32_onnx_path (str): path to fp32 onnx model.
    """
    # collect weights, bias from int8 PT model
    fp32_onnx_model = onnx.load(fp32_onnx_path)
    model_dict = int8_model.state_dict()
    int8_model_dict = {}
    for name, param in model_dict.items():
        # '_packed_params._packed_weight' is specific for quantized Embedding
        if '_packed_params._packed_weight' in name:
            name = name.replace('._packed_params._packed_weight', '').split('.module')[0]
            int8_model_dict[name+'.weight'] = param.dequantize()
        # '_packed_params._packed_params' is specific for quantized Linear
        elif '_packed_params._packed_params' in name and isinstance(param, tuple):
            name = name.replace('._packed_params._packed_params', '').split('.module')[0]
            int8_model_dict[name+'.bias'] = param[1]
            int8_model_dict[name+'.weight'] = param[0].dequantize()
        # '.weight' and '.bias' is specific for quantized Conv
        elif '.weight' in name:
            int8_model_dict[name] = param.dequantize()
        elif '.bias' in name:
            int8_model_dict[name] = param
        else:
            int8_model_dict[name] = param

    # replace weight and bias in onnx fp32 model for QAT
    from onnx import helper
    tensor_list = [tensor for tensor in fp32_onnx_model.graph.initializer]
    for tensor in tensor_list:
        if tensor.name in int8_model_dict:
            np_tensor = int8_model_dict[tensor.name].detach().cpu().numpy()
            new_tensor = helper.make_tensor(
                name=tensor.name,
                data_type=tensor.data_type,
                dims=tensor.dims,
                vals=np_tensor,
            )
            fp32_onnx_model.graph.initializer.remove(tensor)
            fp32_onnx_model.graph.initializer.append(new_tensor)
    onnx.save(fp32_onnx_model, fp32_onnx_path)


def set_data_type(
    dtype,
):
    """Set data type of activation and weight with string dtype.

    Args:
        dtype (str): data type description.

    Returns:
        activation_type: activation type.
        weight_type: weight type.
    """
    # Get data type for activation and weight from dtype
    if 'U8U8' in dtype:   # pragma: no cover
        activation_type = ortq.QuantType.QUInt8
        weight_type = ortq.QuantType.QUInt8
    elif 'S8S8' in dtype:   # pragma: no cover
        activation_type = ortq.QuantType.QInt8
        weight_type = ortq.QuantType.QInt8
    elif 'U8S8' in dtype:
        activation_type = ortq.QuantType.QUInt8
        weight_type = ortq.QuantType.QInt8
    else:   # pragma: no cover 
        logger.error("Right now, we don't support dtype: {}, \
                        please use U8U8/U8S8/S8S8.".format(dtype))
    logger.info("Weight type: {}.".format(weight_type))
    logger.info("Activation type: {}.".format(activation_type))
    return activation_type, weight_type


def get_node_mapping(
    fp32_model,
    fp32_onnx_path,
):
    """Get PyTorch module and ONNX node mapping.

    Args:
        fp32_model (torch.nn.Module): quantization configuration from PyTorch.
        fp32_onnx_path (str): path to fp32 onnx model.

    Returns:
        module_node_mapping: op mapping from PyTorch to ONNX.
    """
    def check_data(op_type, data, module_dict):
        for name, value in module_dict.items():
            if value.shape == data.shape:
                if (value == data).all():
                    module_dict.pop(name)
                    return name
                elif op_type == 'Conv':
                    # Convolution weight data have fluction and BN fusion will insert scale.
                    # We use the weight scale of the first output channel to check.
                    weight_scale = value[0] / data[0]
                    if np.allclose(weight_scale - np.mean(weight_scale), 0, atol=1.e-5):
                        module_dict.pop(name)
                        return name
        return None

    module_dict = {}
    for name, module in fp32_model.named_modules():
        if 'Conv' in str(module.__class__.__name__) or \
          'Embedding' in str(module.__class__.__name__) or \
          'Linear' in str(module.__class__.__name__):
            if hasattr(module, 'weight'):
                value = module.weight.detach().cpu().numpy()
                module_dict[name] = value

    module_node_mapping = {}
    fp32_onnx_model = onnx.load(fp32_onnx_path)
    initializer_data = {tensor.name: tensor for tensor in fp32_onnx_model.graph.initializer}
    from onnx import numpy_helper
    for node in fp32_onnx_model.graph.node:
        if node.op_type in op_types_to_quantize:
            if node.op_type == 'MatMul' and node.input[1] in initializer_data:
                data = numpy_helper.to_array(initializer_data[node.input[1]]).T
            elif node.op_type == 'Gather' and node.input[0] in initializer_data:
                data = numpy_helper.to_array(initializer_data[node.input[0]])
            elif node.op_type in ['Conv', 'Gemm']:
                data = numpy_helper.to_array(initializer_data[node.input[1]])
            else:
                continue
            pt_name = check_data(node.op_type, data, module_dict)
            if pt_name:
                module_node_mapping[pt_name] = node.name
    return module_node_mapping


def get_quantizable_onnx_ops(
    int8_model,
    module_node_mapping
):
    """Get quantizable onnx ops.

    Args:
        int8_model (torch.nn.Module): PyTorch int8 model.
        module_node_mapping (dict): op mapping from PyTorch to ONNX.

    Returns:
        quantize_nodes: all onnx node that should be quantized.
    """
    quantize_nodes = []
    for name, module in int8_model.named_modules():
        if 'Conv' in str(module.__class__.__name__) or \
          'Embedding' in str(module.__class__.__name__) or \
          'Linear' in str(module.__class__.__name__):
            if hasattr(module, 'weight') and callable(module.weight):
                if module.weight().dtype in [torch.qint8, torch.quint8]:
                    if name.split('.module')[0] in module_node_mapping:
                        node = module_node_mapping[name.split('.module')[0]]
                        quantize_nodes.append(node)
    return quantize_nodes


def build_scale_mapping(
    fp32_onnx_path,
    module_node_mapping,
    int8_scale_info,
):
    """Build scale mapping.

    Args:
        fp32_onnx_path (str): path to fp32 onnx model.
        module_node_mapping (dict): op mapping from PyTorch to ONNX.
        int8_scale_info (dict): int8 scale infomation.

    Returns:
        scale_zp_dict: scale and zero_point dict.
    """
    node_module_mapping = {}
    for module_name, node_name in module_node_mapping.items():
        node_module_mapping[node_name] = module_name
    # Match scale and zeropoint from PyTorch to ONNX node
    scale_zp_dict = {}
    fp32_onnx_model = onnx.load(fp32_onnx_path)
    for node in fp32_onnx_model.graph.node:
        if node.name in node_module_mapping:
            module_name = node_module_mapping[node.name]

            # For fine-grained fx and fuse pattern
            if module_name + '.module' in int8_scale_info:
                module_name = module_name + '.module'
            elif module_name + '.0' in int8_scale_info:
                module_name = module_name + '.0'
            elif module_name + '.module.0' in int8_scale_info:
                module_name = module_name + '.module.0'

            if module_name in int8_scale_info:
                recoder = int8_scale_info[module_name]
                input_scale_args = node.input[0] + '_scale'
                input_zp_args = node.input[0] + '_zero_point'
                scale_zp_dict[input_scale_args] = recoder['input_scale']
                scale_zp_dict[input_zp_args] = recoder['input_zeropoint']
                ### We need Matmul+Add to match Linear for output scale and zero-point
                output_scale_args = node.output[0] + '_scale'
                output_zp_args = node.output[0] + '_zero_point'
                scale_zp_dict[output_scale_args] = recoder['output_scale']
                scale_zp_dict[output_zp_args] = recoder['output_zeropoint']
    return scale_zp_dict


def set_scale_info(
    int8_onnx_model,
    scale_zp_dict,
    activation_type,
):
    """Set scale to ONNX model.

    Args:
        int8_onnx_path (str): path to onnx file.
        scale_zp_dict (dict): scale zero_point dict.
        activation_type : activation type.

    Returns:
        int8_onnx_model: int8 onnx model object.
    """
    # set scale and zeropoint from PyTorch int8 model to ONNX int8 model
    from onnx import helper
    tensor_list = [tensor for tensor in int8_onnx_model.graph.initializer]
    for tensor in tensor_list:
        if tensor.name in scale_zp_dict:
            value = scale_zp_dict[tensor.name]
            if 'zero_point' in tensor.name and activation_type == ortq.QuantType.QInt8:
                value -= 128
            new_tensor = helper.make_tensor(
                name=tensor.name,
                data_type=tensor.data_type,
                dims=tensor.dims,
                vals=[value],
            )
            int8_onnx_model.graph.initializer.remove(tensor)
            int8_onnx_model.graph.initializer.append(new_tensor)
    return int8_onnx_model


def recalculate_bias(
    int8_onnx_path,
    scale_zp_dict,
    quantize_nodes,
    quant_format,
):  
    """Recalculate bias.

    Args:
        int8_onnx_model (ModelProto): onnx int8 model to process.
        scale_zp_dict (dict): scale zero_point dict.
        quantize_nodes (list): quantize nodes list.
        quant_format (QuantFormat): quantization format.

    Returns:
        int8_onnx_model: processed onnx int8 model.
    """
    int8_onnx_model = onnx.load(int8_onnx_path)
    model = ortq.onnx_model.ONNXModel(int8_onnx_model)
    if quant_format == ortq.QuantFormat.QDQ:
        for node in int8_onnx_model.graph.node:
            if node.name in quantize_nodes and (node.op_type == 'Conv' or node.op_type == 'Gemm'):
                input_name, weight_name, bias_name = node.input[:3]
                for parent in model.get_parents(node):
                    if parent.output[0] == input_name:
                        input_scale_name = parent.input[1]
                    elif parent.output[0] == weight_name:
                        weight_scale_name = parent.input[1]
                    elif parent.output[0] == bias_name:
                        bias_quantized_name = parent.input[0]
                        bias_scale_name = parent.input[1]
                weight_scale_data = onnx.numpy_helper.to_array(model.get_initializer(weight_scale_name))
                new_input_scale_data = scale_zp_dict[input_scale_name]
                origin_bias_quantized_data = onnx.numpy_helper.to_array(model.get_initializer(bias_quantized_name))
                origin_bias_scale_data = onnx.numpy_helper.to_array(model.get_initializer(bias_scale_name))
                origin_bias_data = origin_bias_quantized_data * origin_bias_scale_data
                new_bias_scale_data = new_input_scale_data * weight_scale_data
                new_bias_quantized_data = (origin_bias_data / new_bias_scale_data).round().astype(np.int32)
                model.get_initializer(bias_scale_name).raw_data = new_bias_scale_data.tobytes()
                model.get_initializer(bias_quantized_name).raw_data = new_bias_quantized_data.tobytes()
    elif quant_format == ortq.QuantFormat.QOperator:
        for node in int8_onnx_model.graph.node:
            if node.op_type == 'QLinearConv' or node.op_type == 'QGemm':
                input_scale_name, weight_scale_name = node.input[1], node.input[4]
                bias_quantized_name = node.input[8] if node.op_type == 'QLinearConv' else node.input[6]
                weight_scale_data = onnx.numpy_helper.to_array(model.get_initializer(weight_scale_name))
                new_input_scale_data = scale_zp_dict[input_scale_name]
                origin_input_scale_data = onnx.numpy_helper.to_array(model.get_initializer(input_scale_name))
                origin_bias_quantized_data = onnx.numpy_helper.to_array(model.get_initializer(bias_quantized_name))
                origin_bias_scale_data = origin_input_scale_data * weight_scale_data
                origin_bias_data = origin_bias_quantized_data * origin_bias_scale_data
                new_bias_scale_data = new_input_scale_data * weight_scale_data
                new_bias_quantized_data = (origin_bias_data / new_bias_scale_data).round().astype(np.int32)
                model.get_initializer(bias_quantized_name).raw_data = new_bias_quantized_data.tobytes()
    return int8_onnx_model


def remove_nodes_by_name(int8_onnx_model, node_names):
    """Remove nodes from model by names.

    Args:
        int8_onnx_model (ModelProto): onnx int8 model to process.
        node_names (list): names of nodes to remove.

    Returns:
        int8_onnx_model: processed onnx int8 model.
    """
    while node_names:
        for node in int8_onnx_model.graph.node:
            if node.name in node_names:
                int8_onnx_model.graph.node.remove(node)
                node_names.remove(node.name)
    return int8_onnx_model


def sub_graph_with_int32_bias(
    int8_onnx_model, 
    node,
    a_info,
    b_info,
    bias_name,
    output_name,
):
    """Generate a sub graph with int32 bias.

    Args:
        int8_onnx_model (ModelProto): onnx int8 model to process.
        node (NodeProto): MatMul node belonging to nn.quantized.Linear module.
        a_info (list): info of input a for nn.quantized.Linear module.
        b_info (list): info of input b for nn.quantized.Linear module.
        bias_name (str): name of bias.
        output_name (_type_): output name of the sub graph.

    Returns:
        int8_onnx_model: processed onnx int8 model.
    """
    from onnx import TensorProto
    a, a_scale, a_zero_point = a_info
    b, b_scale, b_zero_point = b_info
    a_scale = ortq.onnx_model.ONNXModel(int8_onnx_model).get_initializer(a_scale)
    a_scale = onnx.numpy_helper.to_array(a_scale)
    b_scale = ortq.onnx_model.ONNXModel(int8_onnx_model).get_initializer(b_scale)
    b_scale = onnx.numpy_helper.to_array(b_scale)
    bias = ortq.onnx_model.ONNXModel(int8_onnx_model).get_initializer(bias_name)
    bias_dims = bias.dims
    bias = onnx.numpy_helper.to_array(bias)
    bias_scale = a_scale * b_scale
    quantized_bias = (bias / bias_scale).round().astype(np.int32)
    quantized_bias = np.asarray(quantized_bias, dtype=np.int32).reshape(bias_dims)
    packed_bias_initializer = onnx.numpy_helper.from_array(quantized_bias, 
                                                        bias_name + "_quantized")
    int8_onnx_model.graph.initializer.extend([packed_bias_initializer])

    matmul_node = onnx.helper.make_node("MatMulInteger",
                        inputs=[a, b, a_zero_point, b_zero_point],
                        outputs=[node.output[0] + '_matmulinteger'],
                        name = node.name + '_matmulinteger')
    add_node = onnx.helper.make_node("Add",
                        inputs=[node.output[0] + '_matmulinteger', bias_name + '_quantized'],
                        outputs=[node.output[0] + '_add'],
                        name = node.name + '_add'
                        )
    cast_node = onnx.helper.make_node("Cast",
                                    inputs=[node.output[0] + '_add'],
                                    outputs=[node.output[0] + '_cast'],
                                    to=getattr(TensorProto, 'FLOAT'),
                                    name = node.name + '_cast')

    new_tensor = onnx.helper.make_tensor(
        name=node.name + '_bias_scale',
        data_type=TensorProto.FLOAT,
        dims=list(bias_scale.shape),
        vals=bias_scale,
    )
    int8_onnx_model.graph.initializer.append(new_tensor)

    mul_node = onnx.helper.make_node("Mul",
                                    inputs=[node.output[0] + '_cast', node.name + '_bias_scale'],
                                    outputs=[output_name],
                                    name=node.name + '_mul')
    
    int8_onnx_model.graph.node.extend([matmul_node, add_node, cast_node, mul_node])
    return int8_onnx_model

def qdq_fp32_bias(
    int8_onnx_model,
    quant_format,
):
    """Excute post-process on int8 onnx model with recipe 'QDQ_OP_FP32_BIAS'.

    Insert QDQ before quantizable op and using fp32 bias.

    Args:
        int8_onnx_model (ModelProto): onnx int8 model to process.
        quant_format (QuantFormat): quantization format.

    Returns:
        int8_onnx_model: processed onnx int8 model.
    """
    # For QDQ quantization format, nn.quantized.Linear module will be 
    # converted to the following format:
    #  QuantizeLinear
    #        |
    # DequantizeLinear
    #        |             
    #      MatMul
    #        |
    #       Add
    # 
    # For QOperator quantization format, nn.quantized.Lienar module will be 
    # converted to the following format:
    #     QuantizeLinear
    #           |
    #  MatMulIntegerToFloat
    #           |
    #          Add 
    if quant_format == ortq.QuantFormat.QDQ:
        return int8_onnx_model
    elif quant_format == ortq.QuantFormat.QOperator:
        remove_nodes = set()
        for node in int8_onnx_model.graph.node:
            if node.op_type == 'QLinearMatMul':
                dequantizelinear_node = ortq.onnx_model.ONNXModel(int8_onnx_model).get_children(node)[0]
                add_node = ortq.onnx_model.ONNXModel(int8_onnx_model).get_children(dequantizelinear_node)[0]
                a = node.input[0]
                a_scale = node.input[1]
                a_zero_point = node.input[2]
                b = node.input[3]
                b_scale = node.input[4]
                b_zero_point = node.input[5]
                matmulintegertofloat_node = onnx.helper.make_node("MatMulIntegerToFloat",
                                    inputs=[a, b, a_scale, b_scale, a_zero_point, b_zero_point],
                                    outputs=[node.output[0]],
                                    name=node.name + '_matmulintegertofloat',
                                    domain='com.microsoft')
                for idx in range(len(add_node.input)):
                    if add_node.input[idx] == dequantizelinear_node.output[0]:
                        add_node.input[idx] = node.output[0]
                remove_nodes.add(node.name)
                remove_nodes.add(dequantizelinear_node.name)
                int8_onnx_model.graph.node.extend([matmulintegertofloat_node])

        int8_onnx_model = remove_nodes_by_name(int8_onnx_model, remove_nodes)
        return int8_onnx_model

def qdq_int32_bias(
    int8_onnx_model,
    quantize_nodes,
    quant_format,
):
    """Excute post-process on int8 onnx model with recipe 'QDQ_OP_INT32_BIAS'.

    Insert QDQ before quantizable op and using int32 bias.

    Args:
        int8_onnx_model (ModelProto): onnx int8 model to process.
        quantize_nodes (list): quantize nodes list.
        quant_format (QuantFormat): quantization format.

    Returns:
        int8_onnx_model: processed onnx int8 model.
    """
    # For QDQ/Operator quantization format, nn.quantized.Linear module will be 
    # converted to the following format:
    #  QuantizeLinear
    #        |
    #  MatMulInteger
    #        |
    #       Add
    #        |
    #      Cast
    #        |
    #       Mul
    if quant_format == ortq.QuantFormat.QDQ:
        remove_nodes = set()
        replace_input = {}
        for node in int8_onnx_model.graph.node:
            if node.name in quantize_nodes and node.op_type == 'MatMul':
                parents = ortq.onnx_model.ONNXModel(int8_onnx_model).get_parents(node)
                add_node = ortq.onnx_model.ONNXModel(int8_onnx_model).get_children(node)[0]
                bias_name = None
                for inp in add_node.input:
                    if inp.endswith('.bias'):
                        bias_name = inp
                if not bias_name: # pragma: no cover 
                    continue

                for parent in parents:
                    grand_parent = ortq.onnx_model.ONNXModel(int8_onnx_model).get_parents(parent)
                    if grand_parent:
                        replace_input[parent.output[0]] = grand_parent[0].input[0]

                int8_onnx_model = sub_graph_with_int32_bias(int8_onnx_model, 
                                                                node, 
                                                                parents[0].input[:3],
                                                                parents[1].input[:3],
                                                                bias_name, 
                                                                add_node.output[0])
                remove_nodes.add(node.name)
                remove_nodes.add(parents[0].name)
                remove_nodes.add(parents[1].name)
                remove_nodes.add(add_node.name)
        int8_onnx_model = remove_nodes_by_name(int8_onnx_model, remove_nodes)
        for node in int8_onnx_model.graph.node: # pragma: no cover 
            for i in range(len(node.input)):
                if node.input[i] in replace_input: 
                    node.input[i] = replace_input[node.input[i]]
    elif quant_format == ortq.QuantFormat.QOperator:
        remove_nodes = set()
        for node in int8_onnx_model.graph.node:
            if node.op_type == 'QLinearMatMul':
                dequantizelinear_node = ortq.onnx_model.ONNXModel(int8_onnx_model).get_children(node)[0]
                add_node = ortq.onnx_model.ONNXModel(int8_onnx_model).get_children(dequantizelinear_node)[0]

                bias_name = None
                for inp in add_node.input:
                    if inp.endswith('.bias'):
                        bias_name = inp
                if not bias_name: # pragma: no cover 
                    continue

                int8_onnx_model = sub_graph_with_int32_bias(int8_onnx_model, 
                                                                node, 
                                                                node.input[:3],
                                                                node.input[3:6],
                                                                bias_name, 
                                                                add_node.output[0])
                remove_nodes.add(node.name)
                remove_nodes.add(add_node.name)
                remove_nodes.add(dequantizelinear_node.name)

        int8_onnx_model = remove_nodes_by_name(int8_onnx_model, remove_nodes)
    return int8_onnx_model

def qdq_fp32_bias_qdq(
    int8_onnx_model,
    quantize_nodes,
    quant_format,
):
    """Excute post-process on onnx int8 model with recipe 'QDQ_OP_FP32_BIAS_QDQ'.

    Insert QDQ before and after quantizable op and using fp32 bias.

    Args:
        int8_onnx_model (ModelProto): onnx int8 model to process.
        quantize_nodes (list): quantize nodes list.
        quant_format (QuantFormat): quantization format.

    Returns:
        int8_onnx_model: processed onnx int8 model.
    """
    # For QDQ quantization format, nn.quantized.Linear module will be 
    # converted to the following format:
    #  QuantizeLinear
    #        |
    # DequantizeLinear   
    #        |
    #      MatMul
    #        |
    #       Add
    #        |
    #  QuantizeLinear
    #        |
    #  DequantizeLinear
    # 
    # For QOperator quantization format, nn.quantized.Lienar module will be 
    # converted to the following format:
    #     QuantizeLinear
    #           |
    #  MatMulIntegerToFloat
    #           |
    #          Add
    #           |
    #     QuantizeLinear
    #           |
    #    DequantizeLinear
    if quant_format == ortq.QuantFormat.QDQ:
        for node in int8_onnx_model.graph.node:
            if node.name in quantize_nodes and node.op_type == 'MatMul':
                quantizelinear_node = ortq.onnx_model.ONNXModel(int8_onnx_model).get_children(node)[0]
                deqauntizelinear_node = ortq.onnx_model.ONNXModel(int8_onnx_model).get_children(quantizelinear_node)[0]
                add_node = ortq.onnx_model.ONNXModel(int8_onnx_model).get_children(deqauntizelinear_node)[0]
                deqauntizelinear_node.output[0] = add_node.output[0]
                add_node.output[0] = add_node.output[0] + '_add'
                for i in range(len(add_node.input)):
                    if not add_node.input[i].endswith('.bias'):
                        add_node.input[i] = node.output[0]
                quantizelinear_node.input[0] = add_node.output[0]
    elif quant_format == ortq.QuantFormat.QOperator:
        import copy
        remove_nodes = set()
        for node in int8_onnx_model.graph.node:
            if node.op_type == 'QLinearMatMul':
                dequantizelinear_node = ortq.onnx_model.ONNXModel(int8_onnx_model).get_children(node)[0]
                add_node = ortq.onnx_model.ONNXModel(int8_onnx_model).get_children(dequantizelinear_node)[0]
                a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point = node.input[:8]
                matmulintegertofloat_node = onnx.helper.make_node("MatMulIntegerToFloat",
                                    inputs=[a, b, a_scale, b_scale, a_zero_point, b_zero_point],
                                    outputs=[node.output[0]],
                                    name=node.name + '_matmulintegertofloat',
                                    domain='com.microsoft')

                for idx in range(len(add_node.input)):
                    if add_node.input[idx] == dequantizelinear_node.output[0]:
                        add_node.input[idx] = node.output[0]

                quantizelinear_node = onnx.helper.make_node("QuantizeLinear",
                                    inputs=[add_node.output[0] +'_add', y_scale, y_zero_point],
                                    outputs=[node.output[0] + '_quantizelinear'],
                                    name=node.name + '_quantizelinear')

                dequantizelinear_node.input[0] = node.output[0] + '_quantizelinear'
                dequantizelinear_node.output[0] = copy.deepcopy(add_node.output[0])
                add_node.output[0] = add_node.output[0] +'_add'

                remove_nodes.add(node.name)
                int8_onnx_model.graph.node.extend([matmulintegertofloat_node, quantizelinear_node])

        int8_onnx_model = remove_nodes_by_name(int8_onnx_model, remove_nodes)
    return int8_onnx_model

def torch_to_fp32_onnx(
    fp32_model,
    save_path,
    example_inputs,
    opset_version=14,
    dynamic_axes={"input": {0: "batch_size"},
                  "output": {0: "batch_size"}},
    input_names=None,
    output_names=None,
    do_constant_folding=True,
    verbose=True,
):
    """Export FP32 PyTorch model into FP32 ONNX model.

    Args:
        fp32_model (torch.nn.module): fp32 model.
        int8_model (torch.nn.module): int8 model.
        save_path (str): save path of ONNX model.
        example_inputs (dict|list|tuple|torch.Tensor): used to trace torch model.
        opset_version (int, optional): opset version. Defaults to 14.
        dynamic_axes (dict, optional): dynamic axes. Defaults to {"input": {0: "batch_size"}, 
                                                                  "output": {0: "batch_size"}}.
        input_names (list, optional): input names. Defaults to None.
        output_names (list, optional): output names. Defaults to None.
        do_constant_folding (bool, optional): do constant folding or not. Defaults to True.
        verbose (bool, optional): dump verbose or not. Defaults to True.
    """
    if input_names:
        example_input_names = input_names
    else:
        example_input_names = ['input']
        if isinstance(example_inputs, dict) or isinstance(example_inputs, UserDict):
            example_input_names = list(example_inputs.keys())

    torch.onnx.export(
        fp32_model,
        input2tuple(example_inputs),
        save_path,
        opset_version=opset_version,
        input_names=example_input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=do_constant_folding,
    )
    if verbose:
        info = "The FP32 ONNX Model exported to path: {0}".format(save_path)
        logger.info("*"*len(info))
        logger.info(info)
        logger.info("*"*len(info))


def torch_to_int8_onnx(
    fp32_model,
    int8_model,
    q_config,
    save_path,
    example_inputs,
    opset_version: int = 14,
    dynamic_axes: dict = {"input": {0: "batch_size"},
                          "output": {0: "batch_size"}},
    input_names=None,
    output_names=None,
    quant_format: str = 'QDQ',
    dtype: str = 'U8S8',
    recipe: str = 'QDQ_OP_FP32_BIAS',
):
    """Export INT8 PyTorch model into INT8 ONNX model.

    Args:
        fp32_model (torch.nn.module): fp32 model.
        int8_model (torch.nn.module): int8 model.
        q_config (dict): containing quantization configuration.
        save_path (str): save path of ONNX model.
        example_inputs (dict|list|tuple|torch.Tensor): used to trace torch model.
        opset_version (int, optional): opset version. Defaults to 14.
        dynamic_axes (dict, optional): dynamic axes. Defaults to {"input": {0: "batch_size"}, 
                                                                  "output": {0: "batch_size"}}.
        input_names (list, optional): input names. Defaults to None.
        output_names (list, optional): output names. Defaults to None.
        quant_format (str, optional): quantization format of ONNX model. Defaults to 'QDQ'.
        dtype (str, optional): data types of activation and weight of ONNX model. Defaults to 'U8S8'.
        recipe (str, optionl): Recipe for processing nn.quantized.Linear module. 
            'QDQ_OP_FP32_BIAS': inserting QDQ before quantizable op and using fp32 bias.
            'QDQ_OP_INT32_BIAS': inserting QDQ before quantizable op and using int32 bias.
            'QDQ_OP_FP32_BIAS_QDQ': inserting QDQ before and after quantizable op and using fp32 bias.
            Defaults to 'QDQ_OP_FP32_BIAS'.
    """
    global op_types_to_quantize
    if q_config['approach'] == 'post_training_dynamic_quant':
        op_types_to_quantize=['MatMul', 'Gemm', 'Gather']
    else:
        op_types_to_quantize=['MatMul', 'Gemm', 'Gather', 'Conv']

    if quant_format == 'QDQ' and opset_version < 13:   # pragma: no cover 
        opset_version = 13
        logger.warning("QDQ format requires opset_version >= 13, " + 
                        "we reset opset_version={} here".format(opset_version))

    # pylint: disable=E1101
    fp32_onnx_path = save_path + '.tmp' if save_path else 'int8-model.onnx.tmp'
    torch_to_fp32_onnx(
        fp32_model,
        fp32_onnx_path,
        example_inputs,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False,
    )

    activation_type, weight_type = set_data_type(dtype)
    module_node_mapping = get_node_mapping(fp32_model, fp32_onnx_path)
    quantize_nodes = get_quantizable_onnx_ops(int8_model, module_node_mapping)

    if q_config['approach'] == 'quant_aware_training':
        update_weight_bias(int8_model, fp32_onnx_path)
    if q_config['approach'] != 'post_training_dynamic_quant':
        int8_scale_info = q_config['scale_info']
        scale_mapping = build_scale_mapping(fp32_onnx_path, module_node_mapping, int8_scale_info)

    quant_format = ortq.QuantFormat.QOperator if quant_format != 'QDQ' else ortq.QuantFormat.QDQ

    extra_options = {'OpTypesToExcludeOutputQuantizatioin': ['MatMul']} \
        if (recipe != 'QDQ_OP_FP32_BIAS_QDQ' and quant_format == ortq.QuantFormat.QDQ) else {}

    if q_config['approach'] == 'post_training_dynamic_quant':
        logger.info("Quantization format is not avalible when executing dynamic quantization.")
        ortq.quantize_dynamic(
            fp32_onnx_path,
            save_path,
            per_channel=True,
            weight_type=weight_type,
            nodes_to_quantize=quantize_nodes,
            nodes_to_exclude=[],
            extra_options={}
        )

    else:
        from .utils import DummyDataReader
        dummy_datareader = DummyDataReader(fp32_onnx_path)
        ortq.quantize_static(
            fp32_onnx_path,
            save_path,
            dummy_datareader,
            quant_format=quant_format,
            per_channel=True,
            weight_type=weight_type,
            activation_type=activation_type,
            nodes_to_quantize=quantize_nodes,
            nodes_to_exclude=[],
            extra_options=extra_options,
        )

        int8_onnx_model = recalculate_bias(save_path, scale_mapping, quantize_nodes, quant_format)
        int8_onnx_model = set_scale_info(int8_onnx_model, scale_mapping, activation_type)

        if recipe == 'QDQ_OP_FP32_BIAS':
            int8_onnx_model = qdq_fp32_bias(int8_onnx_model, quant_format)
        elif recipe == 'QDQ_OP_INT32_BIAS':
            int8_onnx_model = qdq_int32_bias(int8_onnx_model, quantize_nodes, quant_format)
        elif recipe == 'QDQ_OP_FP32_BIAS_QDQ':
            int8_onnx_model = qdq_fp32_bias_qdq(int8_onnx_model, quantize_nodes, quant_format)
        
        onnx.save(int8_onnx_model, save_path)

    os.remove(fp32_onnx_path)
    info = "The INT8 ONNX Model is exported to path: {0}".format(save_path)
    logger.info("*"*len(info))
    logger.info(info)
    logger.info("*"*len(info))
