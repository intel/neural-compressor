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

"""Helper functions to export model from PyTorch/TensorFlow to ONNX"""
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


class DummyDataReader(ortq.CalibrationDataReader):
    def __init__(self, fp32_onnx_path):
        session = ort.InferenceSession(fp32_onnx_path, None)
        input_tensors = session.get_inputs()
        input = {}
        for node in input_tensors:
            shape = []
            for dim in node.shape:
                shape.append(dim if isinstance(dim, int) else 1)
            tmp = node.type.lstrip('tensor(').rstrip(')')
            dtype = eval(f'np.{tmp}')
            input[node.name] = np.ones(shape).astype(dtype)
        self.data = [input]
        self.data = iter(self.data)
    def get_next(self):
        return next(self.data, None)

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
    """set data type of activation and weight with string dtype

    Args:
        dtype (str): data type description
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

def get_quantizable_onnx_ops(
    q_config,
    fp32_onnx_path,
):
    fp32_onnx_model = onnx.load(fp32_onnx_path)
    # Clarify ONNX nodes that we can mapping from PyTorch
    if 'dynamic' in q_config['approach']:
        op_types_to_quantize=['MatMul', 'Gather', "LSTM"]
        pytorch_op_types_to_quantize=['Linear', 'Embedding', "LSTM"]
    else:
        op_types_to_quantize=['MatMul', 'Gather', 'Conv']
        pytorch_op_types_to_quantize=['Linear', 'Embedding', 'Conv1d', 'Conv2d']

    addition_op_to_quantize = []

    # if 'U8S8' in dtype:
    #     op_types_to_quantize.remove('Gather')
    #     pytorch_op_types_to_quantize.remove('Embedding')

    all_op_types_to_quantize = op_types_to_quantize + addition_op_to_quantize

    from neural_compressor.adaptor.onnxrt import ONNXRTAdaptor
    # pylint: disable=E1120
    fp32_onnx_model = ONNXRTAdaptor._replace_gemm_with_matmul(fp32_onnx_model).model
    onnx.save(fp32_onnx_model, fp32_onnx_path)

    # Get weight name from onnx initializer
    weight_name_list = []
    for tensor in fp32_onnx_model.graph.initializer:
        weight_name_list.append(tensor.name)

    # Match weight name with onnx node name
    quantize_nodes = []
    tmp_node_mapping = {}
    module_node_mapping = {}
    for node in fp32_onnx_model.graph.node:
        if node.op_type not in op_types_to_quantize:
            for inp in node.input:
                if inp in weight_name_list and 'weight' in inp:
                    tmp_node_mapping.update({node.output[0] : inp.split('.weight')[0]})
                elif inp in tmp_node_mapping:
                    tmp_node_mapping.update({node.output[0] : tmp_node_mapping[inp]})
        else:
            for inp in node.input:
                if inp in weight_name_list and 'weight' in inp:
                    module_node_mapping.update({inp.split('.weight')[0] : node.name})
                elif inp in tmp_node_mapping:
                    module_node_mapping.update({tmp_node_mapping[inp]: node.name})

        # Save all quantizable node name
        if node.op_type in all_op_types_to_quantize:
            quantize_nodes.append(node.name)

    # Match pytorch module name with onnx node name for fallbacked fp32 module
    for k, v in q_config['op'].items():   # pragma: no cover
        if k[1] not in pytorch_op_types_to_quantize or 'int8' in v['weight']['dtype']:
            continue
        k_0 = k[0].split('.module')[0] if k[0] not in module_node_mapping else k[0]
        if k_0 in module_node_mapping:
            fallback_op = module_node_mapping[k_0]
            quantize_nodes.remove(fallback_op)
    return quantize_nodes, module_node_mapping



def get_scale_info(
    int8_model,
    q_config,
):
    # get output scale and zp from module
    int8_scale_info = {}
    import torch.nn.quantized.modules as q_modules
    for name, module in int8_model.named_modules():
        if isinstance(module, q_modules.Conv1d) or \
            isinstance(module, q_modules.Conv2d) or \
            isinstance(module, q_modules.Linear):
            int8_scale_info[name] = {
                'output_scale': module.scale,
                'output_zeropoint': module.zero_point,
            }

    # a name mapping to avoid '_' and '.' mismatch, we only use '.'.
    new_name_mapping = {}
    for name in int8_scale_info.keys():
        new_name = name.replace("_", '.')
        new_name_mapping.update({new_name: name})

    # get input scale and zp from q_config
    for name, value in q_config['get_attr'].items():
        node_name, node_target = name.split('--')
        if 'scale' in name:
            value_dict = {'input_scale': value}
        if 'zero_point' in name:
            value_dict = {'input_zeropoint': value}
        if node_name:
            node_name = node_name + '.'
        if '_input_' in node_target:
            tmp_name = node_name + node_target.split('_input_')[0]
            tmp_name = tmp_name.replace("_", '.')
        # avoid layernorm from qat.
        if tmp_name in new_name_mapping:
            node_name = new_name_mapping[tmp_name]
            int8_scale_info[node_name].update(value_dict)
    return int8_scale_info

def build_scale_mapping(
    fp32_onnx_path,
    module_node_mapping,
    int8_scale_info,
):
    node_module_mapping = {}
    for module_name, node_name in module_node_mapping.items():
        node_module_mapping[node_name] = module_name
    # match scale and zeropoint from PyTorch to ONNX node
    scale_zp_dict = {}
    fp32_onnx_model = onnx.load(fp32_onnx_path)
    for node in fp32_onnx_model.graph.node:
        if node.name in node_module_mapping:
            module_name = node_module_mapping[node.name]
            if module_name not in int8_scale_info:
                module_name = module_name + '.module'
            if module_name in int8_scale_info:
                recoder = int8_scale_info[module_name]
                input_scale_args = node.input[0] + '_scale'
                input_zp_args = node.input[0] + '_zero_point'
                scale_zp_dict[input_scale_args] = recoder['input_scale']
                scale_zp_dict[input_zp_args] = recoder['input_zeropoint']
                ### We need Matmul+Add to match Linear for output scale and zero-point
                # output_scale_args = node.output[0] + '_scale'
                # output_zp_args = node.output[0] + '_zero_point'
                # scale_zp_dict[output_scale_args] = recoder['output_scale']
                # scale_zp_dict[output_zp_args] = recoder['output_zeropoint']
    return scale_zp_dict

def set_scale_info(
    int8_onnx_path,
    scale_zp_dict,
    activation_type,
):
    # set scale and zeropoint from PyTorch int8 model to ONNX int8 model
    from onnx import helper
    int8_onnx_model = onnx.load(int8_onnx_path)
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
    onnx.save(int8_onnx_model, int8_onnx_path)

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
):
    """Export INT8 PyTorch model into INT8 ONNX model

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
    """
    if quant_format == 'QDQ' and opset_version < 13:   # pragma: no cover 
        opset_version = 13
        logger.warning("QDQ format requires opset_version >= 13, " + 
                        "we reset opset_version={} here".format(opset_version))

    activation_type, weight_type = set_data_type(dtype)

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
        do_constant_folding=False,
        verbose=False,
    )

    quantize_nodes, module_node_mapping = get_quantizable_onnx_ops(q_config, fp32_onnx_path)

    if q_config['approach'] == 'quant_aware_training':
        update_weight_bias(int8_model, fp32_onnx_path)
    if q_config['approach'] != 'post_training_dynamic_quant':
        int8_scale_info = get_scale_info(int8_model, q_config)
        scale_mapping = build_scale_mapping(fp32_onnx_path, module_node_mapping, int8_scale_info)

    quant_format = ortq.QuantFormat.QOperator if quant_format != 'QDQ' else ortq.QuantFormat.QDQ

    if q_config['approach'] == 'post_training_dynamic_quant':
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
            extra_options={'OpTypesToExcludeOutputQuantizatioin': ['MatMul']},
        )

        set_scale_info(save_path, scale_mapping, activation_type)

    os.remove(fp32_onnx_path)
    info = "The INT8 ONNX Model is exported to path: {0}".format(save_path)
    logger.info("*"*len(info))
    logger.info(info)
    logger.info("*"*len(info))
