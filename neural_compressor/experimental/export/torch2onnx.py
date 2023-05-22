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
import sys
import numpy as np
from collections import UserDict
from neural_compressor.adaptor.torch_utils.util import input2tuple
from neural_compressor.utils import logger
from neural_compressor.utils.utility import LazyImport

torch = LazyImport('torch')
onnx = LazyImport('onnx')
ort = LazyImport('onnxruntime')
ortq = LazyImport('onnxruntime.quantization')

def _prepare_intputs(pt_model, input_names, example_inputs):
    """prepare input_names and example_inputs"""
    if input_names is None and \
      (isinstance(example_inputs, dict) or isinstance(example_inputs, UserDict)):
        input_names = list(example_inputs.keys())
        example_inputs = list(example_inputs.values())
    elif isinstance(example_inputs, dict) or isinstance(example_inputs, UserDict):
        example_inputs = list(example_inputs.values())
    # match input_names with inspected input_order, especailly for bert in hugginface.
    if input_names and len(input_names) > 1:
        import inspect
        input_order = inspect.signature(pt_model.forward).parameters.keys()
        flag = [name in input_order for name in input_names] # whether should be checked
        if all(flag):
            new_input_names = []
            new_example_inputs = []
            for name in input_order:
                if name in input_names:
                    new_input_names.append(name)
                    id = input_names.index(name)
                    new_example_inputs.append(example_inputs[id])
            input_names = new_input_names
            example_inputs = new_example_inputs
    return input_names, example_inputs


def torch_to_fp32_onnx(
    pt_model,
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
        pt_model (torch.nn.module): PyTorch model.
        save_path (str): save path of ONNX model.
        example_inputs (dict|list|tuple|torch.Tensor): used to trace torch model.
        opset_version (int, optional): opset version. Defaults to 14.
        dynamic_axes (dict, optional): dynamic axes. Defaults to {"input": {0: "batch_size"}, "output": {0: "batch_size"}}.
        input_names (dict, optional): input names. Defaults to None.
        output_names (dict, optional): output names. Defaults to None.
        do_constant_folding (bool, optional): do constant folding or not. Defaults to True.
        verbose (bool, optional): dump verbose or not. Defaults to True.
    """
    from neural_compressor.utils.pytorch import is_int8_model
    assert is_int8_model(pt_model) == False, "The fp32 model is replaced during quantization. " + \
        "please customize a eval_func when quantizing, if not, such as `lambda x: 1`."
    
    input_names, example_inputs = _prepare_intputs(pt_model, input_names, example_inputs)

    with torch.no_grad():
        torch.onnx.export(
            pt_model,
            input2tuple(example_inputs),
            save_path, 
            opset_version=opset_version,
            input_names=input_names,
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
    pt_model,
    save_path,
    example_inputs,
    q_config,
    opset_version=14,
    dynamic_axes={"input": {0: "batch_size"},
                  "output": {0: "batch_size"}},
    input_names=None,
    output_names=None,
    quant_format: str = 'QDQ',
    verbose=True,
):
    """Export INT8 PyTorch model into INT8 ONNX model.

    Args:
        pt_model (torch.nn.module): PyTorch model.
        save_path (str): save path of ONNX model.
        example_inputs (dict|list|tuple|torch.Tensor): used to trace torch model.
        q_config (dict): containing quantization configuration.
        opset_version (int, optional): opset version. Defaults to 14.
        dynamic_axes (dict, optional): dynamic axes. Defaults to {"input": {0: "batch_size"}, "output": {0: "batch_size"}}.
        input_names (dict, optional): input names. Defaults to None.
        output_names (dict, optional): output names. Defaults to None.
        quant_format (str, optional): _quantization format of ONNX model. Defaults to 'QDQ'.
        verbose (bool, optional): dump verbose or not. Defaults to True.
    """
    from neural_compressor.utils.pytorch import is_int8_model
    assert is_int8_model(pt_model), "The exported model is not INT8 model, "\
        "please reset 'dtype' to 'FP32' or check your model."
    
    assert not q_config is None, "'q_config' is needed when export an INT8 model."

    if q_config['approach'] == 'post_training_dynamic_quant':
        assert False, "Post training dynamic quantizated PyTorch model is not supported to export to ONNX. " \
        "Please follow this step to get a post training dynamic quantizated PyTorch model: " \
        "1. export FP32 PyTorch model to FP32 ONNX model. " \
        "2. use FP32 ONNX model as input model to do post training dynamic quantizatation."

    input_names, example_inputs = _prepare_intputs(pt_model, input_names, example_inputs)

    def model_wrapper(model_fn):
        # export doesn't support a dictionary output, so manually turn it into a tuple
        # refer to https://discuss.tvm.apache.org/t/how-to-deal-with-prim-dictconstruct/11978
        def wrapper(*args, **kwargs):
            output = model_fn(*args, **kwargs)
            if isinstance(output, dict):
                return tuple(v for v in output.values() if v is not None)
            else:
                return output
        return wrapper
    pt_model.forward = model_wrapper(pt_model.forward)

    with torch.no_grad():
        try:
            torch.onnx.export(
                pt_model,
                input2tuple(example_inputs),
                save_path, 
                opset_version=opset_version,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                )
        except Exception as e:
            config_name = "QuantizationAwareTrainingConfig" \
                if q_config['approach'] == "quant_aware_training" else "PostTrainingQuantConfig"
            logger.error("Export failed, possibly because unsupported quantized ops. Check " 
                         "neural-compressor/docs/source/export.md#supported-quantized-ops "
                         "for supported ops.")
            logger.error("Please fallback unsupported quantized ops by setting 'op_type_dict' or "
                         "'op_name_dict' in '{}' config. ".format(config_name))
            return
            
    if quant_format != "QDQ":
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess_options.optimized_model_filepath=save_path
        ort.InferenceSession(save_path, sess_options)
    
    if verbose:
        info = "The INT8 ONNX Model exported to path: {0}".format(save_path)
        logger.info("*"*len(info))
        logger.info(info)
        logger.info("*"*len(info))