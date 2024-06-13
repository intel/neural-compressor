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
from collections import UserDict

from neural_compressor.adaptor.torch_utils.util import input2tuple
from neural_compressor.utils import logger
from neural_compressor.utils.utility import LazyImport

torch = LazyImport("torch")
onnx = LazyImport("onnx")
ort = LazyImport("onnxruntime")
ortq = LazyImport("onnxruntime.quantization")


def _prepare_inputs(pt_model, input_names, example_inputs):
    """Prepare input_names and example_inputs."""
    if isinstance(example_inputs, dict) or isinstance(example_inputs, UserDict):
        input_names = input_names or list(example_inputs.keys())
        if isinstance(example_inputs, UserDict):
            example_inputs = dict(example_inputs)
    # match input_names with inspected input_order, especially for bert in hugginface.
    elif input_names and len(input_names) > 1:
        import inspect

        input_order = inspect.signature(pt_model.forward).parameters.keys()
        flag = [name in input_order for name in input_names]  # whether should be checked
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
        example_inputs = input2tuple(example_inputs)
    return input_names, example_inputs


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
        return None

    module_dict = {}
    for name, module in fp32_model.named_modules():
        if (
            "Conv" in str(module.__class__.__name__)
            or "Embedding" in str(module.__class__.__name__)
            or "Linear" in str(module.__class__.__name__)
        ):
            if hasattr(module, "weight"):
                value = module.weight.detach().cpu().numpy()
                module_dict[name] = value

    module_node_mapping = {}
    fp32_onnx_model = onnx.load(fp32_onnx_path)
    initializer_data = {tensor.name: tensor for tensor in fp32_onnx_model.graph.initializer}
    from onnx import numpy_helper

    for node in fp32_onnx_model.graph.node:
        if node.op_type in op_types_to_quantize:
            if node.op_type == "MatMul" and node.input[1] in initializer_data:
                data = numpy_helper.to_array(initializer_data[node.input[1]]).T
            elif node.op_type == "Gather" and node.input[0] in initializer_data:
                data = numpy_helper.to_array(initializer_data[node.input[0]])
            elif node.op_type in ["Gemm"]:
                data = numpy_helper.to_array(initializer_data[node.input[1]])
            else:  # pragma: no cover
                continue
            pt_name = check_data(node.op_type, data, module_dict)
            if pt_name:
                module_node_mapping[pt_name] = node.name
    return module_node_mapping


def get_quantizable_onnx_ops(int8_model, module_node_mapping):
    """Get quantizable onnx ops.

    Args:
        int8_model (torch.nn.Module): PyTorch int8 model.
        module_node_mapping (dict): op mapping from PyTorch to ONNX.

    Returns:
        quantize_nodes: all onnx node that should be quantized.
    """
    quantize_nodes = []
    for name, module in int8_model.named_modules():
        if (
            "Conv" in str(module.__class__.__name__)
            or "Embedding" in str(module.__class__.__name__)
            or "Linear" in str(module.__class__.__name__)
        ):
            if hasattr(module, "weight") and callable(module.weight):
                if module.weight().dtype in [torch.qint8, torch.quint8]:
                    if name.split(".module")[0] in module_node_mapping:
                        node = module_node_mapping[name.split(".module")[0]]
                        quantize_nodes.append(node)
    return quantize_nodes


def dynamic_quant_export(
    pt_fp32_model,
    pt_int8_model,
    save_path,
    example_inputs,
    q_config,
    opset_version,
    dynamic_axes,
    input_names,
    output_names,
    weight_type,
):
    """Export dynamic quantized model.

    Args:
        pt_fp32_model (torch.nn.module): PyTorch FP32 model.
        pt_int8_model (torch.nn.module): PyTorch INT8 model.
        save_path (str): save path of ONNX model.
        example_inputs (dict|list|tuple|torch.Tensor): used to trace torch model.
        q_config (dict): containing quantization configuration.
        opset_version (int, optional): opset version. Defaults to 14.
        dynamic_axes (dict, optional): dynamic axes. Defaults to
            {"input": {0: "batch_size"}, "output": {0: "batch_size"}}.
        input_names (dict, optional): input names. Defaults to None.
        output_names (dict, optional): output names. Defaults to None.
        weight_type (str, optional): data types of weight of ONNX model
            (only needed for exporting dynamic quantized model). Defaults to 'S8'.
    """
    global op_types_to_quantize
    op_types_to_quantize = ["MatMul", "Gemm", "Gather"]

    # pylint: disable=E1101
    fp32_onnx_path = save_path + ".tmp" if save_path else "int8-model.onnx.tmp"
    torch_to_fp32_onnx(
        pt_fp32_model,
        fp32_onnx_path,
        example_inputs,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False,
    )

    module_node_mapping = get_node_mapping(pt_fp32_model, fp32_onnx_path)
    quantize_nodes = get_quantizable_onnx_ops(pt_int8_model, module_node_mapping)

    REDUCE_RANGE = q_config["reduce_range"]
    if REDUCE_RANGE:
        logger.info("Reduce range is {}".format(str(REDUCE_RANGE)))

    logger.info("Quantization format is not available when executing dynamic quantization.")

    if weight_type.upper() == "S8":
        weight_type = ortq.QuantType.QInt8
    elif weight_type.upper() == "U8":
        weight_type = ortq.QuantType.QUInt8
    else:
        assert False, "Right now, we don't support weight type: {}, " "please use S8/U8.".format(weight_type)

    ortq.quantize_dynamic(
        fp32_onnx_path,
        save_path,
        per_channel=True,
        reduce_range=REDUCE_RANGE,
        weight_type=weight_type,
        nodes_to_quantize=quantize_nodes,
        nodes_to_exclude=[],
        extra_options={},
    )

    os.remove(fp32_onnx_path)


def static_quant_export(
    pt_int8_model,
    save_path,
    example_inputs,
    q_config,
    opset_version,
    dynamic_axes,
    input_names,
    output_names,
    quant_format,
):
    """Export static quantized model.

    Args:
        pt_int8_model (torch.nn.module): PyTorch INT8 model.
        save_path (str): save path of ONNX model.
        example_inputs (dict|list|tuple|torch.Tensor): used to trace torch model.
        q_config (dict): containing quantization configuration.
        opset_version (int, optional): opset version. Defaults to 14.
        dynamic_axes (dict, optional): dynamic axes. Defaults to
            {"input": {0: "batch_size"}, "output": {0: "batch_size"}}.
        input_names (dict, optional): input names. Defaults to None.
        output_names (dict, optional): output names. Defaults to None.
        quant_format (str, optional): _quantization format of ONNX model. Defaults to 'QDQ'.
    """
    input_names, example_inputs = _prepare_inputs(pt_int8_model, input_names, example_inputs)

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

    pt_int8_model.forward = model_wrapper(pt_int8_model.forward)

    with torch.no_grad():
        try:
            torch.onnx.export(
                pt_int8_model,
                input2tuple(example_inputs),
                save_path,
                opset_version=opset_version,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )
        except TypeError:
            config_name = (
                "QuantizationAwareTrainingConfig"
                if q_config["approach"] == "quant_aware_training"
                else "PostTrainingQuantConfig"
            )
            logger.error(
                "Export failed, possibly because unsupported quantized ops. Check "
                "neural-compressor/docs/source/export.md#supported-quantized-ops "
                "for supported ops."
            )
            logger.error(
                "Please fallback unsupported quantized ops by setting 'op_type_dict' or "
                "'op_name_dict' in '{}' config. ".format(config_name)
            )
            raise TypeError("Export failed with TypeError.")
        except Exception as e:
            raise e

    if quant_format != "QDQ":
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess_options.optimized_model_filepath = save_path
        ort.InferenceSession(save_path, sess_options)


def torch_to_fp32_onnx(
    pt_fp32_model,
    save_path,
    example_inputs,
    opset_version=14,
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    input_names=None,
    output_names=None,
    do_constant_folding=True,
    verbose=True,
):
    """Export FP32 PyTorch model into FP32 ONNX model.

    Args:
        pt_fp32_model (torch.nn.module): PyTorch FP32 model.
        save_path (str): save path of ONNX model.
        example_inputs (dict|list|tuple|torch.Tensor): used to trace torch model.
        opset_version (int, optional): opset version. Defaults to 14.
        dynamic_axes (dict, optional): dynamic axes. Defaults to
            {"input": {0: "batch_size"}, "output": {0: "batch_size"}}.
        input_names (dict, optional): input names. Defaults to None.
        output_names (dict, optional): output names. Defaults to None.
        do_constant_folding (bool, optional): do constant folding or not. Defaults to True.
        verbose (bool, optional): dump verbose or not. Defaults to True.
    """
    from neural_compressor.utils.pytorch import is_int8_model

    assert is_int8_model(pt_fp32_model) is False, (
        "The fp32 model is replaced during quantization. "
        + "please customize a eval_func when quantizing, if not, such as `lambda x: 1`."
    )

    input_names, example_inputs = _prepare_inputs(pt_fp32_model, input_names, example_inputs)

    with torch.no_grad():
        torch.onnx.export(
            pt_fp32_model,
            example_inputs,
            save_path,
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=do_constant_folding,
        )

    if verbose:
        info = "The FP32 ONNX Model exported to path: {0}".format(save_path)
        logger.info("*" * len(info))
        logger.info(info)
        logger.info("*" * len(info))


def torch_to_int8_onnx(
    pt_fp32_model,
    pt_int8_model,
    save_path,
    example_inputs,
    q_config,
    opset_version=14,
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    input_names=None,
    output_names=None,
    quant_format: str = "QDQ",
    weight_type: str = "S8",
    verbose=True,
):
    """Export INT8 PyTorch model into INT8 ONNX model.

    Args:
        pt_fp32_model (torch.nn.module): PyTorch FP32 model.
        pt_int8_model (torch.nn.module): PyTorch INT8 model.
        save_path (str): save path of ONNX model.
        example_inputs (dict|list|tuple|torch.Tensor): used to trace torch model.
        q_config (dict): containing quantization configuration.
        opset_version (int, optional): opset version. Defaults to 14.
        dynamic_axes (dict, optional): dynamic axes. Defaults to
            {"input": {0: "batch_size"}, "output": {0: "batch_size"}}.
        input_names (dict, optional): input names. Defaults to None.
        output_names (dict, optional): output names. Defaults to None.
        quant_format (str, optional): _quantization format of ONNX model. Defaults to 'QDQ'.
        weight_type (str, optional): data types of weight of ONNX model
            (only needed for exporting dynamic quantized model). Defaults to 'S8'.
        verbose (bool, optional): dump verbose or not. Defaults to True.
    """
    from neural_compressor.utils.pytorch import is_int8_model

    assert is_int8_model(pt_int8_model), (
        "The exported model is not INT8 model, " "please reset 'dtype' to 'FP32' or check your model."
    )

    assert q_config is not None, "'q_config' is needed when export an INT8 model."

    quant_format = quant_format.upper()
    if quant_format == "QDQ" and opset_version < 13:  # pragma: no cover
        opset_version = 13
        logger.warning(
            "QDQ format requires opset_version >= 13, " + "we reset opset_version={} here".format(opset_version)
        )

    if q_config["approach"] == "post_training_dynamic_quant":
        # dynamic quantization export follow these steps:
        # "1. export FP32 PyTorch model to FP32 ONNX model. "
        # "2. use FP32 ONNX model as the input model for post training dynamic quantization."
        # TODO: will be removed once torch supports dynamic quantization export
        dynamic_quant_export(
            pt_fp32_model,
            pt_int8_model,
            save_path,
            example_inputs,
            q_config,
            opset_version,
            dynamic_axes,
            input_names,
            output_names,
            weight_type,
        )
    else:
        static_quant_export(
            pt_int8_model,
            save_path,
            example_inputs,
            q_config,
            opset_version,
            dynamic_axes,
            input_names,
            output_names,
            quant_format,
        )

    if verbose:
        info = "The INT8 ONNX Model exported to path: {0}".format(save_path)
        logger.info("*" * len(info))
        logger.info(info)
        logger.info("*" * len(info))
