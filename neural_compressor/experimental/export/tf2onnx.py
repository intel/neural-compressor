#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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
"""Helper functions to export model from TensorFlow to ONNX."""

import re

from neural_compressor.utils import logger
from neural_compressor.utils.utility import LazyImport

t2o = LazyImport("tf2onnx")


def _split_nodename_and_shape(name):
    """Split input name with shape into name and shape."""
    # pattern for a node name
    inputs = []
    shapes = {}
    # input takes in most cases the format name:0, where 0 is the output number
    # in some cases placeholders don't have a rank which onnx can't handle so we let uses override the shape
    # by appending the same, ie : [1,28,28,3]
    name_pattern = r"(?:([\w\d/\-\._:]+)(\[[\-\d,]+\])?),?"
    splits = re.split(name_pattern, name)
    for i in range(1, len(splits), 3):
        inputs.append(splits[i] + ":0")
        if splits[i + 1] is not None:
            shape = [int(n) for n in splits[i + 1][1:-1].split(",")]
            shape = [n if n >= 0 else None for n in shape]
            shapes[splits[i] + ":0"] = shape
    if not shapes:
        shapes = None
    return inputs, shapes


def tf_to_fp32_onnx(graph_def, save_path, opset_version=14, input_names=None, output_names=None, inputs_as_nchw=None):
    """Export FP32 Tensorflow model into FP32 ONNX model using tf2onnx tool.

    Args:
        graph_def (graph_def to convert): fp32 graph_def.
        save_path (str): save path of ONNX model.
        opset_version (int, optional): opset version. Defaults to 14.
        input_names (list, optional): input names. Defaults to None.
        output_names (list, optional): output names. Defaults to None.
        inputs_as_nchw (list, optional): transpose the input. Defaults to None.
    """
    shape_override = None
    if isinstance(input_names, str):
        input_names, shape_override = _split_nodename_and_shape(input_names)
    else:
        input_names[:] = [o + ":0" for o in input_names]
    output_names[:] = [o + ":0" for o in output_names]
    t2o.convert.from_graph_def(
        graph_def=graph_def,
        input_names=input_names,
        output_names=output_names,
        inputs_as_nchw=inputs_as_nchw,
        shape_override=shape_override,
        opset=opset_version,
        output_path=save_path,
    )
    info = "The FP32 ONNX Model exported to path: {0}".format(save_path)
    logger.info("*" * len(info))
    logger.info(info)
    logger.info("*" * len(info))


def tf_to_int8_onnx(
    int8_model, save_path, opset_version: int = 14, input_names=None, output_names=None, inputs_as_nchw=None
):
    """Export INT8 Tensorflow model into INT8 ONNX model.

    Args:
        int8_model (tensorflow ITEX QDQ model): int8 model.
        save_path (str): save path of ONNX model.
        opset_version (int, optional): opset version. Defaults to 14.
        input_names (list, optional): input names. Defaults to None.
        output_names (list, optional): output names. Defaults to None.
        inputs_as_nchw (list, optional): transpose the input. Defaults to None.
    """
    shape_override = None
    if isinstance(input_names, str):
        input_names, shape_override = _split_nodename_and_shape(input_names)
    else:
        input_names[:] = [o + ":0" for o in input_names]
    output_names[:] = [o + ":0" for o in output_names]
    onnx_convert_graph = "./converted_graph.onnx"
    from neural_compressor.adaptor.tf_utils.tf2onnx_converter import TensorflowQDQToOnnxQDQConverter

    TensorflowQDQToOnnxQDQConverter(
        int8_model, input_names, output_names, shape_override, inputs_as_nchw, opset_version
    ).convert(onnx_convert_graph)

    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath = save_path
    import onnx

    model = onnx.load(onnx_convert_graph)
    ort.InferenceSession(model.SerializeToString(), sess_options)
    info = "The INT8 ONNX Model is exported to path: {0}".format(save_path)
    logger.info("*" * len(info))
    logger.info(info)
    logger.info("*" * len(info))
