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

from neural_compressor.utils import logger
import tf2onnx as t2o
from neural_compressor.utils.utility import LazyImport


def tf_to_fp32_onnx(
    graph_def,
    save_path,
    opset_version=14,
    input_names=None,
    output_names=None,
    inputs_as_nchw=None
):
    """Export FP32 Tensorflow model into FP32 ONNX model using tf2onnx tool.

    Args:
        graph_def (graph_def to convert): fp32 graph_def.
        save_path (str): save path of ONNX model.
        opset_version (int, optional): opset version. Defaults to 14.
        input_names (list, optional): input names. Defaults to None.
        output_names (list, optional): output names. Defaults to None.
    """
    input_names[:] = [i+":0" for i in input_names]
    output_names[:] = [o+":0" for o in output_names]
    t2o.convert.from_graph_def(graph_def=graph_def, input_names=input_names,
                      output_names=output_names, inputs_as_nchw=inputs_as_nchw,
                      opset=opset_version, output_path=save_path)
    info = "The FP32 ONNX Model exported to path: {0}".format(save_path)
    logger.info("*"*len(info))
    logger.info(info)
    logger.info("*"*len(info))


def tf_to_int8_onnx(
    int8_model,
    save_path,
    opset_version: int = 14,
    input_names=None,
    output_names=None,
    inputs_as_nchw=None
):
    """Export INT8 Tensorflow model into INT8 ONNX model.

    Args:
        int8_model (tensorflow ITEX QDQ model): int8 model.
        save_path (str): save path of ONNX model.
        opset_version (int, optional): opset version. Defaults to 14.
        input_names (list, optional): input names. Defaults to None.
        output_names (list, optional): output names. Defaults to None.
    """
    from neural_compressor.adaptor.tf_utils.tf2onnx_converter import TensorflowQDQToOnnxQDQConverter
    TensorflowQDQToOnnxQDQConverter(int8_model, input_names, \
                        output_names, inputs_as_nchw, opset_version).convert(save_path)

    info = "The INT8 ONNX Model is exported to path: {0}".format(save_path)
    logger.info("*"*len(info))
    logger.info(info)
    logger.info("*"*len(info))
