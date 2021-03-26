#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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
#
from onnxruntime.quantization.quant_utils import QuantizationMode
from .operators.base_operator import QuantOperatorBase
from .operators.matmul import MatMulInteger, QLinearMatMul
from .operators.attention import AttentionQuant
from .operators.embed_layernorm import EmbedLayerNormalizationQuant
from .operators.gather import GatherQuant
from .operators.conv import QLinearConv, ConvInteger
from .operators.activation import QLinearActivation
from .operators.binary_op import QLinearBinaryOp
from .operators.maxpool import QMaxPool
from .operators.gavgpool import QGlobalAveragePool
from .operators.lstm import LSTMQuant
from .operators.split import QSplit
from .operators.pad import QPad

CommonOpsRegistry = {"Gather": GatherQuant, \
                     "EmbedLayerNormalization": EmbedLayerNormalizationQuant}

IntegerOpsRegistry = {
    "Conv": ConvInteger,
    "FusedConv": ConvInteger, 
    "MatMul": MatMulInteger,
    "Attention": AttentionQuant,
    "LSTM": LSTMQuant,
}
IntegerOpsRegistry.update(CommonOpsRegistry)

QLinearOpsRegistry = {
    "Conv": QLinearConv,
    "FusedConv": QLinearConv,
    "MatMul": QLinearMatMul,
    "Add": QLinearBinaryOp,
    "Mul": QLinearBinaryOp,
    "Relu": QLinearActivation,
    "Clip": QLinearActivation,
    "LeakyRelu" : QLinearActivation,
    "Sigmoid" : QLinearActivation,
    "MaxPool": QMaxPool,
    "GlobalAveragePool": QGlobalAveragePool,
    "Split": QSplit,
    "Pad": QPad
}
QLinearOpsRegistry.update(CommonOpsRegistry)


def CreateDefaultOpQuantizer(onnx_quantizer, node):
    return QuantOperatorBase(onnx_quantizer, node)


def CreateOpQuantizer(onnx_quantizer, node):
    registry = IntegerOpsRegistry if onnx_quantizer.mode == QuantizationMode.IntegerOps \
                                  else QLinearOpsRegistry
    if node.op_type in registry.keys():
        return registry[node.op_type](onnx_quantizer, node)
    return QuantOperatorBase(onnx_quantizer, node)
