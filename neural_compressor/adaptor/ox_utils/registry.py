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
from .operators.qdq_base_operator import QDQOperatorBase
from .operators.matmul import MatMulInteger, QLinearMatMul, QDQMatMul
from .operators.attention import AttentionQuant, QDQAttention
from .operators.embed_layernorm import EmbedLayerNormalizationQuant
from .operators.gather import GatherConverter, GatherQuant
from .operators.conv import QLinearConv, ConvInteger, QDQConv
from .operators.activation import QLinearActivation, QDQRemovableActivation, QDQActivation
from .operators.binary_op import QLinearBinaryOp, QDQBinaryOp
from .operators.maxpool import QMaxPool, QDQMaxPool
from .operators.gavgpool import QGlobalAveragePool
from .operators.lstm import LSTMQuant, QDQLSTM
from .operators.split import QSplit, QDQSplit
from .operators.concat import QLinearConcat, QDQConcat
from .operators.pad import QPad, QDQPad
from .operators.pooling import QLinearPool, QDQPool
from .operators.direct_q8 import QDQDirect8BitOp, Direct8BitOp, DirectCast
from .operators.base_operator_cast import CastOperatorBase

CommonOpsRegistry = {"Gather": GatherConverter, \
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
    "Concat": QLinearConcat,
    "Attention": AttentionQuant,
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
    "Pad": QPad,
    "AveragePool" : QLinearPool,
    "Reshape": Direct8BitOp,
    "Transpose" : Direct8BitOp,
    "Squeeze" : Direct8BitOp,
    "Unsqueeze" : Direct8BitOp,
}
QLinearOpsRegistry.update(CommonOpsRegistry)

QDQRegistry = {
    "FusedConv": QDQConv,
    "Conv": QDQConv,
    "Clip": QDQRemovableActivation,
    "Relu": QDQRemovableActivation,
    "LeakyRelu": QDQActivation,
    "Sigmoid": QDQActivation,
    "MaxPool": QDQMaxPool,
    "MatMul": QDQMatMul,
    "Add": QDQBinaryOp,
    "Mul": QDQBinaryOp,
    "Gather": GatherQuant,
    "Attention": QDQAttention,
    "LSTM": QDQLSTM,
    "Pad": QDQPad,
    "Reshape": QDQDirect8BitOp,
    "Transpose" : QDQDirect8BitOp,
    "Squeeze" : QDQDirect8BitOp,
    "AveragePool": QDQPool,
    "Unsqueeze" : QDQDirect8BitOp,
    "Concat": QDQConcat,
    "Split": QDQSplit
}

CastRegistry = {
    "Shape": DirectCast,
    "Squeeze": DirectCast,
    "Unsqueeze": DirectCast,
    "Reshape": DirectCast,
    "Unsqueeze": DirectCast,
    "Transpose": DirectCast,
    "Loop": DirectCast,
    "Slice": DirectCast,
    "Split": DirectCast,
    "Concat": DirectCast,

}


def CreateOpConverter(onnx_quantizer, node):
    registry = IntegerOpsRegistry if onnx_quantizer.mode == QuantizationMode.IntegerOps \
                                  else QLinearOpsRegistry
    if node.op_type in registry.keys():
        return registry[node.op_type](onnx_quantizer, node)
    return QuantOperatorBase(onnx_quantizer, node)

def CreateQDQQuantizer(onnx_quantizer, node):
    if node.op_type in QDQRegistry.keys():
        return QDQRegistry[node.op_type](onnx_quantizer, node)
    return QDQOperatorBase(onnx_quantizer, node)

def CreateCaster(onnx_quantizer, node):
    if node.op_type in CastRegistry.keys():
        return CastRegistry[node.op_type](onnx_quantizer, node)
    return CastOperatorBase(onnx_quantizer, node)
