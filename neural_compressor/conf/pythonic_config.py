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

import logging
from .dotdict import DotDict
from ..config import _BaseQuantizationConfig, accuracy_criterion, BenchmarkConfig, \
                     check_value, DistillationConfig, options, WeightPruningConfig

logger = logging.getLogger("neural_compressor")


class QuantizationConfig(_BaseQuantizationConfig):
    def __init__(self,
                 inputs=[],
                 outputs=[],
                 backend='default',
                 device='cpu',
                 approach='post_training_static_quant',
                 calibration_sampling_size=[100],
                 op_type_list=None,
                 op_name_list=None,
                 strategy='basic',
                 strategy_kwargs=None,
                 objective='performance',
                 timeout=0,
                 max_trials=100,
                 performance_only=False,
                 reduce_range=None,
                 use_bf16=True,
                 quant_level=1,
                 accuracy_criterion=accuracy_criterion):
        excluded_precisions = ["bf16"] if not use_bf16 else []
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            backend=backend,
            device=device,
            calibration_sampling_size=calibration_sampling_size,
            op_type_list=op_type_list,
            op_name_list=op_name_list,
            strategy=strategy,
            strategy_kwargs=strategy_kwargs,
            objective=objective,
            timeout=timeout,
            max_trials=max_trials,
            performance_only=performance_only,
            reduce_range=reduce_range,
            excluded_precisions=excluded_precisions,
            accuracy_criterion=accuracy_criterion,
            quant_level=quant_level
        )
        self._approach = approach

    @property
    def approach(self):
        return self._approach

    @approach.setter
    def approach(self, approach):
        if check_value(
            'approach', approach, str,
            ['post_training_static_quant', 'post_training_dynamic_quant', 'quant_aware_training']
        ):
            self._approach = approach


class WeightConf:
    def __init__(self, datatype=None, scheme=None, granularity=None, algorithm=None):
        self._datatype = datatype
        self._scheme = scheme
        self._granularity = granularity
        self._algorithm = algorithm

    @property
    def datatype(self):
        return self._datatype

    @datatype.setter
    def datatype(self, datatype):
        if check_value('datatype', datatype, list, ['fp32', 'bf16', 'uint8', 'int8']):
            self._datatype = datatype

    @property
    def scheme(self):
        return self._scheme

    @scheme.setter
    def scheme(self, scheme):
        if check_value('scheme', scheme, list, ['sym', 'asym']):
            self._scheme = scheme

    @property
    def granularity(self):
        return self._granularity

    @granularity.setter
    def granularity(self, granularity):
        if check_value('granularity', granularity, list, ['per_channel', 'per_tensor']):
            self._granularity = granularity

    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, algorithm):
        if check_value('algorithm', algorithm, list, ['minmax', 'kl']):
            self._algorithm = algorithm

class ActivationConf(WeightConf):
    def __init__(self, datatype=None, scheme=None, granularity=None, algorithm=None):
        super().__init__(datatype, scheme, granularity, algorithm)

weight = WeightConf()
activation = ActivationConf()

class OpQuantConf:
    def __init__(self, op_type=None, weight=weight, activation=activation):
        self._op_type = op_type
        self._weight = weight
        self._activation = activation

    @property
    def op_type(self):
        return self._op_type

    @op_type.setter
    def op_type(self, op_type):
        if check_value('op_type', op_type, str):
            self._op_type = op_type

    @property
    def weight(self):
        return self._weight

    @property
    def activation(self):
        return self._activation

class MXNet:
    def __init__(self, precisions=None):
        self._precisions = precisions

    @property
    def precisions(self):
        return self._precisions

    @precisions.setter
    def precisions(self, precisions):
        if not isinstance(precisions, list):
            precisions = [precisions]
        if check_value('precisions', precisions, str, ['int8', 'uint8', 'fp32', 'bf16']):
            self._precisions = precisions

class ONNX(MXNet):
    def __init__(self, graph_optimization_level=None, precisions=None):
        super().__init__(precisions)
        self._graph_optimization_level = graph_optimization_level

    @property
    def graph_optimization_level(self):
        return self._graph_optimization_level

    @graph_optimization_level.setter
    def graph_optimization_level(self, graph_optimization_level):
        if check_value('graph_optimization_level', graph_optimization_level, str,
            ['DISABLE_ALL', 'ENABLE_BASIC', 'ENABLE_EXTENDED', 'ENABLE_ALL']):
            self._graph_optimization_level = graph_optimization_level

class TensorFlow(MXNet):
    def __init__(self, precisions=None):
        super().__init__(precisions)

class Keras(MXNet):
    def __init__(self, precisions=None):
        super().__init__(precisions)

class PyTorch(MXNet):
    def __init__(self, precisions=None):
        super().__init__(precisions)


class DyNASConfig:
    def __init__(self, supernet=None, metrics=None, population=50, num_evals=100000,
                 results_csv_path=None, dataset_path=None, batch_size=64):
        self.config = {
            'supernet': supernet,
            'metrics': metrics,
            'population': population,
            'num_evals': num_evals,
            'results_csv_path': results_csv_path,
            'dataset_path': dataset_path,
            'batch_size': batch_size,
        }

class NASConfig:
    def __init__(self, approach=None, search_space=None, search_algorithm=None,
                 metrics=[], higher_is_better=[], max_trials=3, seed=42, dynas=None):
        self._approach = approach
        self._search = DotDict({
            'search_space': search_space,
            'search_algorithm': search_algorithm,
            'metrics': metrics,
            'higher_is_better': higher_is_better,
            'max_trials': max_trials,
            'seed': seed
        })
        self.dynas = None
        if approach == 'dynas' and dynas:
            self.dynas = dynas.config

    @property
    def approach(self):
        return self._approach

    @approach.setter
    def approach(self, approach):
        self._approach = approach

    @property
    def search(self):
        return self._search

    @search.setter
    def search(self, search):
        self._search = search


quantization = QuantizationConfig()
benchmark = BenchmarkConfig()
pruning = WeightPruningConfig()
distillation = DistillationConfig(teacher_model=None)
nas = NASConfig()
onnxruntime_config = ONNX()
tensorflow_config = TensorFlow()
keras_config = Keras()
pytorch_config = PyTorch()
mxnet_config = MXNet()


class Config:
    def __init__(self,
                 quantization=quantization,
                 benchmark=benchmark,
                 options=options,
                 pruning=pruning,
                 distillation=distillation,
                 nas=nas,
                 onnxruntime=onnxruntime_config,
                 tensorflow=tensorflow_config,
                 pytorch=pytorch_config,
                 mxnet=mxnet_config,
                 keras=keras_config):
        self._quantization = quantization
        self._benchmark = benchmark
        self._options = options
        self._onnxruntime = onnxruntime
        self._pruning = pruning
        self._distillation = distillation
        self._nas = nas
        self._tensorflow = tensorflow
        self._pytorch = pytorch
        self._mxnet = mxnet
        self._keras = keras

    @property
    def distillation(self):
        return self._distillation

    @property
    def nas(self):
        return self._nas

    @property
    def tensorflow(self):
        return self._tensorflow

    @property
    def keras(self):
        return self._keras

    @property
    def pytorch(self):
        return self._pytorch

    @property
    def mxnet(self):
        return self._mxnet

    @property
    def pruning(self):
        return self._pruning

    @property
    def quantization(self):
        return self._quantization

    @property
    def benchmark(self):
        return self._benchmark

    @property
    def options(self):
        return self._options

    @property
    def onnxruntime(self):
        return self._onnxruntime

config = Config()
