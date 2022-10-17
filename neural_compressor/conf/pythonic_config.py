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
import datetime
from schema import Schema, And, Use, Optional, Or
from .dotdict import DotDict
from .config import Pruner

logger = logging.getLogger()

default_workspace = './nc_workspace/{}/'.format(
    datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

ops_schema = Schema({
    Optional('weight', default=None): {
        Optional('granularity'): And(
            list,
            lambda s: all(i in ['per_channel', 'per_tensor'] for i in s)),
        Optional('scheme'): And(
            list,
            lambda s: all(i in ['asym', 'sym', 'asym_float'] for i in s)),
        Optional('dtype'): And(
            list,
            lambda s: all(i in ['int8', 'uint8', 'fp32', 'bf16', 'fp16'] for i in s)),
        Optional('algorithm'): And(
            list,
            lambda s: all(i in ['minmax'] for i in s))},
    Optional('activation', default=None): {
        Optional('granularity'): And(
            list,
            lambda s: all(i in ['per_channel', 'per_tensor'] for i in s)),
        Optional('scheme'): And(
            list,
            lambda s: all(i in ['asym', 'sym'] for i in s)),
        Optional('dtype'): And(
            list,
            lambda s: all(i in ['int8', 'uint8', 'fp32', 'bf16', 'fp16', 'None'] for i in s)),
        Optional('algorithm'): And(
            list,
            lambda s: all(i in ['minmax', 'kl', 'placeholder'] for i in s))}})


def check_value(name, src, supported_type, supported_value=[]):
    if isinstance(src, list) and any([not isinstance(i, supported_type) for i in src]):
        logger.warning("Type of {} items should be {} but not {}, " \
            "use its default value.".format(name, str(supported_type), [type(i) for i in src]))
        return False
    elif not isinstance(src, list) and not isinstance(src, supported_type):
        logger.warning("Type of {} should be {} but not {}, " \
            "use its default value.".format(name, str(supported_type), type(src)))
        return False

    if len(supported_value) > 0:
        if isinstance(src, str) and src not in supported_value:
            logger.warning("{} is not in supported {}: {}. Skip setting it and" \
                " use default value.".format(src, name, str(supported_value)))
            return False
        elif isinstance(src, list) and all([isinstance(i, str) for i in src]) and \
            any([i not in supported_value for i in src]):
            logger.warning("{} is not in supported {}: {}. Skip setting it and" \
                " use default value.".format(src, name, str(supported_value)))
            return False
 
    return True

class BenchmarkConfig:
    def __init__(self, warmup=5, iteration=-1, cores_per_instance=None, num_of_instance=None,
        inter_num_of_threads=None, intra_num_of_threads=None):
        self._warmup = warmup
        self._iteration = iteration
        self._cores_per_instance = cores_per_instance
        self._num_of_instance = num_of_instance
        self._inter_num_of_threads = inter_num_of_threads
        self._intra_num_of_threads = intra_num_of_threads

    @property
    def warmup(self):
        return self._warmup

    @warmup.setter
    def warmup(self, warmup):
        if check_value('warmup', warmup, int):
            self._warmup = warmup

    @property
    def iteration(self):
        return self._iteration

    @iteration.setter
    def iteration(self, iteration):
        if check_value('iteration', iteration, int):
            self._iteration = iteration

    @property
    def cores_per_instance(self):
        return self._cores_per_instance

    @cores_per_instance.setter
    def cores_per_instance(self, cores_per_instance):
        if check_value('cores_per_instance', cores_per_instance, int):
            self._cores_per_instance = cores_per_instance

    @property
    def num_of_instance(self):
        return self._num_of_instance

    @num_of_instance.setter
    def num_of_instance(self, num_of_instance):
        if check_value('num_of_instance', num_of_instance, int):
            self._num_of_instance = num_of_instance

    @property
    def inter_num_of_threads(self):
        return self._inter_num_of_threads

    @inter_num_of_threads.setter
    def inter_num_of_threads(self, inter_num_of_threads):
        if check_value('inter_num_of_threads', inter_num_of_threads, int):
            self._inter_num_of_threads = inter_num_of_threads

    @property
    def intra_num_of_threads(self):
        return self._intra_num_of_threads

    @intra_num_of_threads.setter
    def intra_num_of_threads(self, intra_num_of_threads):
        if check_value('intra_num_of_threads', intra_num_of_threads, int):
            self._intra_num_of_threads = intra_num_of_threads

class AccuracyLoss:
    def __init__(self, loss=0.01):
        self._loss = loss

    @property
    def relative(self):
        return self._loss

    @relative.setter
    def relative(self, relative):
        if check_value('relative tolerable loss', relative, float):
            self._loss = relative

    @property
    def absolute(self):
        return self._loss

    @absolute.setter
    def absolute(self, absolute):
        if check_value('absolute tolerable loss', absolute, float):
            self._loss = absolute

tolerable_loss = AccuracyLoss()

class AccuracyCriterion:
    def __init__(self, higher_is_better=True, criterion='relative', tolerable_loss=tolerable_loss):
        self._higher_is_better = higher_is_better
        self._criterion = criterion
        self._tolerable_loss = tolerable_loss

    @property
    def higher_is_better(self):
        return self._higher_is_better

    @higher_is_better.setter
    def higher_is_better(self, higher_is_better):
        if check_value('higher_is_better', higher_is_better, bool):
            self._higher_is_better = higher_is_better
        
    @property
    def relative(self):
        if self._criterion != 'relative':
            return None
        return self._tolerable_loss.relative

    @relative.setter
    def relative(self, relative):
        self._criterion = 'relative'
        self._tolerable_loss.relative = relative

    @property
    def absolute(self):
        if self._criterion != 'absolute':
            return None
        return self._tolerable_loss.absolute

    @absolute.setter
    def absolute(self, absolute):
        self._criterion = 'absolute'
        self._tolerable_loss.absolute = absolute

    def __str__(self):
        return self._criterion

accuracy_criterion = AccuracyCriterion()

class QuantizationConfig:
    def __init__(self, inputs=[], outputs=[], backend='NA', device='cpu', 
        approach='post_training_static_quant', calibration_sampling_size=[100],
        op_type_list=None, op_name_list=None, strategy='basic', objective='performance',
        timeout=0, max_trials=100, performance_only=False, reduce_range=None,
        use_bf16=True, accuracy_criterion=accuracy_criterion):
        self._inputs = inputs
        self._outputs = outputs
        self._backend = backend
        self._device = device
        self._op_type_list = op_type_list
        self._op_name_list = op_name_list
        self._strategy = strategy
        self._objective = objective
        self._timeout = timeout
        self._max_trials = max_trials
        self._performance_only = performance_only
        self._reduce_range = reduce_range
        self._use_bf16 = use_bf16
        self._accuracy_criterion = accuracy_criterion
        self._approach = approach
        self._calibration_sampling_size = calibration_sampling_size
        
    @property
    def accuracy_criterion(self):
        return self._accuracy_criterion

    @property
    def use_bf16(self):
        return self._use_bf16

    @use_bf16.setter
    def use_bf16(self, use_bf16):
        if check_value('use_bf16', use_bf16, bool):
            self._use_bf16 = use_bf16

    @property
    def reduce_range(self):
        return self._reduce_range

    @reduce_range.setter
    def reduce_range(self, reduce_range):
        if check_value('reduce_range', reduce_range, bool):
            self._reduce_range = reduce_range

    @property
    def performance_only(self):
        return self._performance_only

    @performance_only.setter
    def performance_only(self, performance_only):
        if check_value('performance_only', performance_only, bool):
            self._performance_only = performance_only

    @property
    def max_trials(self):
        return self._max_trials

    @max_trials.setter
    def max_trials(self, max_trials):
        if check_value('max_trials', max_trials, int):
            self._max_trials = max_trials

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, timeout):
        if check_value('timeout', timeout, int):
            self._timeout = timeout

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, objective):
        if check_value('objective', objective, str,
            ['performance', 'accuracy', 'modelsize', 'footprint']):
            self._objective = objective

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy):
        if check_value('strategy', strategy, str,
            ['basic', 'mse', 'bayesian', 'random', 'exhaustive']):
            self._strategy = strategy

    @property
    def op_name_list(self):
        return self._op_name_list

    @op_name_list.setter
    def op_name_list(self, op_name_list):
        if not isinstance(op_name_list, dict):
            logger.warning("Type of op_name_list should be dict but not {}, " \
                "use its default value.".format(type(op_name_list)))
        else:
            for k, v in op_name_list.items():
                ops_schema.validate(v)
            self._op_name_list = op_name_list

    @property
    def op_type_list(self):
        return self._op_type_list

    @op_type_list.setter
    def op_type_list(self, op_type_list):
        if not isinstance(op_type_list, dict):
            logger.warning("Type of op_type_list should be dict but not {}, " \
                "use its default value.".format(type(op_type_list)))
        else:
            for k, v in op_type_list.items():
                ops_schema.validate(v)
            self._op_type_list = op_type_list

    @property
    def calibration_sampling_size(self):
        return self._calibration_sampling_size

    @calibration_sampling_size.setter
    def calibration_sampling_size(self, sampling_size):
        if check_value('calibration_sampling_size', sampling_size, int):
            self._calibration_sampling_size = sampling_size

    @property
    def approach(self):
        return self._approach

    @approach.setter
    def approach(self, approach):
        if check_value('approach', approach, str, ['post_training_static_quant',
            'post_training_dynamic_quant', 'quant_aware_training']):
            self._approach = approach

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        if check_value('device', device, str, ['cpu', 'gpu']):
            self._device = device

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, backend):
        if check_value('backend', backend, str, 
            ['tensorflow', 'tensorflow_itex', 'pytorch',
            'pytorch_ipex', 'pytorch_fx', 'onnxrt_qlinearops', 'onnxrt_integerops',
            'onnxrt_qdq', 'onnxrt_qoperator', 'mxnet']):
            self._backend = backend

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, outputs):
        if check_value('outputs', outputs, str):
            self._outputs = outputs

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        if check_value('inputs', inputs, str):
            self._inputs = inputs

class Options:
    def __init__(self, random_seed=1978, workspace=default_workspace, 
        resume_from=None, tensorboard=False):
        self._random_seed = random_seed
        self._workspace = workspace
        self._resume_from = resume_from
        self._tensorboard = tensorboard

    @property
    def random_seed(self):
        return self._random_seed

    @random_seed.setter
    def random_seed(self, random_seed):
        if check_value('random_seed', random_seed, int):
            self._random_seed = random_seed

    @property
    def workspace(self):
        return self._workspace

    @workspace.setter
    def workspace(self, workspace):
        if check_value('workspace', workspace, str):
            self._workspace = workspace

    @property
    def resume_from(self):
        return self._resume_from

    @resume_from.setter
    def resume_from(self, resume_from):
        if check_value('resume_from', resume_from, str):
            self._resume_from = resume_from

    @property
    def tensorboard(self):
        return self._tensorboard

    @tensorboard.setter
    def tensorboard(self, tensorboard):
        if check_value('tensorboard', tensorboard, bool):
            self._tensorboard = tensorboard

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
        if check_value('precisions', precisions, str, ['int8', 'uint8', 'fp32', 'bf16', 'fp16']):
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

class PyTorch(MXNet):
    def __init__(self, precisions=None):
        super().__init__(precisions)

pruners = [Pruner()]

class PruningConfig:
    def __init__(self, pruners=pruners, initial_sparsity=0.0, target_sparsity=0.97,
                 max_sparsity_ratio_per_layer=0.98, prune_type="basic_magnitude",
                 start_epoch=0, end_epoch=4, start_step=0, end_step=0, update_frequency=1.0,
                 update_frequency_on_step=1, not_to_prune_names=[], prune_domain="global",
                 names=[], exclude_names=[], prune_layer_type=[], sparsity_decay_type="exp",
                 pattern="tile_pattern_1x1"):
        self._weight_compression = DotDict({
            'initial_sparsity': initial_sparsity,
            'target_sparsity': target_sparsity,
            'max_sparsity_ratio_per_layer': max_sparsity_ratio_per_layer,
            'prune_type': prune_type,
            'start_epoch': start_epoch,
            'end_epoch': end_epoch,
            'start_step': start_step,
            'end_step': end_step,
            'update_frequency': update_frequency,
            'update_frequency_on_step': update_frequency_on_step,
            'not_to_prune_names': not_to_prune_names,
            'prune_domain': prune_domain,
            'names': names,
            'exclude_names': exclude_names,
            'prune_layer_type': prune_layer_type,
            'sparsity_decay_type': sparsity_decay_type,
            'pattern': pattern,
            'pruners': pruners
        })

    @property
    def weight_compression(self):
        return self._weight_compression

    @weight_compression.setter
    def weight_compression(self, weight_compression):
        self._weight_compression = weight_compression

class KnowledgeDistillationLossConfig:
    def __init__(self, temperature=1.0, loss_types=['CE', 'CE'], loss_weights=[0.5, 0.5]):
        self.config = DotDict({
            'KnowledgeDistillationLoss': {
                'temperature': temperature, 
                'loss_types': loss_types, 
                'loss_weights': loss_weights
            }
        })

class IntermediateLayersKnowledgeDistillationLossConfig:
    def __init__(self, layer_mappings=[], loss_types=[], loss_weights=[], add_origin_loss=False):
        self.config = DotDict({
            'IntermediateLayersKnowledgeDistillationLoss': {
                'layer_mappings': layer_mappings,
                'loss_types': loss_types,
                'loss_weights': loss_weights,
                'add_origin_loss': add_origin_loss
            }
        })

criterion = KnowledgeDistillationLossConfig()

class DistillationConfig:
    def __init__(self, criterion=criterion, optimizer={'SGD':{'learning_rate':0.0001}}):
        self._criterion = criterion.config
        self._optimizer = optimizer

    @property
    def criterion(self):
        return self._criterion

    @criterion.setter
    def criterion(self, criterion):
        self._criterion = criterion

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

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
options = Options()
pruning = PruningConfig()
distillation = DistillationConfig()
nas = NASConfig()
onnxruntime_config = ONNX()
tensorflow_config = TensorFlow()
pytorch_config = PyTorch()
mxnet_config = MXNet()

class Config:
    def __init__(self, quantization=quantization, benchmark=benchmark, options=options,
        pruning=pruning, distillation=distillation, nas=nas, onnxruntime=onnxruntime_config,
        tensorflow=tensorflow_config, pytorch=pytorch_config, mxnet=mxnet_config):
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
