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
"""Configs for Neural Compressor."""
import datetime
import logging
from schema import Schema, And, Optional
from .conf.dotdict import DotDict
from .conf.config import Pruner

logger = logging.getLogger("neural_compressor")
default_workspace = './nc_workspace/{}/'.format(
    datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

QUANTMAPPING = {
    "auto": "post_training_auto_quant",
    "dynamic": "post_training_dynamic_quant",
    "static": "post_training_static_quant",
    "qat": "quant_aware_training",
}


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
            lambda s: all(i in ['int8', 'uint8', 'fp32', 'bf16'] for i in s)),
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
            lambda s: all(i in ['int8', 'uint8', 'fp32', 'bf16', 'None'] for i in s)),
        Optional('algorithm'): And(
            list,
            lambda s: all(i in ['minmax', 'kl', 'placeholder'] for i in s))}})


def check_value(name, src, supported_type, supported_value=[]):
    """Check if the given object is the given supported type and in the given supported value."""
    if isinstance(src, list) and any([not isinstance(i, supported_type) for i in src]):
        assert False, ("Type of {} items should be {} but not {}".format(
            name, str(supported_type), [type(i) for i in src]))
    elif not isinstance(src, list) and not isinstance(src, supported_type):
        assert False, ("Type of {} should be {} but not {}".format(
            name, str(supported_type), type(src)))

    if len(supported_value) > 0:
        if isinstance(src, str) and src not in supported_value:
            assert False, ("{} is not in supported {}: {}. Skip setting it.".format(
                src, name, str(supported_value)))
        elif isinstance(src, list) and all([isinstance(i, str) for i in src]) and \
            any([i not in supported_value for i in src]):
            assert False, ("{} is not in supported {}: {}. Skip setting it.".format(
                src, name, str(supported_value)))

    return True


class Options:
    """Option Class for configs."""
    def __init__(self, random_seed=1978, workspace=default_workspace,
                 resume_from=None, tensorboard=False):
        """Init an Option object."""
        self.random_seed = random_seed
        self.workspace = workspace
        self.resume_from = resume_from
        self.tensorboard = tensorboard

    @property
    def random_seed(self):
        """Get random seed."""
        return self._random_seed

    @random_seed.setter
    def random_seed(self, random_seed):
        """Set random seed."""
        if check_value('random_seed', random_seed, int):
            self._random_seed = random_seed

    @property
    def workspace(self):
        """Get workspace."""
        return self._workspace

    @workspace.setter
    def workspace(self, workspace):
        """Set workspace."""
        if check_value('workspace', workspace, str):
            self._workspace = workspace

    @property
    def resume_from(self):
        """Get resume_from."""
        return self._resume_from

    @resume_from.setter
    def resume_from(self, resume_from):
        """Set resume_from."""
        if resume_from is None or check_value('resume_from', resume_from, str):
            self._resume_from = resume_from

    @property
    def tensorboard(self):
        """Get tensorboard."""
        return self._tensorboard

    @tensorboard.setter
    def tensorboard(self, tensorboard):
        """Set tensorboard."""
        if check_value('tensorboard', tensorboard, bool):
            self._tensorboard = tensorboard


options = Options()


class BenchmarkConfig:
    """Config Class for Benchmark."""
    def __init__(self,
                 inputs=[],
                 outputs=[],
                 backend='default',
                 warmup=5,
                 iteration=-1,
                 cores_per_instance=None,
                 num_of_instance=None,
                 inter_num_of_threads=None,
                 intra_num_of_threads=None):
        """Init a BenchmarkConfig object."""
        self.inputs = inputs
        self.outputs = outputs
        self.backend = backend
        self.warmup = warmup
        self.iteration = iteration
        self.cores_per_instance = cores_per_instance
        self.num_of_instance = num_of_instance
        self.inter_num_of_threads = inter_num_of_threads
        self.intra_num_of_threads = intra_num_of_threads

    @property
    def backend(self):
        """Get backend."""
        return self._backend

    @backend.setter
    def backend(self, backend):
        """Set backend."""
        if check_value('backend', backend, str, [
                'default', 'itex', 'ipex', 'onnxrt_trt_ep', 'onnxrt_cuda_ep']):
            self._backend = backend

    @property
    def outputs(self):
        """Get outputs."""
        return self._outputs

    @outputs.setter
    def outputs(self, outputs):
        """Set outputs."""
        if check_value('outputs', outputs, str):
            self._outputs = outputs

    @property
    def inputs(self):
        """Get inputs."""
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        """Set inputs."""
        if check_value('inputs', inputs, str):
            self._inputs = inputs

    @property
    def warmup(self):
        """Get warmup."""
        return self._warmup

    @warmup.setter
    def warmup(self, warmup):
        """Set warmup."""
        if check_value('warmup', warmup, int):
            self._warmup = warmup

    @property
    def iteration(self):
        """Get iteration."""
        return self._iteration

    @iteration.setter
    def iteration(self, iteration):
        """Set iteration."""
        if check_value('iteration', iteration, int):
            self._iteration = iteration

    @property
    def cores_per_instance(self):
        """Get cores_per_instance."""
        return self._cores_per_instance

    @cores_per_instance.setter
    def cores_per_instance(self, cores_per_instance):
        """Set cores_per_instance."""
        if cores_per_instance is None or check_value('cores_per_instance', cores_per_instance,
                                                     int):
            self._cores_per_instance = cores_per_instance

    @property
    def num_of_instance(self):
        """Get num_of_instance."""
        return self._num_of_instance

    @num_of_instance.setter
    def num_of_instance(self, num_of_instance):
        """Set num_of_instance."""
        if num_of_instance is None or check_value('num_of_instance', num_of_instance, int):
            self._num_of_instance = num_of_instance

    @property
    def inter_num_of_threads(self):
        """Get inter_num_of_threads."""
        return self._inter_num_of_threads

    @inter_num_of_threads.setter
    def inter_num_of_threads(self, inter_num_of_threads):
        """Set inter_num_of_threads."""
        if inter_num_of_threads is None or check_value('inter_num_of_threads',
                                                       inter_num_of_threads, int):
            self._inter_num_of_threads = inter_num_of_threads

    @property
    def intra_num_of_threads(self):
        """Get intra_num_of_threads."""
        return self._intra_num_of_threads

    @intra_num_of_threads.setter
    def intra_num_of_threads(self, intra_num_of_threads):
        """Get intra_num_of_threads."""
        if intra_num_of_threads is None or check_value('intra_num_of_threads',
                                                       intra_num_of_threads, int):
            self._intra_num_of_threads = intra_num_of_threads


class AccuracyCriterion:
    """Class of Accuracy Criterion."""
    def __init__(self, higher_is_better=True, criterion='relative', tolerable_loss=0.01):
        """Init an AccuracyCriterion object."""
        self.higher_is_better = higher_is_better
        self.criterion = criterion
        self.tolerable_loss = tolerable_loss

    @property
    def higher_is_better(self):
        """Get higher_is_better."""
        return self._higher_is_better

    @higher_is_better.setter
    def higher_is_better(self, higher_is_better):
        """Set higher_is_better."""
        if check_value('higher_is_better', higher_is_better, bool):
            self._higher_is_better = higher_is_better

    @property
    def relative(self):
        """Get tolerable_loss when criterion is relative."""
        if self.criterion != 'relative':
            return None
        return self.tolerable_loss

    @relative.setter
    def relative(self, relative):
        """Set tolerable_loss and criterion to relative."""
        self.criterion = 'relative'
        self.tolerable_loss = relative

    @property
    def absolute(self):
        """Get tolerable_loss when criterion is absolute."""
        if self.criterion != 'absolute':
            return None
        return self.tolerable_loss

    @absolute.setter
    def absolute(self, absolute):
        """Set tolerable_loss and criterion to absolute."""
        self.criterion = 'absolute'
        self.tolerable_loss = absolute

    @property
    def criterion(self):
        """Get criterion."""
        return self._criterion

    @criterion.setter
    def criterion(self, criterion):
        """Set criterion."""
        if check_value('criterion', criterion, str, ['relative', 'absolute']):
            self._criterion = criterion

    @property
    def tolerable_loss(self):
        """Get tolerable_loss."""
        return self._tolerable_loss

    @tolerable_loss.setter
    def tolerable_loss(self, tolerable_loss):
        """Set tolerable_loss."""
        if check_value('tolerable_loss', tolerable_loss, float):
            self._tolerable_loss = tolerable_loss

    def __str__(self):
        """Get criterion."""
        return self.criterion


accuracy_criterion = AccuracyCriterion()


class _BaseQuantizationConfig:
    def __init__(self,
                 inputs=[],
                 outputs=[],
                 backend="default",
                 quant_format="default",
                 device="cpu",
                 calibration_sampling_size=[100],
                 op_type_list=None,
                 op_name_list=None,
                 strategy="basic",
                 strategy_kwargs=None,
                 objective="performance",
                 timeout=0,
                 max_trials=100,
                 performance_only=False,
                 reduce_range=None,
                 excluded_precisions=[],
                 quant_level=1,
                 accuracy_criterion=accuracy_criterion):
        self.inputs = inputs
        self.outputs = outputs
        self.backend = backend
        self.quant_format = quant_format
        self.device = device
        self.op_type_list = op_type_list
        self.op_name_list = op_name_list
        self.strategy = strategy
        self.strategy_kwargs = strategy_kwargs
        self.objective = objective
        self.timeout = timeout
        self.max_trials = max_trials
        self.performance_only = performance_only
        self.reduce_range = reduce_range
        self.excluded_precisions = excluded_precisions
        self.use_bf16 = "bf16" not in self.excluded_precisions
        self.accuracy_criterion = accuracy_criterion
        self.calibration_sampling_size = calibration_sampling_size
        self.quant_level = quant_level

    @property
    def accuracy_criterion(self):
        return self._accuracy_criterion

    @accuracy_criterion.setter
    def accuracy_criterion(self, accuracy_criterion):
        if check_value("accuracy_criterion", accuracy_criterion, AccuracyCriterion):
            self._accuracy_criterion = accuracy_criterion

    @property
    def excluded_precisions(self):
        return self._excluded_precisions

    @excluded_precisions.setter
    def excluded_precisions(self, excluded_precisions):
        if check_value("excluded_precisions", excluded_precisions, str, ["bf16"]):
            self._excluded_precisions = excluded_precisions
            self._use_bf16 = "bf16" not in excluded_precisions

    @property
    def quant_level(self):
        return self._quant_level

    @quant_level.setter
    def quant_level(self, quant_level):
        self._quant_level = quant_level

    @property
    def reduce_range(self):
        return self._reduce_range

    @reduce_range.setter
    def reduce_range(self, reduce_range):
        if reduce_range is None or check_value('reduce_range', reduce_range, bool):
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
            ['basic', 'mse', 'bayesian', 'random', 'exhaustive', 'sigopt', 'tpe', 'mse_v2', 'hawq_v2']):
            self._strategy = strategy

    @property
    def strategy_kwargs(self):
        return self._strategy_kwargs

    @strategy_kwargs.setter
    def strategy_kwargs(self, strategy_kwargs):
        self._strategy_kwargs = strategy_kwargs

    @property
    def op_name_list(self):
        return self._op_name_list

    @op_name_list.setter
    def op_name_list(self, op_name_list):
        if op_name_list is None:
            self._op_name_list = op_name_list
        elif isinstance(op_name_list, dict):
            for k, v in op_name_list.items():
                ops_schema.validate(v)
            self._op_name_list = op_name_list
        else:
            assert False, ("Type of op_name_list should be dict but not {}, ".format(
                type(op_name_list)))

    @property
    def op_type_list(self):
        return self._op_type_list

    @op_type_list.setter
    def op_type_list(self, op_type_list):
        if op_type_list is None:
            self._op_type_list = op_type_list
        elif isinstance(op_type_list, dict):
            for k, v in op_type_list.items():
                ops_schema.validate(v)
            self._op_type_list = op_type_list
        else:
            assert False, ("Type of op_type_list should be dict but not {}".format(
                type(op_type_list)))

    @property
    def calibration_sampling_size(self):
        return self._calibration_sampling_size

    @calibration_sampling_size.setter
    def calibration_sampling_size(self, sampling_size):
        if check_value('calibration_sampling_size', sampling_size, int):
            if isinstance(sampling_size, int):
                sampling_size =[sampling_size]
            self._calibration_sampling_size = sampling_size

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        if check_value('device', device, str, ['cpu', 'gpu']):
            self._device = device

    @property
    def quant_format(self):
        return self._quant_format

    @quant_format.setter
    def quant_format(self, quant_format):
        if check_value('quant_format', quant_format, str,
            ['default', 'QDQ', 'QOperator']):
            self._quant_format = quant_format

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, backend):
        if check_value('backend', backend, str, [
                'default', 'itex', 'ipex', 'onnxrt_trt_ep', 'onnxrt_cuda_ep']):
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


class TuningCriterion:
    """Class for Tuning Criterion."""
    def __init__(self, strategy="basic", strategy_kwargs=None, timeout=0, max_trials=100, objective="performance"):
        """Init a TuningCriterion object."""
        self.strategy = strategy
        self.timeout = timeout
        self.max_trials = max_trials
        self.objective = objective
        self.strategy_kwargs = strategy_kwargs

    @property
    def max_trials(self):
        """Get max_trials."""
        return self._max_trials

    @max_trials.setter
    def max_trials(self, max_trials):
        """Set max_trials."""
        if check_value('max_trials', max_trials, int):
            self._max_trials = max_trials

    @property
    def timeout(self):
        """Get timeout."""
        return self._timeout

    @timeout.setter
    def timeout(self, timeout):
        """Set timeout."""
        if check_value('timeout', timeout, int):
            self._timeout = timeout

    @property
    def objective(self):
        """Get objective."""
        return self._objective

    @objective.setter
    def objective(self, objective):
        """Set objective."""
        if check_value('objective', objective, str,
            ['performance', 'accuracy', 'modelsize', 'footprint']):
            self._objective = objective

    @property
    def strategy(self):
        """Get strategy."""
        return self._strategy

    @strategy.setter
    def strategy(self, strategy):
        """Set strategy."""
        if check_value('strategy', strategy, str,
            ['basic', 'mse', 'bayesian', 'random', 'exhaustive', 'sigopt', 'tpe', 'mse_v2', 'hawq_v2']):
            self._strategy = strategy

    @property
    def strategy_kwargs(self):
        """Get strategy_kwargs."""
        return self._strategy_kwargs

    @strategy_kwargs.setter
    def strategy_kwargs(self, strategy_kwargs):
        """Set strategy_kwargs."""
        self._strategy_kwargs = strategy_kwargs

tuning_criterion = TuningCriterion()


class PostTrainingQuantConfig(_BaseQuantizationConfig):
    """Config Class for Post Training Quantization."""
    def __init__(self,
                 device="cpu",
                 backend="default",
                 quant_format="default",
                 inputs=[],
                 outputs=[],
                 approach="static",
                 calibration_sampling_size=[100],
                 op_type_list=None,
                 op_name_list=None,
                 reduce_range=None,
                 excluded_precisions=[],
                 quant_level=1,
                 tuning_criterion=tuning_criterion,
                 accuracy_criterion=accuracy_criterion,
    ):
        """Init a PostTrainingQuantConfig object."""
        self.tuning_criterion = tuning_criterion
        super().__init__(inputs=inputs,
                         outputs=outputs,
                         device=device,
                         backend=backend,
                         quant_format=quant_format,
                         calibration_sampling_size=calibration_sampling_size,
                         op_type_list=op_type_list,
                         op_name_list=op_name_list,
                         strategy=tuning_criterion.strategy,
                         strategy_kwargs=tuning_criterion.strategy_kwargs,
                         objective=tuning_criterion.objective,
                         timeout=tuning_criterion.timeout,
                         max_trials=tuning_criterion.max_trials,
                         reduce_range=reduce_range,
                         excluded_precisions=excluded_precisions,
                         quant_level=quant_level,
                         accuracy_criterion=accuracy_criterion)
        self.approach = approach

    @property
    def approach(self):
        """Get approach."""
        return self._approach

    @approach.setter
    def approach(self, approach):
        """Set approach."""
        if check_value("approach", approach, str, ["static", "dynamic", "auto"]):
            self._approach = QUANTMAPPING[approach]

    @property
    def tuning_criterion(self):
        """Get tuning_criterion."""
        return self._tuning_criterion

    @tuning_criterion.setter
    def tuning_criterion(self, tuning_criterion):
        """Set tuning_criterion."""
        if check_value("tuning_criterion", tuning_criterion, TuningCriterion):
            self._tuning_criterion = tuning_criterion


class QuantizationAwareTrainingConfig(_BaseQuantizationConfig):
    """Config Class for Quantization Aware Training."""
    def __init__(self,
                 device="cpu",
                 backend="default",
                 inputs=[],
                 outputs=[],
                 op_type_list=None,
                 op_name_list=None,
                 reduce_range=None,
                 excluded_precisions=[],
                 quant_level=1):
        """Init a QuantizationAwareTrainingConfig object."""
        super().__init__(inputs=inputs,
                         outputs=outputs,
                         device=device,
                         backend=backend,
                         op_type_list=op_type_list,
                         op_name_list=op_name_list,
                         reduce_range=reduce_range,
                         excluded_precisions=excluded_precisions,
                         quant_level=quant_level)
        self._approach = 'quant_aware_training'

    @property
    def approach(self):
        """Get approach."""
        return self._approach


pruners = [Pruner()]


class WeightPruningConfig:
    """Similiar to torch optimizer's interface."""
    def __init__(self, pruning_configs=[{}],  ##empty dict will use global values
                 target_sparsity=0.9, pruning_type="snip_momentum", pattern="4x1", op_names=[],
                 excluded_op_names=[],
                 start_step=0, end_step=0, pruning_scope="global", pruning_frequency=1,
                 min_sparsity_ratio_per_op=0.0, max_sparsity_ratio_per_op=0.98,
                 sparsity_decay_type="exp", pruning_op_types=['Conv', 'Linear'],
                 **kwargs):
        """Init a WeightPruningConfig object."""
        self.pruning_configs = pruning_configs
        self._weight_compression = DotDict({
            'target_sparsity': target_sparsity,
            'pruning_type': pruning_type,
            'pattern': pattern,
            'op_names': op_names,
            'excluded_op_names': excluded_op_names,  ##global only
            'start_step': start_step,
            'end_step': end_step,
            'pruning_scope': pruning_scope,
            'pruning_frequency': pruning_frequency,
            'min_sparsity_ratio_per_op': min_sparsity_ratio_per_op,
            'max_sparsity_ratio_per_op': max_sparsity_ratio_per_op,
            'sparsity_decay_type': sparsity_decay_type,
            'pruning_op_types': pruning_op_types,
        })
        self._weight_compression.update(kwargs)

    @property
    def weight_compression(self):
        """Get weight_compression."""
        return self._weight_compression

    @weight_compression.setter
    def weight_compression(self, weight_compression):
        """Set weight_compression."""
        self._weight_compression = weight_compression


class KnowledgeDistillationLossConfig:
    """Config Class for Knowledge Distillation Loss."""
    def __init__(self, temperature=1.0, loss_types=['CE', 'CE'], loss_weights=[0.5, 0.5]):
        """Init a KnowledgeDistillationLossConfig object."""
        self.config = DotDict({
            'KnowledgeDistillationLoss': {
                'temperature': temperature,
                'loss_types': loss_types,
                'loss_weights': loss_weights
            }
        })


class IntermediateLayersKnowledgeDistillationLossConfig:
    """Config Class for Intermediate Layers Knowledge Distillation Loss."""
    def __init__(self, layer_mappings=[], loss_types=[], loss_weights=[], add_origin_loss=False):
        """Init an IntermediateLayersKnowledgeDistillationLossConfig object."""
        self.config = DotDict({
            'IntermediateLayersKnowledgeDistillationLoss': {
                'layer_mappings': layer_mappings,
                'loss_types': loss_types,
                'loss_weights': loss_weights,
                'add_origin_loss': add_origin_loss
            }
        })


class SelfKnowledgeDistillationLossConfig:
    """Config Class for Self Knowledge Distillation Loss."""
    def __init__(self,
                 layer_mappings=[],
                 temperature=1.0,
                 loss_types=[],
                 loss_weights=[],
                 add_origin_loss=False):
        """Init a SelfKnowledgeDistillationLossConfig object."""
        self.config = DotDict({
            'SelfKnowledgeDistillationLoss': {
                'layer_mappings': layer_mappings,
                'temperature': temperature,
                'loss_types': loss_types,
                'loss_weights': loss_weights,
                'add_origin_loss': add_origin_loss,
            }
        })


criterion = KnowledgeDistillationLossConfig()


class DistillationConfig:
    """Config of distillation.
    
    Args:
        teacher_model (Callable): Teacher model for distillation. Defaults to None.
        features (optional): Teacher features for distillation, features and teacher_model are alternative.
                             Defaults to None.
        criterion (Callable, optional): Distillation loss configure.
        optimizer (dictionary, optional): Optimizer configure.
    """
    def __init__(self,
                 teacher_model=None,
                 criterion=criterion,
                 optimizer={'SGD': {
                     'learning_rate': 0.0001
                 }}):
        """Init a DistillationConfig object."""
        self.criterion = criterion.config
        self.optimizer = optimizer
        self.teacher_model = teacher_model

    @property
    def criterion(self):
        """Get criterion."""
        return self._criterion

    @criterion.setter
    def criterion(self, criterion):
        """Set criterion."""
        self._criterion = criterion

    @property
    def optimizer(self):
        """Get optimizer."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """Set optimizer."""
        self._optimizer = optimizer

    @property
    def teacher_model(self):
        """Get teacher_model."""
        return self._teacher_model

    @teacher_model.setter
    def teacher_model(self, teacher_model):
        """Set teacher_model."""
        self._teacher_model = teacher_model


class MixedPrecisionConfig(PostTrainingQuantConfig):
    """Config Class for MixedPrecision."""
    def __init__(self,
                 device="cpu",
                 backend="default",
                 inputs=[],
                 outputs=[],
                 tuning_criterion=tuning_criterion,
                 accuracy_criterion=accuracy_criterion,
                 excluded_precisions=[]):
        """Init a MixedPrecisionConfig object."""
        super().__init__(inputs=inputs,
                         outputs=outputs,
                         device=device,
                         backend=backend,
                         tuning_criterion=tuning_criterion,
                         accuracy_criterion=accuracy_criterion,
                         excluded_precisions=excluded_precisions,
        )


class ExportConfig:
    """Config Class for Export."""
    def __init__(
        self,
        dtype="int8",
        opset_version=14,
        quant_format="QDQ",
        example_inputs=None,
        input_names=None,
        output_names=None,
        dynamic_axes=None,
    ):
        """Init an ExportConfig object."""
        self.dtype = dtype
        self.opset_version = opset_version
        self.quant_format = quant_format
        self.example_inputs = example_inputs
        self.input_names = input_names
        self.output_names = output_names
        self.dynamic_axes = dynamic_axes

    @property
    def dtype(self):
        """Get dtype."""
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        """Set dtype."""
        self._dtype = dtype

    @property
    def opset_version(self):
        """Get opset_version."""
        return self._opset_version

    @opset_version.setter
    def opset_version(self, opset_version):
        """Set opset_version."""
        self._opset_version = opset_version

    @property
    def quant_format(self):
        """Get quant_format."""
        return self._quant_format

    @quant_format.setter
    def quant_format(self, quant_format):
        """Set quant_format."""
        self._quant_format = quant_format

    @property
    def example_inputs(self):
        """Get example_inputs."""
        return self._example_inputs

    @example_inputs.setter
    def example_inputs(self, example_inputs):
        """Set example_inputs."""
        self._example_inputs = example_inputs

    @property
    def input_names(self):
        """Get input_names."""
        return self._input_names

    @input_names.setter
    def input_names(self, input_names):
        """Set input_names."""
        self._input_names = input_names

    @property
    def output_names(self):
        """Get output_names."""
        return self._output_names

    @output_names.setter
    def output_names(self, output_names):
        """Set output_names."""
        self._output_names = output_names

    @property
    def dynamic_axes(self):
        """Get dynamic_axes."""
        return self._dynamic_axes

    @dynamic_axes.setter
    def dynamic_axes(self, dynamic_axes):
        """Set dynamic_axes."""
        self._dynamic_axes = dynamic_axes

class ONNXQlinear2QDQConfig:
    """Config Class for ONNXQlinear2QDQ."""
    def __init__(self):
        """Init an ONNXQlinear2QDQConfig object."""
        pass

class Torch2ONNXConfig(ExportConfig):
    """Config Class for Torch2ONNX."""
    def __init__(
       self,
       dtype="int8",
       opset_version=14,
       quant_format="QDQ",
       example_inputs=None,
       input_names=None,
       output_names=None,
       dynamic_axes=None,
       recipe='QDQ_OP_FP32_BIAS',
       **kwargs,
    ):
        """Init a Torch2ONNXConfig object."""
        super().__init__(
            dtype=dtype,
            opset_version=opset_version,
            quant_format=quant_format,
            example_inputs=example_inputs,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
        self.recipe = recipe
        self.kwargs = kwargs


class TF2ONNXConfig(ExportConfig):
    """Config Class for TF2ONNX."""
    def __init__(
       self,
       dtype="int8",
       opset_version=14,
       quant_format="QDQ",
       example_inputs=None,
       input_names=None,
       output_names=None,
       dynamic_axes=None,
       **kwargs,
    ):
        """Init a TF2ONNXConfig object."""
        super().__init__(
            dtype=dtype,
            opset_version=opset_version,
            quant_format=quant_format,
            example_inputs=example_inputs,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
        self.kwargs = kwargs
