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
"""Configs for Neural Compressor 1.x."""
import datetime
import logging

from schema import And, Optional, Schema

from .dotdict import DotDict

logger = logging.getLogger("neural_compressor")
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


def _check_value(name, src, supported_type, supported_value=[]):
    """Check if the given object is the given supported type and in the given supported value.

    Example::

        from neural_compressor.config import _check_value

        def datatype(self, datatype):
            if _check_value('datatype', datatype, list, ['fp32', 'bf16', 'uint8', 'int8']):
                self._datatype = datatype
    """
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
    """Option Class for configs.

    This class is used for configuring global variables. The global variable options is created with this class.
    If you want to change global variables, you should use functions from utils.utility.py:
        set_random_seed(seed: int)
        set_workspace(workspace: str)
        set_resume_from(resume_from: str)
        set_tensorboard(tensorboard: bool)

    Args:
        random_seed(int): Random seed used in neural compressor.
                          Default value is 1978.
        workspace(str): The directory where intermediate files and tuning history file are stored.
                        Default value is:
                            './nc_workspace/{}/'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')).
        resume_from(str): The directory you want to resume tuning history file from.
                          The tuning history was automatically saved in the workspace directory
                               during the last tune process.
                          Default value is None.
        tensorboard(bool): This flag indicates whether to save the weights of the model and the inputs of each layer
                               for visual display.
                           Default value is False.

    Example::

        from neural_compressor import set_random_seed, set_workspace, set_resume_from, set_tensorboard
        set_random_seed(2022)
        set_workspace("workspace_path")
        set_resume_from("workspace_path")
        set_tensorboard(True)
    """
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
        if _check_value('random_seed', random_seed, int):
            self._random_seed = random_seed

    @property
    def workspace(self):
        """Get workspace."""
        return self._workspace

    @workspace.setter
    def workspace(self, workspace):
        """Set workspace."""
        if _check_value('workspace', workspace, str):
            self._workspace = workspace

    @property
    def resume_from(self):
        """Get resume_from."""
        return self._resume_from

    @resume_from.setter
    def resume_from(self, resume_from):
        """Set resume_from."""
        if resume_from is None or _check_value('resume_from', resume_from, str):
            self._resume_from = resume_from

    @property
    def tensorboard(self):
        """Get tensorboard."""
        return self._tensorboard

    @tensorboard.setter
    def tensorboard(self, tensorboard):
        """Set tensorboard."""
        if _check_value('tensorboard', tensorboard, bool):
            self._tensorboard = tensorboard


class AccuracyCriterion:
    """Class of Accuracy Criterion.

    Args:
        higher_is_better(bool, optional): This flag indicates whether the metric higher is the better.
                                          Default value is True.
        criterion:(str, optional): This flag indicates whether the metric loss is 'relative' or 'absolute'.
                                   Default value is 'relative'.
        tolerable_loss(float, optional): This float indicates how much metric loss we can accept.
                                         Default value is 0.01.

    Example::

        from neural_compressor.config import AccuracyCriterion

        accuracy_criterion = AccuracyCriterion(
            higher_is_better=True,  # optional.
            criterion='relative',  # optional. Available values are 'relative' and 'absolute'.
            tolerable_loss=0.01,  # optional.
        )
    """
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
        if _check_value('higher_is_better', higher_is_better, bool):
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
        if _check_value('criterion', criterion, str, ['relative', 'absolute']):
            self._criterion = criterion

    @property
    def tolerable_loss(self):
        """Get tolerable_loss."""
        return self._tolerable_loss

    @tolerable_loss.setter
    def tolerable_loss(self, tolerable_loss):
        """Set tolerable_loss."""
        if _check_value('tolerable_loss', tolerable_loss, float):
            self._tolerable_loss = tolerable_loss

    def __str__(self):
        """Get criterion."""
        return self.criterion

    def keys(self):
        """Returns keys of the dict."""
        return ('higher_is_better', 'criterion', 'tolerable_loss')

    def __getitem__(self, item):
        """Get the dict."""
        return getattr(self, item)


accuracy_criterion = AccuracyCriterion()


class _BaseQuantizationConfig:
    """Basic class for quantization config. Inherited by PostTrainingQuantConfig and QuantizationAwareTrainingConfig.

    Args:
        inputs: Inputs of model, only required in tensorflow.
        outputs: Outputs of model, only required in tensorflow.
        backend: Backend for model execution. Support 'default', 'itex', 'ipex', 'onnxrt_trt_ep', 'onnxrt_cuda_ep'
        domain: Model domain. Support 'auto', 'cv', 'object_detection', 'nlp' and 'recommendation_system'.
                Adaptor will use specific quantization settings for different domains automatically, and
                explicitly specified quantization settings will override the automatic setting.
                If users set domain as auto, automatic detection for domain will be executed.
        recipes: Recipes for quantiztaion, support list is as below.
                 'smooth_quant': whether do smooth quant
                 'smooth_quant_args': parameters for smooth_quant
                 'fast_bias_correction': whether do fast bias correction
                 'weight_correction': whether do weight correction
                 'gemm_to_matmul': whether convert gemm to matmul and add, only valid for onnx models
                 'graph_optimization_level': support 'DISABLE_ALL', 'ENABLE_BASIC', 'ENABLE_EXTENDED', 'ENABLE_ALL'
                                           only valid for onnx models
                 'first_conv_or_matmul_quantization': whether quantize the first conv or matmul
                 'last_conv_or_matmul_quantization': whether quantize the last conv or matmul
                 'pre_post_process_quantization': whether quantize the ops in preprocess and postprocess
                 'add_qdq_pair_to_weight': whether add QDQ pair for weights, only valid for onnxrt_trt_ep
                 'optypes_to_exclude_output_quant': don't quantize output of specified optypes
                 'dedicated_qdq_pair': whether dedicate QDQ pair, only valid for onnxrt_trt_ep
        quant_format: Support 'default', 'QDQ' and 'QOperator', only required in ONNXRuntime.
        device: Support 'cpu' and 'gpu'.
        calibration_sampling_size: Number of calibration sample.
        op_type_dict: Tuning constraints on optype-wise  for advance user to reduce tuning space.
                      User can specify the quantization config by op type:
                      example:
                      {
                          'Conv': {
                              'weight': {
                                  'dtype': ['fp32']
                              },
                              'activation': {
                                  'dtype': ['fp32']
                              }
                          }
                      }
        op_name_dict: Tuning constraints on op-wise for advance user to reduce tuning space.
                      User can specify the quantization config by op name:
                      example:
                      {
                          "layer1.0.conv1": {
                              "activation": {
                                  "dtype": ["fp32"]
                              },
                              "weight": {
                                  "dtype": ["fp32"]
                              }
                          },
                      }
        strategy: Strategy name used in tuning, Please refer to docs/source/tuning_strategies.md.
        strategy_kwargs: Parameters for strategy, Please refer to docs/source/tuning_strategies.md.
        objective: Objective with accuracy constraint guaranteed, support 'performance', 'modelsize', 'footprint'.
                   Please refer to docs/source/objective.md.
                   Default value is 'performance'.
        timeout: Tuning timeout (seconds). default value is 0 which means early stop
        max_trials: Max tune times. default value is 100. Combine with timeout field to decide when to exit
        performance_only: Whether do evaluation
        reduce_range: Whether use 7 bit to quantization.
        example_inputs: Used to trace PyTorch model with torch.jit/torch.fx.
        excluded_precisions: Precisions to be excluded, Default value is empty list.
                             Neural compressor enable the mixed precision with fp32 + bf16 + int8 by default.
                             If you want to disable bf16 data type, you can specify excluded_precisions = ['bf16].
        quant_level: Support auto, 0 and 1, 0 is conservative strategy, 1 is basic or user-specified
                     strategy, auto (default) is the combination of 0 and 1.
        accuracy_criterion: Accuracy constraint settings.
        use_distributed_tuning: Whether use distributed tuning or not.
    """
    def __init__(self,
                 inputs=[],
                 outputs=[],
                 backend="default",
                 domain="auto",
                 recipes={},
                 quant_format="default",
                 device="cpu",
                 calibration_sampling_size=[100],
                 op_type_dict=None,
                 op_name_dict=None,
                 strategy="basic",
                 strategy_kwargs=None,
                 objective="performance",
                 timeout=0,
                 max_trials=100,
                 performance_only=False,
                 reduce_range=None,
                 example_inputs=None,
                 excluded_precisions=[],
                 quant_level="auto",
                 accuracy_criterion=accuracy_criterion,
                 use_distributed_tuning=False,
                 diagnosis=False):
        """Initialize _BaseQuantizationConfig class."""
        self.inputs = inputs
        self.outputs = outputs
        self.backend = backend
        self.domain = domain
        self.recipes = recipes
        self.quant_format = quant_format
        self.device = device
        self.op_type_dict = op_type_dict
        self.op_name_dict = op_name_dict
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
        self.use_distributed_tuning = use_distributed_tuning
        self.diagnosis = diagnosis
        self._example_inputs = example_inputs

    @property
    def domain(self):
        """Get domain."""
        return self._domain

    @domain.setter
    def domain(self, domain):
        """Set domain."""
        if _check_value("domain", domain, str,
            ["auto", "cv", "object_detection", "nlp", "recommendation_system"]):
            self._domain = domain

    @property
    def recipes(self):
        """Get recipes."""
        return self._recipes

    @recipes.setter
    def recipes(self, recipes):
        """Set recipes."""
        if recipes is not None and not isinstance(recipes, dict):
            raise ValueError("recipes should be a dict.")

        def smooth_quant(val=None):
            if val is not None:
                return _check_value("smooth_quant", val, bool)
            else:
                return False

        def smooth_quant_args(val=None):
            if val is not None:
                _check_value("smooth_quant_args", val, dict)
                for k, v in val.items():
                    if k == "alpha":
                        if isinstance(v, str):
                            assert v == "auto", "the alpha of sq only supports float and 'auto'"
                        else:
                            _check_value("alpha", v, float)

                return True
            else:
                return {}

        def fast_bias_correction(val=None):
            if val is not None:
                return _check_value("fast_bias_correction", val, bool)
            else:
                return False

        def weight_correction(val=None):
            if val is not None:
                return _check_value("weight_correction", val, bool)
            else:
                return False

        def gemm_to_matmul(val=None):
            if val is not None:
                return _check_value("gemm_to_matmul", val, bool)
            else:
                return True

        def graph_optimization_level(val=None):
            if val is not None:
                return _check_value("graph_optimization_level", val, str,
                    ["DISABLE_ALL", "ENABLE_BASIC", "ENABLE_EXTENDED", "ENABLE_ALL"])
            else:
                return None

        def first_conv_or_matmul_quantization(val=None):
            if val is not None:
                return _check_value("first_conv_or_matmul_quantization", val, bool)
            else:
                return True

        def last_conv_or_matmul_quantization(val=None):
            if val is not None:
                return _check_value("last_conv_or_matmul_quantization", val, bool)
            else:
                return True

        def pre_post_process_quantization(val=None):
            if val is not None:
                return _check_value("pre_post_process_quantization", val, bool)
            else:
                return True

        def add_qdq_pair_to_weight(val=None):
            if val is not None:
                return _check_value("add_qdq_pair_to_weight", val, bool)
            else:
                return False

        def optypes_to_exclude_output_quant(val=None):
            if val is not None:
                return isinstance(val, list)
            else:
                return []

        def dedicated_qdq_pair(val=None):
            if val is not None:
                return _check_value("dedicated_qdq_pair", val, bool)
            else:
                return False

        RECIPES = {"smooth_quant": smooth_quant,
                   "smooth_quant_args": smooth_quant_args,
                   "fast_bias_correction": fast_bias_correction,
                   "weight_correction": weight_correction,
                   "gemm_to_matmul": gemm_to_matmul,
                   "graph_optimization_level": graph_optimization_level,
                   "first_conv_or_matmul_quantization": first_conv_or_matmul_quantization,
                   "last_conv_or_matmul_quantization": last_conv_or_matmul_quantization,
                   "pre_post_process_quantization": pre_post_process_quantization,
                   "add_qdq_pair_to_weight": add_qdq_pair_to_weight,
                   "optypes_to_exclude_output_quant": optypes_to_exclude_output_quant,
                   "dedicated_qdq_pair": dedicated_qdq_pair
                   }
        self._recipes = {}
        for k in RECIPES.keys():
            if k in recipes and RECIPES[k](recipes[k]):
                self._recipes.update({k: recipes[k]})
            else:
                self._recipes.update({k: RECIPES[k]()})

    @property
    def accuracy_criterion(self):
        return self._accuracy_criterion

    @accuracy_criterion.setter
    def accuracy_criterion(self, accuracy_criterion):
        if _check_value("accuracy_criterion", accuracy_criterion, AccuracyCriterion):
            self._accuracy_criterion = accuracy_criterion

    @property
    def excluded_precisions(self):
        return self._excluded_precisions

    @excluded_precisions.setter
    def excluded_precisions(self, excluded_precisions):
        if _check_value("excluded_precisions", excluded_precisions, str, ["bf16", "fp16"]):
            self._excluded_precisions = excluded_precisions
            self._use_bf16 = "bf16" not in excluded_precisions

    @property
    def quant_level(self):
        return self._quant_level

    @quant_level.setter
    def quant_level(self, quant_level):
        self._quant_level = quant_level

    @property
    def use_distributed_tuning(self):
        return self._use_distributed_tuning

    @use_distributed_tuning.setter
    def use_distributed_tuning(self, use_distributed_tuning):
        if _check_value('use_distributed_tuning', use_distributed_tuning, bool):
            self._use_distributed_tuning = use_distributed_tuning

    @property
    def reduce_range(self):
        return self._reduce_range

    @reduce_range.setter
    def reduce_range(self, reduce_range):
        if reduce_range is None or _check_value('reduce_range', reduce_range, bool):
            self._reduce_range = reduce_range

    @property
    def performance_only(self):
        return self._performance_only

    @performance_only.setter
    def performance_only(self, performance_only):
        if _check_value('performance_only', performance_only, bool):
            self._performance_only = performance_only

    @property
    def max_trials(self):
        return self._max_trials

    @max_trials.setter
    def max_trials(self, max_trials):
        if _check_value('max_trials', max_trials, int):
            self._max_trials = max_trials

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, timeout):
        if _check_value('timeout', timeout, int):
            self._timeout = timeout

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, objective):
        if _check_value('objective', objective, str,
            ['performance', 'accuracy', 'modelsize', 'footprint']):
            self._objective = objective

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy):
        if _check_value('strategy', strategy, str,
            ['basic', 'mse', 'bayesian', 'random', 'exhaustive', 'sigopt', 'tpe', 'mse_v2', 'hawq_v2']):
            self._strategy = strategy

    @property
    def strategy_kwargs(self):
        return self._strategy_kwargs

    @strategy_kwargs.setter
    def strategy_kwargs(self, strategy_kwargs):
        self._strategy_kwargs = strategy_kwargs

    @property
    def op_name_dict(self):
        return self._op_name_dict

    @op_name_dict.setter
    def op_name_dict(self, op_name_dict):
        if op_name_dict is None:
            self._op_name_dict = op_name_dict
        elif isinstance(op_name_dict, dict):
            for k, v in op_name_dict.items():
                ops_schema.validate(v)
            self._op_name_dict = op_name_dict
        else:
            assert False, ("Type of op_name_dict should be dict but not {}, ".format(
                type(op_name_dict)))

    @property
    def op_type_dict(self):
        return self._op_type_dict

    @op_type_dict.setter
    def op_type_dict(self, op_type_dict):
        if op_type_dict is None:
            self._op_type_dict = op_type_dict
        elif isinstance(op_type_dict, dict):
            for k, v in op_type_dict.items():
                ops_schema.validate(v)
            self._op_type_dict = op_type_dict
        else:
            assert False, ("Type of op_type_dict should be dict but not {}".format(
                type(op_type_dict)))

    @property
    def calibration_sampling_size(self):
        return self._calibration_sampling_size

    @calibration_sampling_size.setter
    def calibration_sampling_size(self, sampling_size):
        if _check_value('calibration_sampling_size', sampling_size, int):
            if isinstance(sampling_size, int):
                sampling_size = [sampling_size]
            self._calibration_sampling_size = sampling_size

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        if _check_value('device', device, str, ['cpu', 'gpu']):
            self._device = device

    @property
    def quant_format(self):
        return self._quant_format

    @quant_format.setter
    def quant_format(self, quant_format):
        if _check_value('quant_format', quant_format, str,
            ['default', 'QDQ', 'QOperator']):
            self._quant_format = quant_format

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, backend):
        if _check_value('backend', backend, str, [
                'default', 'itex', 'ipex', 'onnxrt_trt_ep', 'onnxrt_cuda_ep']):
            self._backend = backend

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, outputs):
        if _check_value('outputs', outputs, str):
            self._outputs = outputs

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        if _check_value('inputs', inputs, str):
            self._inputs = inputs

    @property
    def example_inputs(self):
        """Get strategy_kwargs."""
        return self._example_inputs

    @example_inputs.setter
    def example_inputs(self, example_inputs):
        """Set example_inputs."""
        self._example_inputs = example_inputs


class BenchmarkConfig:
    """Config Class for Benchmark.

    Args:
        inputs (list, optional): A list of strings containing the inputs of model. Default is an empty list.
        outputs (list, optional): A list of strings containing the outputs of model. Default is an empty list.
        backend (str, optional): Backend name for model execution. Supported values include: 'default', 'itex',
                                'ipex', 'onnxrt_trt_ep', 'onnxrt_cuda_ep'. Default value is 'default'.
        warmup (int, optional): The number of iterations to perform warmup before running performance tests.
                                Default value is 5.
        iteration (int, optional): The number of iterations to run performance tests. Default is -1.
        cores_per_instance (int, optional): The number of CPU cores to use per instance. Default value is None.
        num_of_instance (int, optional): The number of instances to use for performance testing.
                                         Default value is None.
        inter_num_of_threads (int, optional): The number of threads to use for inter-thread operations.
                                              Default value is None.
        intra_num_of_threads (int, optional): The number of threads to use for intra-thread operations.
                                              Default value is None.

    Example::

        # Run benchmark according to config
        from neural_compressor.benchmark import fit

        conf = BenchmarkConfig(iteration=100, cores_per_instance=4, num_of_instance=7)
        fit(model='./int8.pb', config=conf, b_dataloader=eval_dataloader)
    """
    def __init__(self,
                 inputs=[],
                 outputs=[],
                 backend='default',
                 device='cpu',
                 warmup=5,
                 iteration=-1,
                 model=None,
                 model_name='',
                 cores_per_instance=None,
                 num_of_instance=None,
                 inter_num_of_threads=None,
                 intra_num_of_threads=None,
                 diagnosis=False):
        """Init a BenchmarkConfig object."""
        self.inputs = inputs
        self.outputs = outputs
        self.backend = backend
        self.device=device
        self.warmup = warmup
        self.iteration = iteration
        self.model = model
        self.model_name = model_name
        self.cores_per_instance = cores_per_instance
        self.num_of_instance = num_of_instance
        self.inter_num_of_threads = inter_num_of_threads
        self.intra_num_of_threads = intra_num_of_threads
        self.diagnosis = diagnosis
        self._framework = None

    def keys(self):
        """Returns keys of the dict."""
        return ('inputs', 'outputs', 'backend', 'device', 'warmup', 'iteration', 'model', \
                'model_name', 'cores_per_instance', 'num_of_instance', 'framework', \
                'inter_num_of_threads','intra_num_of_threads')

    def __getitem__(self, item):
        """Get the dict."""
        return getattr(self, item)

    @property
    def backend(self):
        """Get backend."""
        return self._backend

    @backend.setter
    def backend(self, backend):
        """Set backend."""
        if _check_value('backend', backend, str, [
                'default', 'itex', 'ipex', 'onnxrt_trt_ep', 'onnxrt_cuda_ep']):
            self._backend = backend

    @property
    def device(self):
        """Get device name."""
        return self._device

    @device.setter
    def device(self, device):
        if _check_value('device', device, str, ['cpu', 'gpu']):
            self._device = device

    @property
    def outputs(self):
        """Get outputs."""
        return self._outputs

    @outputs.setter
    def outputs(self, outputs):
        """Set outputs."""
        if _check_value('outputs', outputs, str):
            self._outputs = outputs

    @property
    def inputs(self):
        """Get inputs."""
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        """Set inputs."""
        if _check_value('inputs', inputs, str):
            self._inputs = inputs

    @property
    def warmup(self):
        """Get warmup."""
        return self._warmup

    @warmup.setter
    def warmup(self, warmup):
        """Set warmup."""
        if _check_value('warmup', warmup, int):
            self._warmup = warmup

    @property
    def iteration(self):
        """Get iteration."""
        return self._iteration

    @iteration.setter
    def iteration(self, iteration):
        """Set iteration."""
        if _check_value('iteration', iteration, int):
            self._iteration = iteration

    @property
    def cores_per_instance(self):
        """Get cores_per_instance."""
        return self._cores_per_instance

    @cores_per_instance.setter
    def cores_per_instance(self, cores_per_instance):
        """Set cores_per_instance."""
        if cores_per_instance is None or _check_value('cores_per_instance', cores_per_instance,
                                                     int):
            self._cores_per_instance = cores_per_instance

    @property
    def num_of_instance(self):
        """Get num_of_instance."""
        return self._num_of_instance

    @num_of_instance.setter
    def num_of_instance(self, num_of_instance):
        """Set num_of_instance."""
        if num_of_instance is None or _check_value('num_of_instance', num_of_instance, int):
            self._num_of_instance = num_of_instance

    @property
    def inter_num_of_threads(self):
        """Get inter_num_of_threads."""
        return self._inter_num_of_threads

    @inter_num_of_threads.setter
    def inter_num_of_threads(self, inter_num_of_threads):
        """Set inter_num_of_threads."""
        if inter_num_of_threads is None or _check_value('inter_num_of_threads',
                                                       inter_num_of_threads, int):
            self._inter_num_of_threads = inter_num_of_threads

    @property
    def intra_num_of_threads(self):
        """Get intra_num_of_threads."""
        return self._intra_num_of_threads

    @intra_num_of_threads.setter
    def intra_num_of_threads(self, intra_num_of_threads):
        """Get intra_num_of_threads."""
        if intra_num_of_threads is None or _check_value('intra_num_of_threads',
                                                       intra_num_of_threads, int):
            self._intra_num_of_threads = intra_num_of_threads

    @property
    def model(self):
        """Get model."""
        return self._model

    @model.setter
    def model(self, model):
        """Set model."""
        self._model = model

    @property
    def model_name(self):
        """Get model name."""
        return self._model_name

    @model_name.setter
    def model_name(self, model_name):
        """Set model name."""
        if _check_value("model_name", model_name, str):
            self._model_name = model_name

    @property
    def framework(self):
        """Set framework."""
        return self._framework

    @framework.setter
    def framework(self, framework):
        """Get framework."""
        self._framework = framework


class QuantizationConfig(_BaseQuantizationConfig):
    def __init__(self,
                 inputs=[],
                 outputs=[],
                 backend='default',
                 device='cpu',
                 approach='post_training_static_quant',
                 calibration_sampling_size=[100],
                 op_type_dict=None,
                 op_name_dict=None,
                 strategy='basic',
                 strategy_kwargs=None,
                 objective='performance',
                 timeout=0,
                 max_trials=100,
                 performance_only=False,
                 reduce_range=None,
                 use_bf16=True,
                 quant_level="auto",
                 accuracy_criterion=accuracy_criterion,
                 diagnosis=False):
        excluded_precisions = ["bf16"] if not use_bf16 else []
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            backend=backend,
            device=device,
            calibration_sampling_size=calibration_sampling_size,
            op_type_dict=op_type_dict,
            op_name_dict=op_name_dict,
            strategy=strategy,
            strategy_kwargs=strategy_kwargs,
            objective=objective,
            timeout=timeout,
            max_trials=max_trials,
            performance_only=performance_only,
            reduce_range=reduce_range,
            excluded_precisions=excluded_precisions,
            accuracy_criterion=accuracy_criterion,
            quant_level=quant_level,
            diagnosis=diagnosis
        )
        self.approach = approach

    @property
    def approach(self):
        return self._approach

    @approach.setter
    def approach(self, approach):
        if _check_value(
            'approach', approach, str,
            ['post_training_static_quant', 'post_training_dynamic_quant', 'quant_aware_training']
        ):
            self._approach = approach


class WeightPruningConfig:
    """Config Class for Pruning. Define a single or a sequence of pruning configs.

    Args:
        pruning_configs (list of dicts, optional): Local pruning configs only valid to linked layers.
            Parameters defined out of pruning_configs are valid for all layers.
            By defining dicts in pruning_config, users can set different pruning strategies for corresponding layers.
            Defaults to [{}].
        target_sparsity (float, optional): Sparsity ratio the model can reach after pruning.
            Supports a float between 0 and 1.
            Default to 0.90.
        pruning_type (str, optional): A string define the criteria for pruning.
            Supports "magnitude", "snip", "snip_momentum",
                     "magnitude_progressive", "snip_progressive", "snip_momentum_progressive", "pattern_lock"
            Default to "snip_momentum", which is the most feasible pruning criteria under most situations.
        pattern (str, optional): Sparsity's structure (or unstructure) types.
            Supports "NxM" (e.g "4x1", "8x1"), "channelx1" & "1xchannel"(channel-wise), "N:M" (e.g "2:4").
            Default to "4x1", which can be directly processed by our kernels in ITREX.
        op_names (list of str, optional): Layers contains some specific names to be included for pruning.
            Defaults to [].
        excluded_op_names: Layers contains some specific names to be excluded for pruning.
            Defaults to [].
        start_step (int, optional): The step to start pruning.
            Supports an integer.
            Default to 0.
        end_step: (int, optional): The step to end pruning.
            Supports an integer.
            Default to 0.
        pruning_scope (str, optional): Determine layers' scores should be gather together to sort
            Supports "global" and "local".
            Default: "global", since this leads to less accuracy loss.
        pruning_frequency: the frequency of pruning operation.
            Supports an integer.
            Default to 1.
        min_sparsity_ratio_per_op (float, optional): Minimum restriction for every layer's sparsity.
            Supports a float between 0 and 1.
            Default to 0.0.
        max_sparsity_ratio_per_op (float, optional): Maximum restriction for every layer's sparsity.
            Supports a float between 0 and 1.
            Default to 0.98.
        sparsity_decay_type (str, optional): how to schedule the sparsity increasing methods.
            Supports "exp", "cube", "cube", "linear".
            Default to "exp".
        pruning_op_types (list of str): Operator types currently support for pruning.
            Supports ['Conv', 'Linear'].
            Default to ['Conv', 'Linear'].

    Example::

        from neural_compressor.config import WeightPruningConfig
        local_configs = [
            {
                "pruning_scope": "local",
                "target_sparsity": 0.6,
                "op_names": ["query", "key", "value"],
                "pattern": "channelx1",
            },
            {
                "pruning_type": "snip_momentum_progressive",
                "target_sparsity": 0.5,
                "op_names": ["self.attention.dense"],
            }
        ]
        config = WeightPruningConfig(
            pruning_configs = local_configs,
            target_sparsity=0.8
        )
        prune = Pruning(config)
        prune.update_config(start_step=1, end_step=10)
        prune.model = self.model
    """

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
        if _check_value('datatype', datatype, str, ['fp32', 'bf16', 'uint8', 'int8']):
            self._datatype = datatype if isinstance(datatype, list) else [datatype]

    @property
    def scheme(self):
        return self._scheme

    @scheme.setter
    def scheme(self, scheme):
        if _check_value('scheme', scheme, str, ['sym', 'asym']):
            self._scheme = scheme if isinstance(scheme, list) else [scheme]

    @property
    def granularity(self):
        return self._granularity

    @granularity.setter
    def granularity(self, granularity):
        if _check_value('granularity', granularity, str, ['per_channel', 'per_tensor']):
            self._granularity = granularity if isinstance(granularity, list) else [granularity]

    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, algorithm):
        if _check_value('algorithm', algorithm, str, ['minmax', 'kl']):
            self._algorithm = algorithm if isinstance(algorithm, list) else [algorithm]


class KnowledgeDistillationLossConfig:
    """Config Class for Knowledge Distillation Loss.

    Args:
        temperature (float, optional): Hyperparameters that control the entropy
            of probability distributions. Defaults to 1.0.
        loss_types (list[str], optional): loss types, should be a list of length 2.
            First item is the loss type for student model output and groundtruth label,
            second item is the loss type for student model output and teacher model output.
            Supported types for first item are "CE", "MSE".
            Supported types for second item are "CE", "MSE", "KL".
            Defaults to ['CE', 'CE'].
        loss_weights (list[float], optional): loss weights, should be a list of length 2 and sum to 1.0.
            First item is the weight multiplied to the loss of student model output and groundtruth label,
            second item is the weight multiplied to the loss of student model output and teacher model output.
            Defaults to [0.5, 0.5].

    Example::

        from neural_compressor.config import DistillationConfig, KnowledgeDistillationLossConfig
        from neural_compressor.training import prepare_compression

        criterion_conf = KnowledgeDistillationLossConfig()
        d_conf = DistillationConfig(teacher_model=teacher_model, criterion=criterion_conf)
        compression_manager = prepare_compression(model, d_conf)
        model = compression_manager.model
    """
    def __init__(self, temperature=1.0, loss_types=['CE', 'CE'], loss_weights=[0.5, 0.5]):
        """Init a KnowledgeDistillationLossConfig object."""
        self.config = DotDict({
            'KnowledgeDistillationLoss': {
                'temperature': temperature,
                'loss_types': loss_types,
                'loss_weights': loss_weights
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

    Example::

        from neural_compressor.training import prepare_compression
        from neural_compressor.config import DistillationConfig, KnowledgeDistillationLossConfig

        distil_loss = KnowledgeDistillationLossConfig()
        conf = DistillationConfig(teacher_model=model, criterion=distil_loss)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
        compression_manager = prepare_compression(model, conf)
        model = compression_manager.model
    """
    def __init__(self,
                 teacher_model=None,
                 criterion=criterion,
                 optimizer={'SGD': {
                     'learning_rate': 0.0001
                 }}):
        """Init a DistillationConfig object."""
        self.criterion = criterion
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
        if _check_value('op_type', op_type, str):
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
        if _check_value('precisions', precisions, str, ['int8', 'uint8', 'fp32', 'bf16', 'fp16']):
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
        if _check_value('graph_optimization_level', graph_optimization_level, str,
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
options = Options()
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
