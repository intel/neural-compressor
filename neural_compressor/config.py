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

        from neural_compressor.utils.utility import set_random_seed, set_workspace, set_resume_from, set_tensorboard
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


options = Options()


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
        if _check_value('backend', backend, str, [
                'default', 'itex', 'ipex', 'onnxrt_trt_ep', 'onnxrt_cuda_ep']):
            self._backend = backend

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
                 'add_qdq_pair_to_weight': whether add QDQ pair for weights, only vaild for onnxrt_trt_ep
                 'optypes_to_exclude_output_quant': don't quantize output of specified optypes
                 'dedicated_qdq_pair': whether dedicate QDQ pair, only vaild for onnxrt_trt_ep
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
                 use_distributed_tuning=False):
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
        self.use_distributed_tuning=use_distributed_tuning
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


class TuningCriterion:
    """Class for Tuning Criterion.

    Args:
        strategy(str, optional): Name of the tuning strategy. Please refer to docs/source/tuning_strategies.md.
                                 Default is 'basic'.
        strategy_kwargs(dict, optional): The strategy setting dictionary.
                                         Please refer to docs/source/tuning_strategies.md.
                                         Default value is None.
        timeout(int, optional): Tuning timeout(seconds). When set to 0, early stopping is enabled.
                                Default value is 0.
        max_trials(int, optional): Max tuning times. Combined with the `timeout` field to decide when to exit tuning.
                                   Default is 100.
        objective(str, optinal): Objective with accuracy constraint guaranteed,
                                     support 'performance', 'modelsize', 'footprint'.
                                 Please refer to docs/source/objective.md.
                                 Default value is 'performance'.

    Example::

        from neural_compressor.config import TuningCriterion

        tuning_criterion=TuningCriterion(
            strategy="basic",
            strategy_kwargs=None,
            timeout=0,
            max_trials=100,
        )
    """
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
        if _check_value('max_trials', max_trials, int):
            self._max_trials = max_trials

    @property
    def timeout(self):
        """Get timeout."""
        return self._timeout

    @timeout.setter
    def timeout(self, timeout):
        """Set timeout."""
        if _check_value('timeout', timeout, int):
            self._timeout = timeout

    @property
    def objective(self):
        """Get objective."""
        return self._objective

    @objective.setter
    def objective(self, objective):
        """Set objective."""
        if _check_value('objective', objective, str,
            ['performance', 'accuracy', 'modelsize', 'footprint']):
            self._objective = objective

    @property
    def strategy(self):
        """Get strategy."""
        return self._strategy

    @strategy.setter
    def strategy(self, strategy):
        """Set strategy."""
        if _check_value('strategy', strategy, str,
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
    """Config Class for Post Training Quantization.

    Args:
        device: Support 'cpu' and 'gpu'.
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
                 'add_qdq_pair_to_weight': whether add QDQ pair for weights, only vaild for onnxrt_trt_ep
                 'optypes_to_exclude_output_quant': don't quantize output of specified optypes
                 'dedicated_qdq_pair': whether dedicate QDQ pair, only vaild for onnxrt_trt_ep
        quant_format: Support 'default', 'QDQ' and 'QOperator', only required in ONNXRuntime.
        inputs: Inputs of model, only required in tensorflow.
        outputs: Outputs of model, only required in tensorflow.
        approach: Post-Training Quantization method. Neural compressor support 'static', 'dynamic' and 'auto' method.
                  Default value is 'auto'.
                  For strategy 'basic', 'auto' method means neural compressor will quantize all OPs support PTQ static
                      or PTQ dynamic. For OPs supporting both PTQ static and PTQ dynamic,
                      PTQ static will be tried first, and PTQ dynamic will be tried when none of the OP type wise
                      tuning configs meet the accuracy loss criteria.
                  For strategy 'bayesian', 'mse', 'mse_v2' and 'HAWQ_V2', 'exhaustive', and 'random',
                      'auto' means neural compressor will quantize all OPs support PTQ static or PTQ dynamic.
                      if OPs supporting both PTQ static and PTQ dynamic, PTQ static will be tried, else PTQ dynamic
                      will be tried.
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
        reduce_range: Whether use 7 bit to quantization.
        excluded_precisions: Precisions to be excluded, Default value is empty list.
                             Neural compressor enable the mixed precision with fp32 + bf16 + int8 by default.
                             If you want to disable bf16 data type, you can specify excluded_precisions = ['bf16].
        quant_level: Support auto, 0 and 1, 0 is conservative strategy, 1 is basic or user-specified 
                     strategy, auto (default) is the combination of 0 and 1.
        tuning_criterion: Instance of TuningCriterion class. In this class you can set strategy, strategy_kwargs,
                              timeout, max_trials and objective.
                          Please refer to docstring of TuningCriterion class.
        accuracy_criterion: Instance of AccuracyCriterion class. In this class you can set higher_is_better,
                                criterion and tolerable_loss.
                            Please refer to docstring of AccuracyCriterion class.
        use_distributed_tuning: Whether use distributed tuning or not.

    Example::

        from neural_compressor.config PostTrainingQuantConfig, TuningCriterion

        conf = PostTrainingQuantConfig(
            quant_level="auto",
            tuning_criterion=TuningCriterion(
                timeout=0,
                max_trials=100,
            ),
        )
    """
    def __init__(self,
                 device="cpu",
                 backend="default",
                 domain="auto",
                 recipes={},
                 quant_format="default",
                 inputs=[],
                 outputs=[],
                 approach="static",
                 calibration_sampling_size=[100],
                 op_type_dict=None,
                 op_name_dict=None,
                 reduce_range=None,
                 excluded_precisions=[],
                 quant_level="auto",
                 tuning_criterion=tuning_criterion,
                 accuracy_criterion=accuracy_criterion,
                 use_distributed_tuning=False,
    ):
        """Init a PostTrainingQuantConfig object."""
        self.tuning_criterion = tuning_criterion
        super().__init__(inputs=inputs,
                         outputs=outputs,
                         device=device,
                         backend=backend,
                         domain=domain,
                         recipes=recipes,
                         quant_format=quant_format,
                         calibration_sampling_size=calibration_sampling_size,
                         op_type_dict=op_type_dict,
                         op_name_dict=op_name_dict,
                         strategy=tuning_criterion.strategy,
                         strategy_kwargs=tuning_criterion.strategy_kwargs,
                         objective=tuning_criterion.objective,
                         timeout=tuning_criterion.timeout,
                         max_trials=tuning_criterion.max_trials,
                         reduce_range=reduce_range,
                         excluded_precisions=excluded_precisions,
                         quant_level=quant_level,
                         accuracy_criterion=accuracy_criterion,
                         use_distributed_tuning=use_distributed_tuning)
        self.approach = approach

    @property
    def approach(self):
        """Get approach."""
        return self._approach

    @approach.setter
    def approach(self, approach):
        """Set approach."""
        if _check_value("approach", approach, str, ["static", "dynamic", "auto"]):
            self._approach = QUANTMAPPING[approach]

    @property
    def tuning_criterion(self):
        """Get tuning_criterion."""
        return self._tuning_criterion

    @tuning_criterion.setter
    def tuning_criterion(self, tuning_criterion):
        """Set tuning_criterion."""
        if _check_value("tuning_criterion", tuning_criterion, TuningCriterion):
            self._tuning_criterion = tuning_criterion


class QuantizationAwareTrainingConfig(_BaseQuantizationConfig):
    """Config Class for Quantization Aware Training.

    Args:
        device: Support 'cpu' and 'gpu'.
        backend: Backend for model execution. Support 'default', 'itex', 'ipex', 'onnxrt_trt_ep', 'onnxrt_cuda_ep'
        inputs: Inputs of model, only required in tensorflow.
        outputs: Outputs of model, only required in tensorflow.
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
        reduce_range: Whether use 7 bit to quantization.
        excluded_precisions: Precisions to be excluded, Default value is empty list.
                             Neural compressor enable the mixed precision with fp32 + bf16 + int8 by default.
                             If you want to disable bf16 data type, you can specify excluded_precisions = ['bf16].
        quant_level: Support auto, 0 and 1, 0 is conservative strategy, 1 is basic or user-specified 
                     strategy, auto (default) is the combination of 0 and 1.
        tuning_criterion: Instance of TuningCriterion class. In this class you can set strategy, strategy_kwargs,
                              timeout, max_trials and objective.
                          Please refer to docstring of TuningCriterion class.
                          This parameter only required by Quantization Aware Training with tuning.
        accuracy_criterion: Instance of AccuracyCriterion class. In this class you can set higher_is_better,
                                criterion and tolerable_loss.
                            Please refer to docstring of AccuracyCriterion class.
                            This parameter only required by Quantization Aware Training with tuning.

    Example::

        from neural_compressor.config import QuantizationAwareTrainingConfig

        if approach == "qat":
            model = copy.deepcopy(model_origin)
            conf = QuantizationAwareTrainingConfig(
                op_name_dict=qat_op_name_dict
            )
            compression_manager = prepare_compression(model, conf)
    """
    def __init__(self,
                 device="cpu",
                 backend="default",
                 inputs=[],
                 outputs=[],
                 op_type_dict=None,
                 op_name_dict=None,
                 reduce_range=None,
                 excluded_precisions=[],
                 quant_level="auto",
                 tuning_criterion=tuning_criterion,
                 accuracy_criterion=accuracy_criterion):
        """Init a QuantizationAwareTrainingConfig object."""
        super().__init__(inputs=inputs,
                         outputs=outputs,
                         device=device,
                         backend=backend,
                         op_type_dict=op_type_dict,
                         op_name_dict=op_name_dict,
                         strategy=tuning_criterion.strategy,
                         strategy_kwargs=tuning_criterion.strategy_kwargs,
                         objective=tuning_criterion.objective,
                         timeout=tuning_criterion.timeout,
                         max_trials=tuning_criterion.max_trials,
                         reduce_range=reduce_range,
                         excluded_precisions=excluded_precisions,
                         accuracy_criterion=accuracy_criterion,
                         quant_level=quant_level)
        self._approach = 'quant_aware_training'

    @property
    def approach(self):
        """Get approach."""
        return self._approach


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


class KnowledgeDistillationLossConfig:
    """Config Class for Knowledge Distillation Loss.

    Args:
        temperature (float, optional): Hyperparameters that control the entropy
            of probability distributions. Defaults to 1.0.
        loss_types (list[str], optional): loss types, should be a list of length 2.
            First item is the loss type for student model output and groundtruth label,
            second item is the loss type for student model output and teacher model output.
            Supported tpyes for first item are "CE", "MSE". 
            Supported tpyes for second item are "CE", "MSE", "KL".
            Defaults to ['CE', 'CE'].
        loss_weights (list[float], optional): loss weights, should be a list of length 2 and sum to 1.0.
            First item is the weight multipled to the loss of student model output and groundtruth label,
            second item is the weight multipled to the loss of student model output and teacher model output.
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


class IntermediateLayersKnowledgeDistillationLossConfig:
    """Config Class for Intermediate Layers Knowledge Distillation Loss.

    Args:
        layer_mappings (list): A list for specifying the layer mappings relationship between
            the student model and the teacher model. Each item in layer_mappings should be a
            list with the format [(student_layer_name, student_layer_output_process),
            (teacher_layer_name, teacher_layer_output_process)], where the student_layer_name
            and the teacher_layer_name are the layer names of the student and the teacher models,
            e.g. 'bert.layer1.attention'. The student_layer_output_process and teacher_layer_output_process
            are output process method to get the desired output from the layer specified in the layer
            name, its value can be either a function or a string, in function case, the function
            takes output of the specified layer as input, in string case, when output of the
            specified layer is a dict, this string will serve as key to get corresponding value,
            when output of the specified layer is a list or tuple, the string should be numeric and
            will serve as the index to get corresponding value. 
            When output process is not needed, the item in layer_mappings can be abbreviated to
            [(student_layer_name, ), (teacher_layer_name, )], if student_layer_name and teacher_layer_name
            are the same, it can be abbreviated further to [(layer_name, )].
            Some examples of the item in layer_mappings are listed below:
              [('student_model.layer1.attention', '1'), ('teacher_model.layer1.attention', '1')]
              [('student_model.layer1.output', ), ('teacher_model.layer1.output', )].
              [('model.layer1.output', )].
        loss_types (list[str], optional): loss types, should be a list with the same length of
            layer_mappings. Each item is the loss type for each layer mapping specified in the
            layer_mappings. Supported tpyes for each item are "MSE", "KL", "L1". Defaults to
            ["MSE", ]*len(layer_mappings).
        loss_weights (list[float], optional): loss weights, should be a list with the same length of
            layer_mappings. Each item is the weight multipled to the loss of each layer mapping specified
            in the layer_mappings. Defaults to [1.0 / len(layer_mappings)] * len(layer_mappings).
        add_origin_loss (bool, optional): Whether to add origin loss of the student model. Defaults to False.

    Example::

        from neural_compressor.config import DistillationConfig, IntermediateLayersKnowledgeDistillationLossConfig
        from neural_compressor.training import prepare_compression

        criterion_conf = IntermediateLayersKnowledgeDistillationLossConfig(
            layer_mappings=[['layer1.0', ],
                            [['layer1.1.conv1', ], ['layer1.1.conv1', '0']],],
            loss_types=['MSE']*len(layer_mappings),
            loss_weights=[1.0 / len(layer_mappings)]*len(layer_mappings),
            add_origin_loss=True
        )
        d_conf = DistillationConfig(teacher_model=teacher_model, criterion=criterion_conf)
        compression_manager = prepare_compression(model, d_conf)
        model = compression_manager.model
    """
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
    """Config Class for Self Knowledge Distillation Loss.

    Args:
        layer_mappings (list): layers of distillation. Format like
                [[[student1_layer_name1, teacher_layer_name1],[student2_layer_name1, teacher_layer_name1]],
                [[student1_layer_name2, teacher_layer_name2],[student2_layer_name2, teacher_layer_name2]]]
        temperature (float, optional): use to calculate the soft label CE.
        loss_types (list, optional):  loss types, should be a list with the same length of
            layer_mappings. Each item is the loss type for each layer mapping specified in the
            layer_mappings. Supported tpyes for each item are "CE", "KL", "L2". Defaults to
            ["CE", ]*len(layer_mappings).
        loss_weights (list, optional): loss weights. Defaults to [1.0 / len(layer_mappings)] *
            len(layer_mappings).
        add_origin_loss (bool, optional): whether to add origin loss for hard label loss.

    Example::

        from neural_compressor.training import prepare_compression
        from neural_compressor.config import DistillationConfig, SelfKnowledgeDistillationLossConfig

        criterion_conf = SelfKnowledgeDistillationLossConfig(
            layer_mappings=[
                [['resblock.1.feature.output', 'resblock.deepst.feature.output'],
                ['resblock.2.feature.output','resblock.deepst.feature.output']],
                [['resblock.2.fc','resblock.deepst.fc'],
                ['resblock.3.fc','resblock.deepst.fc']],
                [['resblock.1.fc','resblock.deepst.fc'],
                ['resblock.2.fc','resblock.deepst.fc'],
                ['resblock.3.fc','resblock.deepst.fc']]
            ],
            temperature=3.0,
            loss_types=['L2', 'KL', 'CE'],
            loss_weights=[0.5, 0.05, 0.02],
            add_origin_loss=True,)
        conf = DistillationConfig(teacher_model=model, criterion=criterion_conf)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
        compression_manager = prepare_compression(model, conf)
        model = compression_manager.model
    """
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

    Example::

        from neural_compressor.training import prepare_compression
        from neural_compressor.config import DistillationConfig, SelfKnowledgeDistillationLossConfig

        distil_loss = SelfKnowledgeDistillationLossConfig()
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
    """Config Class for MixedPrecision.
    
    Args:
        device (str, optional): Device for execution.
                                Support 'cpu' and 'gpu', default is 'cpu'.
        backend (str, optional): Backend for model execution.
                                 Support 'default', 'itex', 'ipex', 'onnxrt_trt_ep', 'onnxrt_cuda_ep',
                                 default is 'default'.
        precision (str, optional): Target precision for mix precision conversion.
                                   Support 'bf16' and 'fp16', default is 'bf16'.
        inputs (list, optional): Inputs of model, default is [].
        outputs (list, optional): Outputs of model, default is [].
        tuning_criterion (TuningCriterion object, optional): Accuracy tuning settings,
                                                             it won't work if there is no accuracy tuning process.
        accuracy_criterion (AccuracyCriterion object, optional): Accuracy constraint settings,
                                                                 it won't work if there is no accuracy tuning process.
        excluded_precisions (list, optional): Precisions to be excluded during mix precision conversion, default is [].

    Example::

        from neural_compressor import mix_precision
        from neural_compressor.config import MixedPrecisionConfig

        conf = MixedPrecisionConfig()
        converted_model = mix_precision.fit(model, config=conf)
    """
    def __init__(self,
                 device="cpu",
                 backend="default",
                 precision="bf16",
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
        self.precision = precision

    @property
    def precision(self):
        """Get precision."""
        return self._precision

    @precision.setter
    def precision(self, precision):
        """Set precision."""
        if isinstance(precision, str):
            assert precision in ["fp16", "bf16"], "Only support 'fp16' and 'bf16' for mix precision."
            self._precision = [precision]
        elif isinstance(precision, list):
            assert all([i in ["fp16", "bf16"] for i in precision]), "Only " \
                "support 'fp16' and 'bf16' for mix precision."
            self._precision = precision

class ExportConfig:
    """Common Base Config for Export.

    Args:
        dtype (str, optional): The data type of the exported model, select from ["fp32", "int8"]. 
                               Defaults to "int8".
        opset_version (int, optional): The ONNX opset version used for export. Defaults to 14.
        quant_format (str, optional): The quantization format of the exported int8 onnx model, 
                                      select from ["QDQ", "QLinear"]. Defaults to "QDQ".
        example_inputs (tensor|list|tuple|dict, optional): Example inputs used for tracing model. Defaults to None.
        input_names (list, optional): A list of model input names. Defaults to None.
        output_names (list, optional): A list of model output names. Defaults to None.
        dynamic_axes (dict, optional): A dictionary of dynamic axes information. Defaults to None.
    """
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
    """Config Class for ONNXQlinear2QDQ.
    
    Example::

        from neural_compressor.config import ONNXQlinear2QDQConfig
        from neural_compressor.model import Model
        
        conf = ONNXQlinear2QDQConfig()
        model = Model(model)
        model.export('new_model.onnx', conf)
    """
    def __init__(self):
        """Init an ONNXQlinear2QDQConfig object."""
        pass

class Torch2ONNXConfig(ExportConfig):
    """Config Class for Torch2ONNX.

    Args:
        dtype (str, optional): The data type of the exported model, select from ["fp32", "int8"]. 
                               Defaults to "int8".
        opset_version (int, optional): The ONNX opset version used for export. Defaults to 14.
        quant_format (str, optional): The quantization format of the exported int8 onnx model, 
                                      select from ["QDQ", "QLinear"]. Defaults to "QDQ".
        example_inputs (tensor|list|tuple|dict, required): Example inputs used for tracing model. Defaults to None.
        input_names (list, optional): A list of model input names. Defaults to None.
        output_names (list, optional): A list of model output names. Defaults to None.
        dynamic_axes (dict, optional): A dictionary of dynamic axes information. Defaults to None.
        recipe (str, optional): A string to select recipes used for Linear -> Matmul + Add, select from 
                                ["QDQ_OP_FP32_BIAS", "QDQ_OP_INT32_BIAS", "QDQ_OP_FP32_BIAS_QDQ"]. 
                                Defaults to 'QDQ_OP_FP32_BIAS'.

    Example:
        # resnet50
        from neural_compressor.config import Torch2ONNXConfig
        int8_onnx_config = Torch2ONNXConfig(
            dtype="int8",
            opset_version=14,
            quant_format="QDQ", # or QLinear
            example_inputs=torch.randn(1, 3, 224, 224),
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={"input": {0: "batch_size"},
                            "output": {0: "batch_size"}},
        )
        q_model.export('int8-model.onnx', int8_onnx_config)
    """
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
    """Config Class for TF2ONNX.

    Args:
        dtype (str, optional): The data type of export target model. Supports 'fp32' and 'int8'.
                               Defaults to 'int8'.
        opset_version (int, optional): The version of the ONNX operator set to use. Defaults to 14.
        quant_format (str, optional): The quantization format for the export target model.
                                      Supports 'default', 'QDQ' and 'QOperator'. Defaults to 'QDQ'.
        example_inputs (list, optional): A list example inputs to use for tracing the model.
                                        Defaults to None.
        input_names (list, optional): A list of model input names. Defaults to None.
        output_names (list, optional): A list of model output names. Defaults to None.
        dynamic_axes (dict, optional): A dictionary of dynamic axis information. Defaults to None.
        **kwargs: Additional keyword arguments.

    Examples::

        # tensorflow QDQ int8 model 'q_model' export to ONNX int8 model
        from neural_compressor.config import TF2ONNXConfig
        config = TF2ONNXConfig()
        q_model.export(output_graph, config)
    """
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
