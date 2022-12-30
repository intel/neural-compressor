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

"""The base class for tuning strategy."""

from abc import abstractmethod
from enum import EnumMeta
import os
import math
import copy
import pickle
from collections import OrderedDict, defaultdict
from pathlib import Path
import yaml
import numpy as np
from typing import OrderedDict as T_OrderedDict

from neural_compressor.adaptor.tensorflow import TensorFlowAdaptor
from ..objective import MultiObjective
from ..adaptor import FRAMEWORKS
from ..utils.utility import Statistics, dump_data_to_local
from ..utils.utility import fault_tolerant_file, equal_dicts, GLOBAL_STATE, MODE
from ..utils.create_obj_from_config import create_eval_func, create_train_func
from ..utils import logger
from ..utils import OPTIONS
from ..version import __version__
from ..conf.dotdict import DotDict, deep_get, deep_set
from ..algorithm import AlgorithmScheduler
from ..algorithm.fast_bias_correction import FastBiasCorrection

import copy
import numpy as np
from collections import OrderedDict
from time import time
from ..utils import logger


from .utils.tuning_space import TuningItem, TuningSpace
from .utils.tuning_structs import OpTuningConfig


STRATEGIES = {}


def strategy_registry(cls):
    """Class decorator used to register all TuneStrategy subclasses.

    Args:
        cls (class): The class of register.

    Returns:
        cls: The class of register.
    """
    assert cls.__name__.endswith(
        'TuneStrategy'
    ), "The name of subclass of TuneStrategy should end with \'TuneStrategy\' substring."
    if cls.__name__[:-len('TuneStrategy')].lower() in STRATEGIES:
        raise ValueError('Cannot have two strategies with the same name')
    STRATEGIES[cls.__name__[:-len('TuneStrategy')].lower()] = cls
    return cls

@strategy_registry
class TuneStrategy(object):
    """Basic class for tuning strategy."""

    def __init__(self, model, conf, q_dataloader=None, q_func=None, eval_dataloader=None, 
                 eval_func=None, resume=None, q_hooks=None):
        """Init the TuneStrategy.

        Args:
            model: The FP32 model specified for low precision tuning.
            conf: The Conf class instance includes all user configurations.
            q_dataloader: Data loader for calibration, mandatory for post-training quantization.  Defaults to None.
            q_func: Training function for quantization aware training. Defaults to None. Defaults to None.
            eval_dataloader: Data loader for evaluation. Defaults to None.
            eval_func: The evaluation function provided by user. This function takes model as parameter, and 
                evaluation dataset and metrics should be encapsulated in this function implementation and 
                outputs a higher-is-better accuracy scalar value.
            resume: The dict containing resume information. Defaults to None.
            q_hooks: The dict of training hooks, supported keys are: on_epoch_begin, on_epoch_end, on_step_begin,
                on_step_end. Their values are functions to be executed in adaptor layer.. Defaults to None.
        """
        self.model = model
        self.cfg = conf.usr_cfg
        self.history_path = self._create_path(self.cfg.tuning.workspace.path, './history.snapshot')
        self.deploy_path = self._create_path(self.cfg.tuning.workspace.path, 'deploy.yaml')
        logger.debug("Dump user yaml configuration:")
        logger.debug(self.cfg)

        self.eval_dataloader = eval_dataloader
        self.calib_dataloader = q_dataloader
        self.q_func = q_func
        self.q_hooks = q_hooks
        self.eval_func = eval_func
        GLOBAL_STATE.STATE = MODE.QUANTIZATION
        framework, framework_specific_info = self._set_framework_info(q_dataloader, q_func)
        self.adaptor = FRAMEWORKS[framework](framework_specific_info)
        self.framework = framework

        self.set_q_func()
        self._set_objectives()
        self.tune_data = {}
        self.tune_result_record = []
        self.tuning_history = []
        self.tuning_result_data = []
        # The tuning history ever made, structured like below:
        # [
        #   {
        #     'version': __version__,
        #     'cfg': cfg1,
        #     'framework': tensorflow
        #     'baseline': baseline1,
        #     'last_tune_result': last_tune_result1,
        #     'best_tune_result': best_tune_result1,
        #     'history': [
        #                  # tuning history under same yaml config
        #                  {'tune_cfg': tune_cfg1, 'tune_result': \
        #                               tune_result1, 'q_config': q_config1, ...},

        #                   ...,
        #                ],
        #     # new fields added by subclass for resuming
        #     ...,
        #   },
        #   # tuning history under different yaml configs
        #   ...,
        # ]

        self.baseline = None
        self.last_tune_result = None
        self.last_qmodel = None
        self.best_tune_result = None
        self.best_qmodel = None
        self.cur_best_acc = self.initial_best_acc() # track the current best accuracy
        self.cur_best_tuning_cfg = {} # track tuning cfg with the current best accuracy
        self.cur_best_qmodel = None   # track quantized model with the current best accuracy
        self.re_quant = False

        self.capability = self.adaptor.query_fw_capability(model)
        logger.debug(self.capability)
        self.set_tuning_space(conf)

        self.algo = AlgorithmScheduler(self.cfg.quantization.recipes)
        self.algo.dataloader = self.calib_dataloader  # reuse the calibration iteration
        self.algo.origin_model = self.model
        self.algo.adaptor = self.adaptor

        self._optype_statistics = None
        self.fallback_stats_baseline = None
        self.fallback_stats = None
        self.tuning_times = 0
        self.fallback_start_point = 0
        self.metric_met_point = 0

        if resume is not None: self.setup_resume(resume)


    @abstractmethod
    def next_tune_cfg(self):
        """Interface for generate the next tuning config.

        The generator of yielding next tuning config to traverse by concrete strategies or quantization level
        according to last tuning result and traverse logic.

        It should be implemented by the sub-class.

        Yields:
            tune_config (dict): It's a dict containing the tuning configuration to traverse.
        """
        raise NotImplementedError


    def traverse(self):
        """Traverse the tuning space.
        
        The main traverse logic which could be override by some concrete strategy which needs more hooks.
        """
        if not (self.cfg.evaluation and self.cfg.evaluation.accuracy and \
            (self.cfg.evaluation.accuracy.metric or self.cfg.evaluation.accuracy.multi_metrics)) \
            and self.eval_func is None:
            logger.info("Neither evaluation function nor metric is defined." \
                        " Generate a quantized model with default quantization configuration.")
            self.cfg.tuning.exit_policy.performance_only = True
            logger.info("Force setting 'tuning.exit_policy.performance_only = True'.")
            logger.info("Generate a fake evaluation function.")
            self.eval_func = self._fake_eval_func

        # get fp32 model baseline
        if self.baseline is None:
            logger.info("Get FP32 model baseline.")
            self._fp32_model = self.model
            self.baseline = self._evaluate(self.model)       
            self.objectives.baseline = self.baseline
            # record the FP32 baseline
            self._add_tuning_history()
        self.show_baseline_info()

        trials_count = 0
        traverse_start_time = time()
        for op_tuning_cfg in self.next_tune_cfg():
            tuning_start_time = time()
            tune_cfg = self._tune_cfg_converter(op_tuning_cfg)
            trials_count += 1
            tuning_history = self._find_tuning_history(tune_cfg)
            if tuning_history and trials_count < self.cfg.tuning.exit_policy.max_trials:
                self.last_tune_result = tuning_history['last_tune_result']
                self.best_tune_result = tuning_history['best_tune_result']
                logger.warn("Find evaluated tuning config, skip.")
                continue
            logger.debug("Dump current tuning configuration:")
            logger.debug(tune_cfg)

            self.tuning_times += 1
            self.q_model = self.adaptor.quantize(
                copy.deepcopy(tune_cfg), self.model, self.calib_dataloader, self.q_func)
            self.algo.calib_iter = tune_cfg['calib_iteration']
            self.algo.q_model = self.q_model
            # TODO align the api to let strategy has access to pre_optimized model
            assert self.adaptor.pre_optimized_model
            self.algo.origin_model = self.adaptor.pre_optimized_model
            if self.cfg.quantization.recipes.fast_bias_correction:
                self.algo.algorithms[0].quantization_cfg = tune_cfg
            self.last_qmodel = self.algo()
            assert self.last_qmodel
            self.last_tune_result = self._evaluate(self.last_qmodel)
            self.cur_best_acc, self.cur_best_tuning_cfg = self.update_best_op_tuning_cfg(op_tuning_cfg)
            need_stop = self.stop(self.cfg.tuning.exit_policy.timeout, trials_count)

            # record the tuning history
            saved_tune_cfg = copy.deepcopy(tune_cfg)
            saved_last_tune_result = copy.deepcopy(self.last_tune_result)
            self._add_tuning_history(saved_tune_cfg,
                                    saved_last_tune_result,
                                    q_config=self.q_model.q_config)
            self.tune_result_record.append(copy.deepcopy(self.last_tune_result))
            self.tune_cfg = tune_cfg
            now_time = time()
            acc_res_msg = ""
            performace_res_msg = ""
            if self.tuning_result_data:
                acc_res_msg = "[ " + "| ".join(self.tuning_result_data[0]) + " ]"
                performace_res_msg = "[ " + "| ".join(self.tuning_result_data[1]) + " ]"
            logger.debug(f"*** The accuracy of last tuning is: {acc_res_msg}")
            logger.debug(f"*** The perfomance of last tuning is: {performace_res_msg}")
            logger.debug(f"*** The last tuning time: {(now_time - tuning_start_time):.2f} s")
            logger.debug(f"*** The tuning process lasted time: {(now_time - traverse_start_time):.2f} s")
            
            self._dump_tuning_process_statistics()
            if need_stop:
                if self.re_quant:
                    logger.info("*** Do not stop the tuning process, re-quantize the ops.")
                    continue
                if self.cfg.tuning.diagnosis and self.cfg.tuning.diagnosis.diagnosis_after_tuning:
                    logger.debug(f'*** Start to do diagnosis (inspect tensor).')
                    self._diagnosis()
                if self.use_multi_objective and len(self.tune_result_record) > 1 and \
                    self.best_tune_result is not None:
                    best_trail, best_result = self.objectives.best_result(self.tune_result_record,
                                                                          copy.deepcopy(self.baseline))
                    if best_result != self.best_tune_result:
                        from neural_compressor.utils.utility import recover
                        self.best_qmodel = recover(self.model.model, 
                            os.path.join(self.cfg.tuning.workspace.path, 'history.snapshot'),
                            best_trail)
                        logger.debug(f"*** Update the best qmodel by recovering from history.")
                        self.best_tune_result = best_result
                    self._dump_tuning_process_statistics()
                break


    def _fallback_started(self):
        self.fallback_start_point = self.tuning_times

    def _update_optype_statistics(self):
        self._optype_statistics = defaultdict(lambda:defaultdict(int))

        for op_name_type, op_tune_cfg in self.tune_cfg['op'].items():
            optype = op_name_type[1]
            quant_mode = op_tune_cfg['activation']['quant_mode']
            if isinstance(quant_mode, tuple) or isinstance(quant_mode, list):
                quant_mode = quant_mode[0]
            dtype = 'INT8' if quant_mode in ('static', 'dynamic') \
                    else quant_mode.upper()
            self._optype_statistics[optype]['Total'] += 1
            self._optype_statistics[optype][dtype] += 1
        return

    def _dump_tuning_process_statistics(self):
        self._update_optype_statistics()
        
        logger.debug("Current tuning process statistics:")
        logger.debug(f"Total Tuning Times: {self.tuning_times}")
        logger.debug("Fallback started at Tune {}".format(self.fallback_start_point))
        logger.debug("Objective(s) met at Tune {}".format(self.metric_met_point))

        fallback_stats = self._calculate_fallback_op_count()
        if self.fallback_stats_baseline == None: 
            self.fallback_stats_baseline = fallback_stats
        logger.debug(f"Fallbacked ops count: {self.fallback_stats_baseline - fallback_stats}")

        if isinstance(self.adaptor, TensorFlowAdaptor):
            self._compare_optype_statistics()
        
        return

    def _calculate_fallback_op_count(self, target_dtype='INT8'):
        fallback_stats = defaultdict(int)
        
        for optype in self._optype_statistics:
            for dtype, count in self._optype_statistics[optype].items():
                fallback_stats[dtype] += count

        return fallback_stats[target_dtype]

    
    def _compare_optype_statistics(self, fields=None, optypes=None,
                                   skip_fields=None, skip_optypes=None):
        assert(fields == None or skip_fields == None)
        assert(optypes == None or skip_optypes == None)
        if not isinstance(self.adaptor, TensorFlowAdaptor):
            logger.debug("OpType statistics comparation is only available for TensorFlow adaptor.")
            return

        adaptor_statistics = self.adaptor.optype_statistics

        def _field_skipped(field):
            if fields != None:
                return field not in fields
            elif skip_fields != None:
                return field in skip_fields

        def _optype_skipped(optype):
            if optypes != None:
                return optype not in optypes
            elif skip_optypes != None:
                return optype in skip_optypes
        

        field_names = adaptor_statistics[0][1:]
        adaptor_data = {
            line[0].lower() : {dtype : count for dtype, count in zip(field_names, line[1:])}
        for line in adaptor_statistics[1]}
        strategy_data = self._optype_statistics
            
        # compare adaptor statistics to strategy statistics
        logger.debug("Statistics difference between adaptor and tuning config:")
        has_difference = False
        difference_count = 0
        for optype in adaptor_data:
            if optype not in strategy_data or _optype_skipped(optype): continue
            for field in field_names:
                if _field_skipped(field): continue
                adaptor_count = adaptor_data[optype][field]
                strategy_count = strategy_data[optype][field]
                if adaptor_count != strategy_count:
                    has_difference = True 
                    if field == 'INT8':
                        difference_count += abs(strategy_count - adaptor_count)                   
                    logger.debug("\t{}: [adaptor: {} | tune_cfg: {}]".format(
                        (optype, field), adaptor_count, strategy_count))
        if not has_difference:
            logger.debug("\tNone")
        logger.debug(f"\tDifference(s) in total: {difference_count}")
        return
        
    def initial_tuning_cfg(self):
        """Init the tuning config.
        
        Initialize the tuning config according to the quantization approach.

        Returns:
            op_item_dtype_dict (OrderedDict): key is (op_name, op_type); value is quantization mode.
            quant_mode_wise_items (OrderedDict): key is quant_mode/precision; value is item list.
            initial_op_tuning_cfg (OrderedDict): key is (op_name, op_type); value is the initialized tuning config.
        """
        if self.cfg.quantization.approach == 'post_training_auto_quant':
            query_order = ['static', 'dynamic', 'bf16', 'fp32']
        elif self.cfg.quantization.approach == 'post_training_dynamic_quant':
            query_order = ['dynamic', 'bf16', 'fp32']
        elif self.cfg.quantization.approach == 'post_training_static_quant':
            query_order = ['static', 'bf16', 'fp32']
        elif self.cfg.quantization.approach == 'quant_aware_training':
            query_order = ['static', 'dynamic', 'bf16', 'fp32']

        quant_mode_wise_items = OrderedDict()
        pre_items = set()
        for quant_mode in query_order:
            items = self.tuning_space.query_items_by_quant_mode(quant_mode)
            filtered_items = [item for item in items if item not in pre_items]
            pre_items = pre_items.union(set(items))
            quant_mode_wise_items[quant_mode] = filtered_items

        def initial_op_quant_mode(items_lst, target_quant_mode, op_item_dtype_dict):
            for item in items_lst:
                op_item_dtype_dict[item.name] = target_quant_mode

        op_item_dtype_dict = OrderedDict()
        for quant_mode, quant_mode_items in quant_mode_wise_items.items():
            initial_op_quant_mode(quant_mode_items, quant_mode, op_item_dtype_dict)

        initial_op_tuning_cfg = {}
        for op_name_dtype, quant_mode in op_item_dtype_dict.items():
            initial_op_tuning_cfg[op_name_dtype] = OpTuningConfig(op_name_dtype[0], op_name_dtype[1], 
                                                                  quant_mode, self.tuning_space)
        return op_item_dtype_dict, quant_mode_wise_items, initial_op_tuning_cfg

    def show_baseline_info(self):
        """Display the accuracy and duration of the the baseline model."""
        if self.baseline:
            self.tune_data['baseline'] = self.baseline[0] if \
                isinstance(self.baseline[0], list) else [self.baseline[0]]
            for name, data in zip(self.metric_name, self.tune_data['baseline']):
                self.tune_data[name] = [data]
            if self.metric_weight:
                # baseline is weighted accuracy
                self.tune_data['Weighted accuracy'] = \
                    [np.mean(np.array(self.tune_data['baseline']) * self.metric_weight)]
                self.tune_data['baseline'] = self.tune_data['Weighted accuracy']
            baseline_msg = '[Accuracy:' + \
                ''.join([' {:.4f}'.format(i) for i in self.tune_data['baseline']]) + \
                ''.join([', {}: {:.4f}'.format(x,y) for x,y in zip( \
                self.objectives.representation, self.baseline[1]) if x != 'Accuracy']) + ']'
        else: # pragma: no cover
            if self.metric_weight:
                self.tune_data['Weighted accuracy'] = ['n/a']
            self.tune_data['baseline'] = ['n/a']

            for name, data in zip(self.metric_name, self.tune_data['baseline']):
                self.tune_data[name] = ['n/a']
            baseline_msg = 'n/a'
        logger.info("FP32 baseline is: {}".format(baseline_msg))

    def initial_best_acc(self):
        """Init the best accuracy.

        Returns:
            The initial value of best accuracy.
        """
        if len(self.metric_name) == 1 or self.metric_weight is not None:
            best_acc = float('-inf') if self.higher_is_better else float('inf')
        else:
            best_acc = [float('-inf') if higher_is_better else float('inf') for \
                        higher_is_better in self.metric_criterion]
        return best_acc

    def _tune_cfg_converter(self, op_tuning_cfg):
        """Convert op_tuning_cfg for adaptor.

        Args:
            op_tuning_cfg (Dict): the op tuning config.
        """
        tune_cfg = {'op': OrderedDict()}
        for op_name_type, op_config in op_tuning_cfg.items():
            if isinstance(op_config, OpTuningConfig):
                tune_cfg['op'][op_name_type] = op_config.get_state()
            else:
                tune_cfg[op_name_type] = op_config
        tune_cfg['calib_sampling_size'] = op_tuning_cfg['calib_sampling_size']
        if self.calib_dataloader is not None:
            tune_cfg['calib_iteration'] =  math.ceil(int(tune_cfg['calib_sampling_size']) / \
                                                    self.calib_dataloader.batch_size) 
        else:
            tune_cfg['calib_iteration'] = 1
        tune_cfg['advance'] = self.cfg.quantization.advance
        tune_cfg['approach'] = self.cfg.quantization.approach
        return tune_cfg

    def set_tuning_space(self, conf):
        """Create the tuning space.
        
        Create the tuning space based on the framework capability and user configuration.

        Args:
            conf: The Conf class instance includes all user configurations.
        """
        calib_sampling_size_lst = self.cfg.quantization.calibration.sampling_size
        calib_sampling_size_lst = [int(calib_sampling_size) for calib_sampling_size in calib_sampling_size_lst]
        if self.calib_dataloader:
            self.calib_iter = [math.ceil(int(x) / self.calib_dataloader.batch_size) \
                               for x in calib_sampling_size_lst]
        else:
            self.calib_iter = 1
        # create tuning space
        adaptor_cap = {
            'calib': {'calib_sampling_size': calib_sampling_size_lst},
            'op': self.capability['opwise']
        }
        self.tuning_space = TuningSpace(adaptor_cap, conf=conf, framework=self.framework)
        logger.debug(self.tuning_space.root_item.get_details())

    def setup_resume(self, resume):
        """Resume the best quantized model from tuning history.

        Args:
            resume: The dict containing resume information.
        """
        self.__dict__.update(resume)
        for history in self.tuning_history:
            if self._same_yaml(history['cfg'], self.cfg):
                self.__dict__.update({k: v for k, v in history.items() \
                                        if k not in ['version', 'history']})
                logger.info("Start to resume tuning process.")
                # resume the best tuning model if needed
                try:
                    index = history['id'] - 1
                    resume_tuning_cfg = history['history'][index]['tune_cfg']
                    self.best_qmodel = self.adaptor.quantize(resume_tuning_cfg,
                                                                self.model, 
                                                                self.calib_dataloader, 
                                                                self.q_func)
                except:
                    logger.debug("Can not resume the best quantize model from history.")
                    
                break

    def set_q_func(self):
        """Set the training function for quantization aware training."""
        if self.q_func == None and self.cfg.quantization.approach == 'quant_aware_training':
            train_cfg = self.cfg.quantization.train
            assert train_cfg, "train field of quantization section in yaml file must " \
                              "be configured for quantization aware training if q_func is NOT set."
            assert self.calib_dataloader, "dataloader field of train field of quantization " \
                                          "section in yaml file must be configured."
            self.q_func = create_train_func(self.framework, self.calib_dataloader, \
                                            self.adaptor, train_cfg, hooks=self.q_hooks)

    def _create_path(self, custom_path, filename):
        new_path = os.path.join(os.path.abspath(os.path.expanduser(custom_path)),filename)
        path = Path(os.path.dirname(new_path))
        path.mkdir(exist_ok=True, parents=True)
        return new_path

    def _set_framework_info(self, q_dataloader, q_func=None):
        framework_specific_info = {'device': self.cfg.device,
                                   'approach': self.cfg.quantization.approach,
                                   'random_seed': self.cfg.tuning.random_seed}
        framework = self.cfg.model.framework.lower()
        framework_specific_info.update({'backend': self.cfg.model.get('backend', 'default')})
        framework_specific_info.update({'format': self.cfg.model.get('quant_format', 'default')})

        self.mixed_precision_mode = bool('mixed_precision' in self.cfg) or \
            bool('graph_optimization' in self.cfg)

        if 'tensorflow' in framework:
            framework_specific_info.update(
                {"inputs": self.cfg.model.inputs,
                 "outputs": self.cfg.model.outputs,
                 'workspace_path': self.cfg.tuning.workspace.path,
                 'recipes': self.cfg.quantization.recipes,
                 'performance_only': self.cfg.tuning.exit_policy.performance_only,
                 'use_bf16': self.cfg.use_bf16 if self.cfg.use_bf16 is not None else False})
            if self.cfg.model.backend == 'itex':
                self.cfg.model.framework = 'tensorflow_itex'
                framework = 'tensorflow_itex'
        if 'keras' in framework:
            framework_specific_info.update({
                 'workspace_path': self.cfg.tuning.workspace.path, })
        if framework == 'mxnet':
            framework_specific_info.update({"q_dataloader": q_dataloader})
        if 'onnx' in framework.lower():
            if self.mixed_precision_mode:
                framework_specific_info.update({"approach": "post_training_dynamic_quant"})
            framework_specific_info.update({"deploy_path": os.path.dirname(self.deploy_path)})
            framework_specific_info.update({'workspace_path': self.cfg.tuning.workspace.path})
            framework_specific_info.update({'recipes': self.cfg.quantization.recipes})
            framework_specific_info.update(
                                {'graph_optimization': OPTIONS[framework].graph_optimization})
            framework_specific_info.update({'reduce_range': self.cfg.reduce_range})
            if framework.lower() == 'onnxrt_qdq' or \
                framework_specific_info['backend'] == 'onnxrt_trt_ep':
                framework_specific_info.update({'format': 'QDQ'})
                framework = 'onnxrt_qdq'
        if framework == 'pytorch_ipex' or framework == 'pytorch' or framework == 'pytorch_fx':
            if self.cfg.model.backend == 'ipex':
                self.cfg.model.framework = 'pytorch_ipex'
                framework = 'pytorch_ipex'
            elif self.cfg.model.backend == 'default':
                self.cfg.model.framework = 'pytorch_fx'
                framework = 'pytorch_fx'
            if self.mixed_precision_mode:
                framework_specific_info.update({"approach": "post_training_dynamic_quant"})
            framework_specific_info.update({"q_dataloader": q_dataloader})
            framework_specific_info.update({"use_bf16": self.cfg.use_bf16 \
                            if self.cfg.use_bf16 is not None else True})
            framework_specific_info.update(
                {"workspace_path": os.path.dirname(self.deploy_path)})
            if self.cfg['quantization']['op_wise'] is not None \
               and 'default_qconfig' in self.cfg['quantization']['op_wise']:
                framework_specific_info.update(
                    {"default_qconfig": self.cfg['quantization']['op_wise']['default_qconfig']})
            framework_specific_info.update({"q_func": q_func})
        return framework, framework_specific_info

    def _set_objectives(self):
        self.higher_is_better = bool(self.cfg.tuning.accuracy_criterion.higher_is_better)
        self.use_multi_objective = deep_get(self.cfg, 'tuning.multi_objectives') and \
            len(self.cfg.tuning.multi_objectives.objective) > 1
        objectives = [i.lower() for i in self.cfg.tuning.multi_objectives.objective] if \
            self.use_multi_objective else [self.cfg.tuning.objective.lower()]
        self.metric_weight = deep_get(self.cfg, 'evaluation.accuracy.multi_metrics.weight')
        self.metric_name = ['Accuracy'] if \
            not deep_get(self.cfg, 'evaluation.accuracy.multi_metrics') else \
            self.cfg.evaluation.accuracy.multi_metrics.keys()-{'weight','higher_is_better'}
        if len(self.metric_name) == 1:
            self.metric_criterion = [self.higher_is_better]
        elif not deep_get(self.cfg, 'evaluation.accuracy.multi_metrics.higher_is_better'):
            # default is True
            self.metric_criterion = [True] * len(self.metric_name)
        else:
            self.metric_criterion = \
                deep_get(self.cfg, 'evaluation.accuracy.multi_metrics.higher_is_better')

        self.objectives = MultiObjective(objectives, 
                             self.cfg.tuning.accuracy_criterion,
                             self.metric_criterion,
                             self.metric_weight,
                             deep_get(self.cfg, 'tuning.multi_objectives.higher_is_better'),
                             deep_get(self.cfg, 'tuning.multi_objectives.weight'))

    def _same_yaml(self, src_yaml, dst_yaml):
        """Check if the two yamls are the same.
        
        The check will exclude those keys which do not really impact the tuning result, such as 
        tensorboard, workspace, resume options under the tuning section of YAML.
        """
        if equal_dicts(src_yaml, dst_yaml, ignore_keys=['tuning']) and \
           equal_dicts(src_yaml.tuning, src_yaml.tuning, compare_keys=['objective',
                                                                       'accuracy_criterion',
                                                                       'random_seed',
                                                                       'exit_policy']):
            return True

        return False

    def update_best_op_tuning_cfg(self, op_tuning_cfg):
        """Track and update the best tuning config with correspondence accuracy result.

        Args:
            op_tuning_cfg: The tuning config.

        Returns:
            The current best tuning results and corresponding configurations.
        """
        acc, _ = self.last_tune_result
        if self.cur_best_tuning_cfg is None:
            self.cur_best_tuning_cfg = copy.deepcopy(op_tuning_cfg)
            self.cur_best_qmodel = self.last_qmodel
        if not isinstance(acc, list) and ((self.higher_is_better and acc >= self.cur_best_acc) \
            or (not self.higher_is_better and acc <= self.cur_best_acc)):
            self.cur_best_acc = acc
            self.cur_best_tuning_cfg = copy.deepcopy(op_tuning_cfg)
            self.cur_best_qmodel = self.last_qmodel
        elif len(self.metric_name) > 1 and self.metric_weight is not None:
            acc = np.mean(np.array(acc) * self.metric_weight)
            if (self.higher_is_better and acc >= self.cur_best_acc) or \
                (not self.higher_is_better and acc <= self.cur_best_acc):
                self.cur_best_acc = acc
                self.cur_best_tuning_cfg = copy.deepcopy(op_tuning_cfg)
                self.cur_best_qmodel = self.last_qmodel
        elif len(self.metric_name) > 1 and self.metric_weight is None:
            if all([acc_i >= best_i if higher_is_better else acc_i <= best_i for \
                acc_i, best_i, higher_is_better in \
                zip(acc, self.cur_best_acc, self.metric_criterion)]):
                self.cur_best_acc = acc
                self.cur_best_tuning_cfg = copy.deepcopy(op_tuning_cfg)            
                self.cur_best_qmodel = self.last_qmodel
        logger.debug(f"Best acc is {self.cur_best_acc}.")
        return self.cur_best_acc, self.cur_best_tuning_cfg

    def deploy_config(self):
        """Save the configuration locally for deployment."""
        acc_dataloader_cfg = deep_get(self.cfg, 'evaluation.accuracy.dataloader')
        perf_dataloader_cfg = deep_get(self.cfg, 'evaluation.performance.dataloader')
        # use acc dataloader if perf dataloader is not configured
        if perf_dataloader_cfg is None:
            perf_dataloader_cfg = acc_dataloader_cfg

        self.deploy_cfg = OrderedDict()
        # int8 dataloader graph transform
        if deep_get(perf_dataloader_cfg, 'transform.QuantizedInput') is not None \
          or deep_get(acc_dataloader_cfg, 'transform.QuantizedInput') is not None:
            self.best_qmodel, scale = self.adaptor.quantize_input(self.best_qmodel)
            deep_set(perf_dataloader_cfg, 'transform.QuantizedInput.dtype', 'int8')
            deep_set(perf_dataloader_cfg, 'transform.QuantizedInput.scale', scale)
            deep_set(acc_dataloader_cfg, 'transform.QuantizedInput.dtype', 'int8')
            deep_set(acc_dataloader_cfg, 'transform.QuantizedInput.scale', scale)

        self.deploy_cfg['model'] = self.cfg.model
        self.deploy_cfg['device'] = self.cfg.device
        if self.cfg.evaluation is not None:
            deep_set(self.cfg, 'evaluation.performance.dataloader',\
                perf_dataloader_cfg)
            deep_set(self.cfg, 'evaluation.accuracy.dataloader', \
                acc_dataloader_cfg)
            self.deploy_cfg['evaluation'] = self.cfg.evaluation

        def setup_yaml():
            represent_dict_order = lambda self, \
                data: self.represent_mapping('tag:yaml.org,2002:map', data.items())
            yaml.add_representer(OrderedDict, represent_dict_order)
            yaml.add_representer(DotDict, represent_dict_order)
        setup_yaml()
        with open(self.deploy_path, 'w+') as f:
            yaml.dump(self.deploy_cfg, f)
            logger.info("Save deploy yaml to {}".format(self.deploy_path))

    def _get_common_cfg(self, model_wise_cfg, op_wise_cfgs):
        """Get the common parts from the model_wise_cfg.
        
            This function is focused on composing the configuration that consists of
            model-wise field and op-wise unique field data.

        Args:
            model_wise_cfg ([DotDict]): The model-wise configuration.
            op_wise_cfgs ([List]): The list of each op's config in DotDict type.

        Returns:
            [DotDict]: The combined configration with the op-wise unique field.
        """
        model_wise_keys = model_wise_cfg.keys()

        result = op_wise_cfgs[0]
        for each_op_wise_cfg in op_wise_cfgs:
            tmp_cfg = {}
            for k in model_wise_keys:
                tmp_cfg[k] = each_op_wise_cfg[k]

            if model_wise_cfg == tmp_cfg:
                result = each_op_wise_cfg
                break

        return result

    @property
    def evaluation_result(self):
        """Evaluate the given model.

        Returns:
            The objective value evaluated.
        """
        return self._evaluate(self.model)

    def _evaluate(self, model):
        """Interface of evaluating model.

        Args:
            model (object): The model to be evaluated.

        Returns:
            Objective: The objective value evaluated.
        """
        if self.eval_func:
            if self.cfg.tuning.tensorboard:
                # Pytorch can insert observer to model in this hook.
                # Tensorflow don't support this mode for now
                model = self.adaptor._pre_eval_hook(model)
            val = self.objectives.evaluate(
                self.eval_func, model if self.framework == "pytorch_ipex" else model.model
            )
            if self.cfg.tuning.tensorboard:
                # post_eval_hook to deal the tensor
                self.adaptor._post_eval_hook(model, accuracy=val[0])
        else:
            assert self.cfg.evaluation and self.cfg.evaluation.accuracy and \
                (self.cfg.evaluation.accuracy.metric or \
                self.cfg.evaluation.accuracy.multi_metrics), \
                "metric or multi_metrics field of accuracy field of evaluation" \
                " section should not be empty"

            postprocess_cfg = self.cfg.evaluation.accuracy.postprocess
            metric_cfg = self.cfg.evaluation.accuracy.metric if \
                self.cfg.evaluation.accuracy.metric else \
                self.cfg.evaluation.accuracy.multi_metrics
            iteration = -1 if self.cfg.evaluation.accuracy.iteration is None \
                else self.cfg.evaluation.accuracy.iteration
            eval_func = create_eval_func(self.framework,
                self.eval_dataloader,
                self.adaptor,
                metric_cfg,
                postprocess_cfg,
                iteration,
                tensorboard = self.cfg.tuning.tensorboard,
                fp32_baseline = self.baseline == None)

            if getattr(self.eval_dataloader, 'distributed', False):
                if 'tensorflow' in self.framework:
                    import horovod.tensorflow as hvd
                elif self.framework in ['pytorch_ipex','pytorch','pytorch_fx']:
                    import horovod.torch as hvd
                else:
                    raise NotImplementedError("Currently only TensorFlow and PyTorch "
                                              "support distributed inference in PTQ.")
                hvd.init()
                try:
                    len_dataloader = len(self.eval_dataloader)
                except:
                    logger.info("The length of the distributed dataloader is unknown."
                                "When the iteration of evaluation dataloader in each "
                                "process is inconsistent, an error may occur.")
                else:
                    list_len_dataloader = hvd.allgather_object(len_dataloader)
                    if hvd.rank() == 0:
                        for i in range(len(list_len_dataloader)-1):
                            if list_len_dataloader[i] != list_len_dataloader[i+1]:
                                raise AttributeError("The evaluation dataloader's iteration is"
                                                     "different between processes, please reset "
                                                     "dataloader's batch_size.")
            val = self.objectives.evaluate(eval_func, model) 
        if isinstance(val[0], list):
            assert all([np.isscalar(i) for i in val[0]]), \
                "The eval_func should return a scalar or list of scalar, " \
                "but not {}!".format(str([type(i) for i in val[0]]))
        else:
            assert np.isscalar(val[0]), \
                "The eval_func should return a scalar or list of scalar, " \
                "but not {}!".format(str(type(val[0])))
            
        return val

    def __getstate__(self):
        """Magic method for pickle saving.

        Returns:
            dict: Saved dict for resuming
        """
        return {'tuning_history': self.tuning_history}

    def __setstate__(self, d):
        """Magic method for pickle loading.

        Args:
            d (dict): The dict to load.
        """
        self.__dict__.update(d)

    def stop(self, timeout, trials_count):
        """Check if need to stop traverse.
        
        Check if need to stop traversing the tuning space, either accuracy goal is met or timeout is reach.

        Returns:
            bool: True if need stop, otherwise False
        """
        need_stop = False
        if self.cfg.tuning.exit_policy.performance_only or \
            self.objectives.compare(self.best_tune_result, self.baseline):
            del self.best_tune_result
            del self.best_qmodel
            self.best_tune_result = self.last_tune_result
            self.best_qmodel = self.last_qmodel
            logger.debug(f"*** Update the best qmodel with the result {self.best_tune_result}")
            if self.metric_met_point == 0:
                self.metric_met_point = self.tuning_times
        
        # track the model with highest acc
        if self.best_tune_result and self.last_tune_result: # (acc, [perf])
            if self.re_quant and self.objectives.accuracy_meets():
                self.best_tune_result = self.last_tune_result
                self.best_qmodel = self.last_qmodel
                logger.debug(f"*** Update the best qmodel with the result {self.best_tune_result}.")
            else:
                logger.debug(f"*** Accuracy not meets the requirements, do not update the best qmodel.")

        if self.last_tune_result:
            last_tune = self.last_tune_result[0] if \
                isinstance(self.last_tune_result[0], list) else [self.last_tune_result[0]]

            for name, data in zip(self.metric_name, last_tune):
                if len(self.tune_data[name]) == 1:
                    self.tune_data[name].append(data)
                else:
                    self.tune_data[name][1] = data

            if self.metric_weight and len(last_tune) > 1:
                weighted_acc = np.mean(np.array(last_tune) * self.metric_weight)
                    
                if len(self.tune_data['Weighted accuracy']) == 1:
                    self.tune_data['Weighted accuracy'].append(weighted_acc)
                else:
                    self.tune_data['Weighted accuracy'][1] = weighted_acc

                last_tune = [weighted_acc]

            last_tune_msg = '[Accuracy (int8|fp32):' + \
                ''.join([' {:.4f}|{:.4f}'.format(last, base) for last, base in \
                zip(last_tune, self.tune_data['baseline'])]) + \
                ''.join([', {} (int8|fp32): {:.4f}|{:.4f}'.format( \
                x, y, z) for x, y, z in zip( \
                self.objectives.representation, self.last_tune_result[1], self.baseline[1]) \
                if x != 'Accuracy']) + ']'
        else: # pragma: no cover
            last_tune_msg = 'n/a'
            for name in self.tune_data.keys() - {'baseline'}:
                if len(self.tune_data[name]) == 1:
                    self.tune_data[name].append('n/a')
                else:
                    self.tune_data[name][1] = 'n/a'

        if self.best_tune_result:
            best_tune = self.best_tune_result[0] if isinstance(self.best_tune_result[0], list) \
                        else [self.best_tune_result[0]]
 
            for name, data in zip(self.metric_name, best_tune):
                if len(self.tune_data[name]) == 2:
                     self.tune_data[name].append(data)
                else:
                    self.tune_data[name][2] = data

            if self.metric_weight and len(best_tune) > 1:
                weighted_acc = np.mean(np.array(best_tune) * self.metric_weight)

                if len(self.tune_data['Weighted accuracy']) == 2:
                    self.tune_data['Weighted accuracy'].append(weighted_acc)
                else: # pragma: no cover
                    self.tune_data['Weighted accuracy'][2] = weighted_acc

                best_tune = [weighted_acc]

            best_tune_msg = '[Accuracy:' + ''.join([' {:.4f}'.format(best) \
                for best in best_tune]) + ''.join([', {}: {:.4f}'.format(x,y) \
                for x,y in zip(self.objectives.representation, \
                self.best_tune_result[1]) if x != 'Accuracy']) + ']'

        else:
            best_tune_msg = 'n/a'
            for name in self.tune_data.keys() - {'baseline'}:
                if len(self.tune_data[name]) == 2:
                    self.tune_data[name].append('n/a')
                else:
                    self.tune_data[name][2] = 'n/a'

        logger.info("Tune {} result is: {}, Best tune result is: {}".format(trials_count,
                                                                            last_tune_msg,
                                                                            best_tune_msg))
        output_data = [[info_type, 
            '{:.4f} '.format(self.tune_data[info_type][0]) if \
            not isinstance(self.tune_data[info_type][0], str) else self.tune_data[info_type][0], 
            '{:.4f} '.format(self.tune_data[info_type][1]) if \
            not isinstance(self.tune_data[info_type][1], str) else self.tune_data[info_type][1],
            '{:.4f} '.format(self.tune_data[info_type][2]) if \
            not isinstance(self.tune_data[info_type][2], str) else self.tune_data[info_type][2]] \
            for info_type in self.tune_data.keys() if info_type != 'baseline']

        output_data.extend([[obj, 
            '{:.4f} '.format(self.baseline[1][i]) if self.baseline else 'n/a',
            '{:.4f} '.format(self.last_tune_result[1][i]) if self.last_tune_result else 'n/a',
            '{:.4f} '.format(self.best_tune_result[1][i]) if self.best_tune_result else 'n/a'] \
            for i, obj in enumerate(self.objectives.representation)])
        self.tuning_result_data = output_data
        Statistics(output_data,
                   header='Tune Result Statistics',
                   field_names=['Info Type', 'Baseline', 'Tune {} result'.format(trials_count), \
                                                                'Best tune result']).print_stat()


        if self.cfg.tuning.exit_policy.performance_only:
            need_stop = True
        elif timeout == 0 and self.best_tune_result:
            need_stop = True
        elif trials_count >= self.cfg.tuning.exit_policy.max_trials:
            need_stop = True
        else:
            need_stop = False

        return need_stop

    def _save(self):
        """Save current tuning state to snapshot for resuming."""
        logger.info("Save tuning history to {}.".format(self.history_path))
        with fault_tolerant_file(self.history_path) as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _find_tuning_history(self, tune_cfg):
        """Check if the specified tune_cfg is evaluated or not on same yaml config.

        Args:
            tune_cfg (dict): The tune_cfg to check if evaluated before.

        Returns:
            tuning_history or None: The tuning history containing evaluated tune_cfg.
        """
        for tuning_history in self.tuning_history:
            # only check if a tune_cfg is evaluated under same yam config, excluding
            # some fields in tuning section of yaml, such as tensorboard, snapshot, resume.
            if self._same_yaml(tuning_history['cfg'], self.cfg):
                for history in tuning_history['history']:
                    if history and history['tune_cfg'] == tune_cfg:
                        return tuning_history

        return None

    def _find_history(self, tune_cfg):
        """Check if the specified tune_cfg is evaluated or not on same yaml config.

        Returns:
            history or None: The history containing evaluated tune_cfg.
        """
        for tuning_history in self.tuning_history:
            # only check if a tune_cfg is evaluated under same yam config, excluding
            # some fields in tuning section of yaml, such as tensorboard, snapshot, resume.
            if self._same_yaml(tuning_history['cfg'], self.cfg):
                for history in tuning_history['history']:
                    if history and history['tune_cfg'] == tune_cfg:
                        return history
        return None

    def _find_self_tuning_history(self):
        """Find self history dict.

        Returns:
            history or None: The history for self.
        """
        for tuning_history in self.tuning_history:
            # only check if a tune_cfg is evaluated under same yam config, excluding
            # some fields in tuning section of yaml, such as tensorboard, snapshot, resume.
            if self._same_yaml(tuning_history['cfg'], self.cfg):
                return tuning_history

        return None

    def _add_tuning_history(self, tune_cfg=None, tune_result=None, **kwargs):
        """Add tuning config to tuining history.

        Note this record is added under same yaml config.
        """
        found = False
        d = {'tune_cfg': tune_cfg, 'tune_result': tune_result}
        for tuning_history in self.tuning_history:
            if self._same_yaml(tuning_history['cfg'], self.cfg):
                d.update(kwargs)
                tuning_history['history'].append(d)
                tuning_history['last_tune_result'] = self.last_tune_result
                tuning_history['best_tune_result'] = self.best_tune_result
                tuning_history['cfg'] = self.cfg
                found = True
                break

        if not found:
            tuning_history = {}
            tuning_history['version']  = __version__
            tuning_history['cfg']     = self.cfg
            tuning_history['baseline'] = self.baseline
            tuning_history['last_tune_result'] = self.last_tune_result
            tuning_history['best_tune_result'] = self.best_tune_result
            tuning_history['history']  = []
            if tune_cfg and tune_result:
                d.update(kwargs)
                tuning_history['history'].append(d)
            self.tuning_history.append(tuning_history)

        self._save()

    def _fake_eval_func(self, model):
        return 1.

    def _collect_ops_by_quant_mode(self, tune_cfg, quant_mode):
        ops_lst = []
        for op_info, op_config in tune_cfg.items():
            if isinstance(op_config, OpTuningConfig) and quant_mode in op_config.op_quant_mode:
                ops_lst.append(op_info)
        return ops_lst

    def _diagnosis(self):
        import logging
        logger = logging.getLogger("neural_compressor")
        iteration_list = self.cfg.tuning.diagnosis.iteration_list
        inspect_type = self.cfg.tuning.diagnosis.inspect_type
        save_to_disk = self.cfg.tuning.diagnosis.save_to_disk
        save_path = self.cfg.tuning.diagnosis.save_path
        inspect_node_lst, updated_cfg = self.adaptor.diagnosis_helper(self._fp32_model, 
                                                                      self.last_qmodel, 
                                                                      self.tune_cfg, 
                                                                      save_path = save_path)
        op_list = self.cfg.tuning.diagnosis.op_list
        if not op_list:
            op_list = list(inspect_node_lst)
        else:
            op_list = list(set(op_list).intersection(inspect_node_lst))

        logger.debug(f'*** Start to inspect tensor :{op_list} in  fp32 model.')
        self.adaptor.inspect_tensor(self._fp32_model,
                                    dataloader=self.calib_dataloader,
                                    op_list=op_list,
                                    iteration_list=iteration_list,
                                    inspect_type=inspect_type, 
                                    save_to_disk=save_to_disk,
                                    save_path= save_path + '/fp32/',
                                    quantization_cfg=updated_cfg)

        logger.debug(f'*** Start to inspect tensor :{op_list} in  quantized model.')
        self.adaptor.inspect_tensor(self.last_qmodel, 
                                    dataloader=self.calib_dataloader,
                                    op_list=op_list,
                                    iteration_list=iteration_list,
                                    inspect_type=inspect_type, 
                                    save_to_disk=save_to_disk,
                                    save_path= save_path + '/quan/',
                                    quantization_cfg=updated_cfg)
    
