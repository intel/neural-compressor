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

from abc import abstractmethod
import os
import math
import copy
import pickle
from collections import OrderedDict
from pathlib import Path
import yaml
import numpy as np
from ..adaptor import FRAMEWORKS
from ..objective import OBJECTIVES
from ..utils.utility import fault_tolerant_file, equal_dicts
from ..utils.create_obj_from_config import create_eval_func, create_train_func
from ..utils import logger
from ..version import __version__
from ..conf.dotdict import DotDict, deep_get, deep_set
from ..algorithm import AlgorithmScheduler

"""The tuning strategies supported by neural_compressor, including basic, random, bayesian and mse.

   User could add new strategies by implementing new TuneStrategy subclass under this directory.
   The naming convention of new strategy subclass should be something like ABCTuneStrategy, user
   could choose this strategy by setting "abc" string in tuning.strategy field of yaml.

   STRATEGIES variable is used to store all implemented TuneStrategy subclasses to support
   different tuning strategies.
"""
STRATEGIES = {}


def strategy_registry(cls):
    """The class decorator used to register all TuneStrategy subclasses.

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


class TuneStrategy(object):
    """The base class of tuning strategy.

    Args:
        model (object):                        The FP32 model specified for low precision tuning.
        conf (Conf):                           The Conf class instance initialized from user yaml
                                               config file.
        q_dataloader (generator):              Data loader for calibration, mandatory for
                                               post-training quantization.
                                               It is iterable and should yield a tuple (input,
                                               label) for calibration dataset containing label,
                                               or yield (input, _) for label-free calibration
                                               dataset. The input could be a object, list, tuple or
                                               dict, depending on user implementation, as well as
                                               it can be taken as model input.
        q_func (function, optional):           Reserved for future use.
        eval_dataloader (generator, optional): Data loader for evaluation. It is iterable
                                               and should yield a tuple of (input, label).
                                               The input could be a object, list, tuple or dict,
                                               depending on user implementation, as well as it can
                                               be taken as model input. The label should be able
                                               to take as input of supported metrics. If this
                                               parameter is not None, user needs to specify
                                               pre-defined evaluation metrics through configuration
                                               file and should set "eval_func" parameter as None.
                                               Tuner will combine model, eval_dataloader and
                                               pre-defined metrics to run evaluation process.
        eval_func (function, optional):        The evaluation function provided by user.
                                               This function takes model as parameter, and
                                               evaluation dataset and metrics should be
                                               encapsulated in this function implementation and
                                               outputs a higher-is-better accuracy scalar value.

                                               The pseudo code should be something like:

                                               def eval_func(model):
                                                    input, label = dataloader()
                                                    output = model(input)
                                                    accuracy = metric(output, label)
                                                    return accuracy
        resume(dict, optional):                The dict containing resume information.
                                               Defaults to None.
    """

    def __init__(self, model, conf, q_dataloader=None, q_func=None,
                 eval_dataloader=None, eval_func=None, resume=None, q_hooks=None):
        self.model = model
        self.cfg = conf.usr_cfg

        self.history_path = os.path.join(os.path.abspath(os.path.expanduser( \
                                                         self.cfg.tuning.workspace.path)),
                                                         './history.snapshot')
        self.deploy_path  = os.path.join(os.path.abspath(os.path.expanduser( \
                                                         self.cfg.tuning.workspace.path)),
                                                         'deploy.yaml')

        path = Path(os.path.dirname(self.history_path))
        path.mkdir(exist_ok=True, parents=True)
        path = Path(os.path.dirname(self.deploy_path))
        path.mkdir(exist_ok=True, parents=True)

        logger.debug("Dump user yaml configuration:")
        logger.debug(self.cfg)

        self.eval_dataloader = eval_dataloader
        self.calib_dataloader = q_dataloader
        self.q_func = q_func
        self.q_hooks = q_hooks
        self.eval_func = eval_func

        framework_specific_info = {'device': self.cfg.device,
                                   'approach': self.cfg.quantization.approach,
                                   'random_seed': self.cfg.tuning.random_seed}
        framework = self.cfg.model.framework.lower()
        if framework == 'tensorflow' or framework == 'tensorflow_itex':
            framework_specific_info.update(
                {"inputs": self.cfg.model.inputs,
                 "outputs": self.cfg.model.outputs,
                 'workspace_path': self.cfg.tuning.workspace.path,
                 'recipes': self.cfg.quantization.recipes})
        if framework == 'mxnet':
            framework_specific_info.update({"q_dataloader": q_dataloader})
        if 'onnxrt' in framework.lower():
            framework_specific_info.update({"backend": framework.lower().split('_')[-1]})
            framework_specific_info.update({"deploy_path": os.path.dirname(self.deploy_path)})
            framework_specific_info.update({'workspace_path': self.cfg.tuning.workspace.path})
        if framework == 'pytorch_ipex' or framework == 'pytorch' or framework == 'pytorch_fx':
            framework_specific_info.update({"q_dataloader": q_dataloader})
            framework_specific_info.update(
                {"workspace_path": os.path.dirname(self.deploy_path)})
            if self.cfg['quantization']['op_wise'] is not None \
               and 'default_qconfig' in self.cfg['quantization']['op_wise']:
                framework_specific_info.update(
                    {"default_qconfig": self.cfg['quantization']['op_wise']['default_qconfig']})
        if framework == 'engine':
            framework_specific_info.update(
                 {'workspace_path': self.cfg.tuning.workspace.path})
 
        self.adaptor = FRAMEWORKS[framework](framework_specific_info)
        self.framework = framework

        if self.q_func == None and self.cfg.quantization.approach == 'quant_aware_training':
            train_cfg = self.cfg.quantization.train
            assert train_cfg, "train field of quantization section in yaml file must " \
                              "be configured for quantization aware training if q_func is NOT set."
            assert self.calib_dataloader, "dataloader field of train field of quantization " \
                                          "section in yaml file must be configured."
            self.q_func = create_train_func(self.framework, self.calib_dataloader, \
                                            self.adaptor, train_cfg, hooks=self.q_hooks)

        self.baseline = None
        self.last_tune_result = None
        self.last_qmodel = None
        self.best_tune_result = None
        self.best_qmodel = None

        objective = self.cfg.tuning.objective.lower()
        self.objective = OBJECTIVES[objective](self.cfg.tuning.accuracy_criterion)

        self.capability = self.adaptor.query_fw_capability(model)
        self.graph_optimization_mode = bool('graph_optimization' in self.cfg)

        self.modelwise_tune_space = conf.modelwise_tune_space(self.capability['optypewise'])
        self.opwise_tune_space = conf.opwise_tune_space(self.capability['opwise'])
        self.model_wise_tune_cfgs = OrderedDict()
        for optype, optype_cfgs in self.modelwise_tune_space.items():
            self.model_wise_tune_cfgs[optype] = conf.expand_tune_cfgs(optype_cfgs)
        self.opwise_tune_cfgs = OrderedDict()
        for key in self.opwise_tune_space:
            expanded_cfg = conf.expand_tune_cfgs(self.opwise_tune_space[key])
            if expanded_cfg:
                self.opwise_tune_cfgs[key] = expanded_cfg

        if self.calib_dataloader:
            self.calib_iter = [math.ceil(int(x) / self.calib_dataloader.batch_size) \
                               for x in self.cfg.quantization.calibration.sampling_size]
        else:
            self.calib_iter = [1]

        fallback_precision_list = ['fp32'] if self.graph_optimization_mode else ['fp32', 'bf16']
        self.model_wise_quant_cfgs = OrderedDict()
        for optype in self.model_wise_tune_cfgs.keys():
            self.model_wise_quant_cfgs[optype] = []
            for cfg in self.model_wise_tune_cfgs[optype]:
                if cfg['activation']['dtype'] not in fallback_precision_list:
                    self.model_wise_quant_cfgs[optype].append(cfg)
        self.combined_model_wise_quant_cfgs = conf._combine_optype_quant_cfgs(
                                         self.model_wise_quant_cfgs)
        if len(self.combined_model_wise_quant_cfgs) == 0:
            logger.warning("No valid model wise quantization config found.")

        self.opwise_quant_cfgs = OrderedDict()
        for key in self.opwise_tune_cfgs:
            cfg_list = self.opwise_tune_cfgs[key]
            new_list = []
            for cfg in cfg_list:
                if self.graph_optimization_mode:
                    if cfg['activation']['dtype'] in self.cfg.graph_optimization.precisions:
                        new_list.append(cfg)
                else:
                    if cfg['activation']['dtype'] not in fallback_precision_list:
                        new_list.append(cfg)
            self.opwise_quant_cfgs[key] = new_list

        self.algo = AlgorithmScheduler(self.cfg.quantization.recipes)
        self.algo.dataloader = self.calib_dataloader
        # reuse the calibration iteration
        self.algo.origin_model = self.model
        self.algo.adaptor = self.adaptor
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
        self.tuning_history = []

        if resume is not None:
            self.__dict__.update(resume)
            for history in self.tuning_history:
                if self._same_yaml(history['cfg'], self.cfg):
                    self.__dict__.update({k: v for k, v in history.items() \
                                          if k not in ['version', 'history']})
                    logger.info("Start to resume tuning process.")
                    break

    def _same_yaml(self, src_yaml, dst_yaml):
        """Check whether two yamls are same, excluding those keys which does not really
           impact tuning result, such as tensorboard, workspace, resume options under tuning
           section of yaml.
        """
        if equal_dicts(src_yaml, dst_yaml, ignore_keys=['tuning']) and \
           equal_dicts(src_yaml.tuning, src_yaml.tuning, compare_keys=['objective',
                                                                       'accuracy_criterion',
                                                                       'random_seed',
                                                                       'exit_policy']):
            return True

        return False

    @abstractmethod
    def next_tune_cfg(self):
        """The generator of yielding next tuning config to traverse by concrete strategies
           according to last tuning result.

        Yields:
            tune_config (dict): It's a dict containing the tuning configuration to run.
        """
        raise NotImplementedError

    def traverse(self):
        """The main traverse logic, which could be override by some concrete strategy which needs
           more hooks.
        """
        if not (self.cfg.evaluation and self.cfg.evaluation.accuracy and \
            self.cfg.evaluation.accuracy.metric) and self.eval_func is None:
            logger.info("Neither evaluation function nor metric is defined." \
                        " Generate a quantized model with default quantization configuration.")
            self.cfg.tuning.exit_policy.performance_only = True
            logger.info("Generate a fake evaluation function.")
            self.eval_func = self._fake_eval_func

        # get fp32 model baseline
        if self.baseline is None:
            logger.info("Get FP32 model baseline.")
            self.baseline = self._evaluate(self.model)
            # record the FP32 baseline
            self._add_tuning_history()
        baseline_msg = '[accuracy: {:.4f}, {}: {:.4f}]'.format(self.baseline[0],
                                                                str(self.objective.measurer),
                                                                self.baseline[1]) \
                                                                if self.baseline else 'n/a'
        logger.info("FP32 baseline is: {}".format(baseline_msg))

        trials_count = 0
        for tune_cfg in self.next_tune_cfg():
            # add tune_cfg here as quantize use tune_cfg
            tune_cfg['advance'] = self.cfg.quantization.advance
            trials_count += 1
            tuning_history = self._find_tuning_history(tune_cfg)
            if tuning_history and trials_count < self.cfg.tuning.exit_policy.max_trials:
                self.last_tune_result = tuning_history['last_tune_result']
                self.best_tune_result = tuning_history['best_tune_result']
                logger.warn("Find evaluated tuning config, skip.")
                continue

            logger.debug("Dump current tuning configuration:")
            logger.debug(tune_cfg)
            self.q_model = self.adaptor.quantize(
                tune_cfg, self.model, self.calib_dataloader, self.q_func)
            self.algo.calib_iter = tune_cfg['calib_iteration']
            self.algo.q_model = self.q_model
            # TODO align the api to let strategy has access to pre_optimized model
            assert self.adaptor.pre_optimized_model
            self.algo.origin_model = self.adaptor.pre_optimized_model
            self.last_qmodel = self.algo()
            assert self.last_qmodel
            self.last_tune_result = self._evaluate(self.last_qmodel)
            need_stop = self.stop(self.cfg.tuning.exit_policy.timeout, trials_count)

            # record the tuning history
            saved_tune_cfg = copy.deepcopy(tune_cfg)
            saved_last_tune_result = copy.deepcopy(self.last_tune_result)
            self._add_tuning_history(saved_tune_cfg,
                                    saved_last_tune_result,
                                    q_config=self.q_model.q_config)
            if need_stop:
                break

    def deploy_config(self):
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

    def _evaluate(self, model):
        """The interface of evaluating model.

        Args:
            model (object): The model to be evaluated.

        Returns:
            Objective: The objective value evaluated
        """
        if self.eval_func:
            if self.cfg.tuning.tensorboard:
                # Pytorch can insert observer to model in this hook.
                # Tensorflow don't support this mode for now
                model = self.adaptor._pre_eval_hook(model)
            val = self.objective.evaluate(self.eval_func, model.model)
            if self.cfg.tuning.tensorboard:
                # post_eval_hook to deal the tensor
                self.adaptor._post_eval_hook(model, accuracy=val[0])
        else:
            assert self.cfg.evaluation and self.cfg.evaluation.accuracy and \
                self.cfg.evaluation.accuracy.metric, \
                'metric field of accuracy field of evaluation section should not be empty'

            postprocess_cfg = self.cfg.evaluation.accuracy.postprocess
            eval_func = create_eval_func(self.framework, \
                                         self.eval_dataloader, \
                                         self.adaptor, \
                                         self.cfg.evaluation.accuracy.metric, \
                                         postprocess_cfg, \
                                         self.cfg.evaluation.accuracy.iteration, \
                                         tensorboard = self.cfg.tuning.tensorboard, \
                                         fp32_baseline = self.baseline == None)

            val = self.objective.evaluate(eval_func, model)
        assert np.isscalar(val[0]), \
            "The eval_func should return a scalar, but not {}!".format(str(type(val[0])))
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
        """Check if need to stop traversing the tuning space, either accuracy goal is met
           or timeout is reach.

        Returns:
            bool: True if need stop, otherwise False
        """
        need_stop = False

        if self.cfg.tuning.exit_policy.performance_only or \
            self.objective.compare(self.best_tune_result, self.baseline):
            del self.best_tune_result
            del self.best_qmodel
            self.best_tune_result = self.last_tune_result
            self.best_qmodel = self.last_qmodel
        else:
            del self.last_qmodel

        last_tune_msg = '[accuracy: {:.4f}, {}: {:.4f}]'.format(self.last_tune_result[0],
                                                                str(self.objective.measurer),
                                                                self.last_tune_result[1]) \
                                                                if self.last_tune_result else 'n/a'
        best_tune_msg = '[accuracy: {:.4f}, {}: {:.4f}]'.format(self.best_tune_result[0],
                                                                str(self.objective.measurer),
                                                                self.best_tune_result[1]) \
                                                                if self.best_tune_result else 'n/a'
        logger.info("Tune {} result is: {}, Best tune result is: {}".format(trials_count,
                                                                            last_tune_msg,
                                                                            best_tune_msg))

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
        """save current tuning state to snapshot for resuming.

        """

        logger.info("Save tuning history to {}.".format(self.history_path))
        with fault_tolerant_file(self.history_path) as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _find_tuning_history(self, tune_cfg):
        """check if the specified tune_cfg is evaluated or not on same yaml config.

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
        """check if the specified tune_cfg is evaluated or not on same yaml config.

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
        """find self history dict.

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
        """add tuning history.
           note this record is added under same yaml config.

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
