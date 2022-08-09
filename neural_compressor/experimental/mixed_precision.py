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

import os
import pickle
import random
import tempfile
import sys
import numpy as np
import yaml
from ..conf.config import MixedPrecision_Conf
from ..conf.dotdict import deep_get, deep_set, DotDict
from ..strategy import STRATEGIES
from ..utils import logger
from ..utils.create_obj_from_config import create_dataloader
from ..utils.utility import CpuInfo, time_limit, set_backend
from .common import Model as NCModel
from ..model import BaseModel
from .graph_optimization import GraphOptimization

class MixedPrecision(GraphOptimization):
    """MixedPrecision class automatically generates low precision model across various DL
       frameworks including tensorflow, pytorch and onnxruntime.

    Args:
        conf_fname_or_obj (string or obj): The path to the YAML configuration file or
            Graph_Optimization_Conf class containing accuracy goal, tuning objective and
            preferred calibration & quantization tuning space etc.

    """

    def __init__(self, conf_fname_or_obj=None):
        self.conf_name = conf_fname_or_obj
        self._model = None
        self._eval_dataloader = None
        self._eval_func = None
        self._precisions = 'fp32'
        self._input = None
        self._output = None
        self.conf = None

        if isinstance(conf_fname_or_obj, MixedPrecision_Conf):
            self.conf = conf_fname_or_obj
        else:
            self.conf = MixedPrecision_Conf(conf_fname_or_obj)
        cfg = self.conf.usr_cfg
        if cfg.model.framework != 'NA':
            self.framework = cfg.model.framework.lower()
            set_backend(self.framework)

        cfg.tuning.strategy.name = 'automixedprecision'
        seed = cfg.tuning.random_seed
        random.seed(seed)
        np.random.seed(seed)

    def __call__(self):
        """The main entry point of mixed precision process.

           This interface works on all the DL frameworks that neural_compressor supports
           and provides three usages:
           a) Fully yaml configuration: User specifies all the info through yaml,
              including dataloaders used in calibration and evaluation phases
              and quantization tuning settings.

              For this usage, only model parameter is mandatory.

           b) Partial yaml configuration: User specifies dataloaders used in evaluation phase by code.
              The tool provides built-in dataloaders and evaluators, user just need provide
              a dataset implemented __iter__ or __getitem__ methods and invoke dataloader()
              with dataset as input parameter to create neural_compressor dataloader before calling this
              function.

              After that, User specifies fp32 "model" and evaluation dataset "eval_dataloader".
              The converted model is evaluated with "eval_dataloader" with evaluation metrics
              specified in the configuration file. The evaluation tells the tuner whether the
              converted model meets the accuracy criteria. If not, the tuner starts a tuning flow.

              For this usage, model and eval_dataloader parameters are mandatory.

        Returns:
            converted model: best converted model found, otherwise return None

        """

        assert isinstance(self._model, BaseModel), 'need set your Model for mixed precision....'
        if 'onnx' in self.framework and 'bf16' in self._precisions:
            logger.warning("Mixed precision doesn't support bf16 for ONNX models.")
            sys.exit(0)
        elif 'onnx' not in self.framework and 'fp16' in self._precisions:
            logger.warning("Mixed precision doesn't support fp16 for {} models.".format(
                self.framework))
            sys.exit(0)

        if self._precisions == ['bf16'] and not CpuInfo().bf16: # pragma: no cover
            if os.getenv('FORCE_BF16') == '1':
                logger.warning("Mixed precision will generate bf16 graph although " \
                               "the hardware doesn't support bf16 instruction.")
            else:
                logger.warning("Mixed precision exits due to the hardware " \
                               "doesn't support bf16 instruction.")
                sys.exit(0)

        if self._precisions == ['fp16'] and self.conf.usr_cfg.device != 'gpu':
            if os.getenv('FORCE_FP16') == '1':
                logger.warning("Mixed precision will generate fp16 graph although " \
                               "the hardware doesn't support fp16 instruction.")
            else:
                logger.warning("Mixed precision exits due to the hardware " \
                               "doesn't support fp16 instruction.")
                sys.exit(0)

        cfg = self.conf.usr_cfg
        if self.framework == 'tensorflow':
            self._model.name = cfg.model.name
            self._model.output_tensor_names = cfg.model.outputs if \
                not self._output else self._output
            self._model.input_tensor_names = cfg.model.inputs if \
                not self._input else self._input
            self._model.workspace_path = cfg.tuning.workspace.path

        # when eval_func is set, will be directly used and eval_dataloader can be None
        if self._eval_func is None:
            if self._eval_dataloader is None:
                eval_dataloader_cfg = deep_get(cfg, 'evaluation.accuracy.dataloader')
                if eval_dataloader_cfg is None:
                    self._eval_func = None
                else:
                    self._eval_dataloader = create_dataloader(self.framework, eval_dataloader_cfg)

        strategy = cfg.tuning.strategy.name.lower()

        assert strategy in STRATEGIES, "Tuning strategy {} is NOT supported".format(strategy)

        _resume = None
        # check if interrupted tuning procedure exists. if yes, it will resume the
        # whole auto tune process.
        self.resume_file = os.path.abspath(os.path.expanduser(cfg.tuning.workspace.resume)) \
                           if cfg.tuning.workspace and cfg.tuning.workspace.resume else None
        if self.resume_file: # pragma: no cover
            assert os.path.exists(self.resume_file), \
                "The specified resume file {} doesn't exist!".format(self.resume_file)
            with open(self.resume_file, 'rb') as f:
                _resume = pickle.load(f).__dict__

        self.strategy = STRATEGIES[strategy](
            self._model,
            self.conf,
            None,
            None,
            self._eval_dataloader,
            self._eval_func,
            _resume)

        try:
            with time_limit(self.conf.usr_cfg.tuning.exit_policy.timeout):
                self.strategy.traverse()
        except KeyboardInterrupt: # pragma: no cover
            pass
        except Exception as e: # pragma: no cover
            logger.info("Unexpected exception {} happened during turing.".format(repr(e)))
        finally: 
            if self.strategy.best_qmodel:
                logger.info(
                    "Specified timeout or max trials is reached! "
                    "Found a converted model which meet accuracy goal. Exit.")
                self.strategy.deploy_config()
            else: # pragma: no cover
                logger.info(
                    "Specified timeout or max trials is reached! "
                    "Not found any converted model which meet accuracy goal. Exit.")

            logger.info("Mixed Precision is done. Please invoke model.save() to save " \
                        "optimized model to disk.")

            return self.strategy.best_qmodel

    fit = __call__

    @property
    def precisions(self):
        return self._precisions

    @precisions.setter
    def precisions(self, customized_precisions):
        if isinstance(customized_precisions, list):
            self._precisions = sorted([i.strip() for i in customized_precisions])
        elif isinstance(customized_precisions, str):
            self._precisions = sorted([i.strip() for i in customized_precisions.split(',')])
        self.conf.usr_cfg.mixed_precision.precisions = self._precisions

    def set_config_by_model(self, model_obj):
        self.conf.usr_cfg.model.framework = model_obj.framework()
        if self._input:
            self.conf.usr_cfg.model.inputs = self._input
        if self._output:
            if isinstance(self._output, str) and ',' in self._output:
                self.conf.usr_cfg.model.outputs = [s.strip() for s in self._output.split(',')]
            else:
                self.conf.usr_cfg.model.outputs = self._output

    def __repr__(self):
        return 'MixedPrecision'
