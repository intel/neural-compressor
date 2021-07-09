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

from ..conf.config import Conf
from ..utils import logger
from .common import Model as LpotModel
from ..model import BaseModel
from .common import Metric, Postprocess

from .quantization import Quantization
from .pruning import Pruning
from .model_conversion import ModelConversion
from .graph_optimization import Graph_Optimization
from .benchmark import Benchmark

SUPPORTED_COMPONENTS = [
  Quantization,
  Pruning,
  Graph_Optimization,
  ModelConversion,
  Benchmark,
]

class Scheduler(object):
    """Scheduler for lpot component pipeline execution.

    LPOT supports serveral seperate components: Quantization, Pruning, Benchmarking.
    This scheduler will sequentially execute specified components by the order of
    appending. This interface provids an unique entry to pipeline execute all supported
    components.

    There are two typical usages:
    1) if all informations are set in user configuration yaml files by using lpot built-in
       dataloaders/datasets/metrics, the code usage is like below:

       prune = Pruning('/path/to/pruning.yaml')
       quantizer = Quantization('/path/to/quantization.yaml')
       scheduler = Scheduler()
       scheduler.model(common.Model('/path/to/model'))
       scheduler.append(prune)
       scheduler.append(quantizer)
       opt_model = scheduler()
       opt_model.save()


    2) if lpot built-in dataloaders/datasets/metrics could not fully meet user requirements,
       customized dataloaders/datasets/metrics are needed, the code usage is like below:

       prune = Pruning('/path/to/pruning.yaml')
       prune.pruning_func = ...                   # optional if it is configured in user yaml.
       prune.eval_dataloader = ...                # optional if it is configured in user yaml.
       prune.eval_func = ...                      # optional if it is configured in user yaml.
       
       quantizer = Quantization('/path/to/quantization.yaml')
       quantizer.metric = ...                     # optional if it is configured in user yaml.
       quantizer.calib_dataloader = ...           # optional if it is configured in user yaml.
       quantizer.eval_dataloader = ...            # optional if it is configured in user yaml.
       
       scheduler = Scheduler()
       scheduler.model(common.Model('/path/to/model'))
       scheduler.append(prune)
       scheduler.append(quantizer)
       opt_model = scheduler()
       opt_model.save()

    """

    def __init__(self):
        self._model = None
        self.components = []

    def append(self, *args):
        """Add lpot component into pipeline for sequential execution.

           Args:
               conf_fname (string):          The path to user configuration yaml file.
               calib_dataloader (generator): Optional. Data loader for calibration of Post-
                                             Training Static Quantization,
                                             or None for Post-Training Dynamic Quantization,
                                             or Data loader for training phase of Quantization-
                                             aware Training,
                                             or Data loader for training phase of Pruning,
                                             if its corresponding field is not configured in yaml.
               eval_dataloader (generator):  Optional. Data loader for evaluation phase of all
                                             lpot components, if its corresponding field is not
                                             configured in yaml and eval_func is not specified.
               postprocess (Postprocess):    Optional. Object initialized from common.Postprocess.
               metric (Metric):              Optional. Object initialized from common.Metric. 
               q_func (func):                Optional. Training function for Quantization-Aware
                                             Training and Pruning cases if user doesn't provide
                                             training info in user configuration yaml file.
               eval_func (func):             Optional. Evaluation function for Quantization-Aware
                                             Training and Pruning, being None for other cases. 
               kwargs (named arguments):     Reserved for interface extension.
        """ 
        for item in args:
            assert any([isinstance(item, supported_component) \
                        for supported_component in SUPPORTED_COMPONENTS])
            self.components.append(item)

    def __call__(self):
        """The main entry point of sequential pipeline execution.

        NOTE: it's user responsibility to ensure the output of each componenet could be fed as
        the input of next component.

        Returns:
            optimized model: best optimized model generated, otherwise return None

        """
        assert self.model, "Scheduler class's model property should be set " \
                           "before invoking this __call__() function"
        model = self.model
        assert len(self.components) > 0
        logger.info('Start sequential pipeline execution...')
        for i, component in enumerate(self.components):
            # print appropriate ordinal number representation (1st, 2nd, 3rd) for each step
            ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
            logger.info('The {} step being executing is {}'.format(
                        ordinal(i),
                        repr(component).upper()
                        ))

            component.model = model
            model = component()
            
        return model

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, user_model):
        """Set the user model and dispatch to framework specific internal model object

        Args:
           user_model: user are supported to set model from original framework model format
                       (eg, tensorflow frozen_pb or path to a saved model),
                       but not recommended. Best practice is to set from a initialized
                       lpot.experimental.common.Model.
                       If tensorflow model is used, model's inputs/outputs will be
                       auto inferenced, but sometimes auto inferenced
                       inputs/outputs will not meet your requests,
                       set them manually in config yaml file.
                       Another corner case is slim model of tensorflow,
                       be careful of the name of model configured in yaml file,
                       make sure the name is in supported slim model list.

        """
        if not isinstance(user_model, BaseModel):
            logger.warning('force convert user raw model to lpot model, ' +
                'better initialize lpot.experimental.common.Model and set....')
            self._model = LpotModel(user_model)
        else:
            self._model = user_model

