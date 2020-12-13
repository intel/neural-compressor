#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Intel Corporation
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
import time
import numpy as np
import tracemalloc
from .utils.utility import get_size

"""The objectives supported by lpot, which is driven by accuracy.
   To support new objective, developer just need implement a new subclass in this file.
"""
OBJECTIVES = {}

def objective_registry(cls):
    """The class decorator used to register all Objective subclasses.

    Args:
        cls (object): The class of register.

    Returns:
        cls (object): The class of register.
    """
    if cls.__name__.lower() in OBJECTIVES:
        raise ValueError('Cannot have two objectives with the same name')
    OBJECTIVES[cls.__name__.lower()] = cls
    return cls

class Measurer(object):
    """The base class for precise benchmark supported by lpot.

    """

    def __init__(self):
        self._result_list = [] 
    
    @abstractmethod
    def reset(self):
        """The interface reset benchmark measuring
        Args:
        """
        self._result_list = []
        return self._result_list

    @abstractmethod
    def start(self, *args, **kwargs):
        """The interface start benchmark measuring
        Args:
           *args: params for start the measuring
           **kargs: params for start the measuring
        """
        raise NotImplementedError
    
    @abstractmethod
    def end(self, *args, **kwargs):
        """The interface end benchmark measuring
        Args:
           *args: params for start the measuring
           **kargs: params for start the measuring
        """
        raise NotImplementedError

    def result(self, start=None, end=None):
        """The interface to get benchmark measuring result
           measurer may sart and end many times, result will
           return the total mean of the result, can set the start
           and end index of the result list to calculate
        
        Args:
            start (int): start point to calculate result from result list
                         used to skip steps for warm up
            end (int): end point to calculate result from result list 
        """
        start_idx = 0
        end_idx = len(self._result_list)
        if start is not None and start in range(0, 1+len(self._result_list)):
            start_idx = start
        if end is not None and end in range(0, 1+len(self._result_list)):
            end_idx = end
        return np.array(self._result_list[start_idx:end_idx]).mean()

    def result_list(self):
        """The interface to get benchmark measuring result list
           this interface will return a list of each start-end loop
           measure value
        Args:
        """
        return self._result_list

class PerformanceMeasure(Measurer):

    def start(self):
        self.start_time = time.time()

    def end(self):
        self.duration = time.time() - self.start_time
        assert self.duration > 0, 'please use start() before end()'
        self._result_list.append(self.duration)

class FootprintMeasure(Measurer):
    def start(self):
        tracemalloc.start()

    def end(self):
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self._result_list.append(peak)

class ModelSizeMeasure(Measurer):
    def start(self, model):
        pass

    def end(self, model):
        model_size = get_size(model)
        self._result_list.append(model_size)


class Objective(object):
    """The base class of objectives supported by lpot.

    Args:
        accuracy_criterion (dict): The dict of supported accuracy criterion.
                                    {'relative': 0.01} or {'absolute': 0.01}
    """

    def __init__(self, accuracy_criterion, is_measure=False):
        assert isinstance(
            accuracy_criterion,
            dict) and len(accuracy_criterion) == 1
        k, v = list(accuracy_criterion.items())[0]
        assert k in ['relative', 'absolute']
        assert float(v) < 1 and float(v) > -1

        self.acc_goal = float(v)
        self.relative = True if k == 'relative' else False
        self.baseline = None
        self.val = None
        self.is_measure = is_measure
        self.measurer = None

    def compare(self, last, baseline):
        """The interface of comparing if metric reaches 
           the goal with acceptable accuracy loss.

        Args:
            last (tuple): The tuple of last metric.
            accuracy_criterion (float): The allowed accuracy absolute loss.
            baseline (tuple): The tuple saving FP32 baseline.
        """
        acc, perf = self.val

        if last is not None:
            _, last_measure = last
        else:
            last_measure = 0

        base_acc, _ = baseline

        acc_target = base_acc - float(self.acc_goal) if not self.relative \
            else base_acc * (1 - float(self.acc_goal))
        if acc >= acc_target and (last_measure == 0 or perf < last_measure):
            return True
        else:
            return False

    def evaluate(self, eval_func, model):
        """The interface of calculating the objective.

        Args:
            eval_func (function): function to do evaluation.
            model (object): model to do evaluation.
        """

        self.measurer.reset()
        if self.is_measure:
            acc = eval_func(model, self.measurer)
        else:
           self.measurer.start()
           acc = eval_func(model)
           self.measurer.end()

        self.val = acc, self.measurer.result()
        return self.val

@objective_registry
class Performance(Objective):
    """The objective class of calculating performance when running quantize model.
    Args:
        accuracy_criterion (dict): The dict of supported accuracy criterion.
                                    {'relative': 0.01} or {'absolute': 0.01}
    """

    def __init__(self, accuracy_criterion, is_measure=False):
        super(Performance, self).__init__(accuracy_criterion, is_measure)
        self.measurer = PerformanceMeasure()

@objective_registry
class Footprint(Objective):
    """The objective class of calculating peak memory footprint 
       when running quantize model.

    Args:
        accuracy_criterion (dict): The dict of supported accuracy criterion.
                                    {'relative': 0.01} or {'absolute': 0.01}
    """

    def __init__(self, accuracy_criterion):
        super(Footprint, self).__init__(accuracy_criterion)
        self.measurer = FootprintMeasure()

@objective_registry
class ModelSize(Objective):
    """The objective class of calculating model size when running quantize model.

    Args:
        accuracy_criterion (dict): The dict of supported accuracy criterion.
                                    {'relative': 0.01} or {'absolute': 0.01}
    """

    def __init__(self, accuracy_criterion):
        super(ModelSize, self).__init__(accuracy_criterion)
        self.measurer = ModelSizeMeasure()

