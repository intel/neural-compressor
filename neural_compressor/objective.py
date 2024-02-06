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
"""The objectives supported by neural_compressor, which is driven by accuracy.

To support new objective, developers just need implement a new subclass in this file.
"""
import time
import tracemalloc
from abc import abstractmethod
from copy import deepcopy
from typing import List, Tuple

import numpy as np

from .utils.utility import get_size

OBJECTIVES = {}


def objective_registry(cls):
    """The class decorator used to register all Objective subclasses.

    Args:
        cls (object): The class of register.

    Returns:
        cls (object): The class of register.
    """
    if cls.__name__.lower() in OBJECTIVES:
        raise ValueError("Cannot have two objectives with the same name")
    OBJECTIVES[cls.__name__.lower()] = cls
    return cls


def objective_custom_registry(name, obj_cls):
    """Register a customized objective.

    Example::

        from eural_compressor.objective import objective_custom_registry

        objective_cls = type(user_objective)
        name = user_objective.__class__.__name__
        objective_cfg = name if deep_get(self.conf.usr_cfg, "tuning.objective") else [name]
        deep_set(self.conf.usr_cfg, user_obj_cfg, objective_cfg)
        self.conf.usr_cfg = DotDict(self.conf.usr_cfg)
        objective_custom_registry(name, objective_cls)
    """
    if name.lower() in OBJECTIVES:
        raise ValueError("Cannot have two objectives with the same name")
    OBJECTIVES[name.lower()] = obj_cls


class Objective(object):
    """The base class for precise benchmark supported by neural_compressor.

    With built-in objectives, users can compress models with different objectives easily.
    In special cases, users can also register their own objective classes.
    """

    representation = ""

    def __init__(self):
        """The definition of the objective."""
        self._result_list = []
        self._model = None

    @abstractmethod
    def reset(self):
        """The interface reset benchmark measuring."""
        self._result_list = []
        return self._result_list

    @abstractmethod
    def start(self):
        """The interface start benchmark measuring."""
        raise NotImplementedError

    @abstractmethod
    def end(self):
        """The interface end benchmark measuring."""
        raise NotImplementedError

    @property
    def model(self):
        """The interface benchmark model."""
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def result(self, start=None, end=None):
        """The result will return the total mean of the result.

        The interface to get benchmark measuring result measurer may start and end many times,
        can set the start and end index of the result list to calculate.

        Args:
            start (int): start point to calculate result from result list
                         used to skip steps for warm up
            end (int): end point to calculate result from result list
        """
        start_idx = 0
        end_idx = len(self._result_list)
        if start is not None and start in range(0, 1 + len(self._result_list)):
            start_idx = start
        if end is not None and end in range(0, 1 + len(self._result_list)):
            end_idx = end
        return np.array(self._result_list[start_idx:end_idx]).mean()

    def result_list(self):
        """The interface to get benchmark measuring result list.

        This interface will return a list of each start-end loop measure value.
        """
        return self._result_list

    def __str__(self):
        """The interface to get the representation."""
        return self.representation


@objective_registry
class Accuracy(Objective):
    """Configuration Accuracy class.

    Evaluate the accuracy.
    """

    representation = "accuracy"

    def start(self):
        """The interface start the measuring."""
        pass

    def end(self, acc):
        """The interface end the measuring."""
        self._result_list.append(acc)


@objective_registry
class Performance(Objective):
    """Configuration Performance class.

    Evaluate the inference time.
    """

    representation = "duration (seconds)"

    def start(self):
        """Record the start time."""
        self.start_time = time.time()

    def end(self):
        """Record the duration time."""
        self.duration = time.time() - self.start_time
        assert self.duration >= 0, "please use start() before end()"
        self._result_list.append(self.duration)


@objective_registry
class Footprint(Objective):
    """Configuration Footprint class.

    Evaluate the peak size of memory blocks during inference.
    """

    representation = "memory footprint (MB)"

    def start(self):
        """Record the space allocation."""
        tracemalloc.start()

    def end(self):
        """Calculate the space usage."""
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self._result_list.append(peak // 1048576)


@objective_registry
class ModelSize(Objective):
    """Configuration ModelSize class.

    Evaluate the model size.
    """

    representation = "model size (MB)"

    def start(self):
        """Start to calculate the model size."""
        pass

    def end(self):
        """Get the actual model size."""
        model_size = get_size(self.model)
        self._result_list.append(model_size)


class MultiObjective:
    """The base class for multiple benchmarks supported by neural_compressor.

    Example::

        from neural_compressor.objective import MultiObjective
        obj = MultiObjective(
            objectives=['accuracy', 'modelsize', 'performance'],
            accuracy_criterion={'relative': 0.1},
            obj_criterion=[True, False, False],
            obj_weight=[0.7, 0.2, 0.1])
    """

    def __init__(
        self,
        objectives,
        accuracy_criterion,
        metric_criterion=[True],
        metric_weight=None,
        obj_criterion=None,
        obj_weight=None,
        is_measure=False,
    ):
        """Load the configuration."""
        assert isinstance(accuracy_criterion, dict), "accuracy criterian should be dict"
        assert (
            "relative" in accuracy_criterion or "absolute" in accuracy_criterion
        ), "accuracy criterion should set relative or absolute"
        self.higher_is_better = True
        for k, v in accuracy_criterion.items():
            if k in ["relative", "absolute"]:
                if k == "relative":
                    assert float(v) < 1 and float(v) > -1
                self.relative = True if k == "relative" else False
                self.acc_goal = float(v)
            elif k == "higher_is_better":
                self.higher_is_better = bool(v)

        self.objectives = [OBJECTIVES[i]() for i in objectives]
        self.representation = [str(i).capitalize() for i in self.objectives]
        self._baseline = None
        self.val = None
        if obj_criterion:
            if len(self.objectives) != len(obj_criterion) and len(obj_criterion) == 1:
                self.obj_criterion = obj_criterion * len(self.objectives)
            else:
                assert len(self.objectives) == len(obj_criterion)
                self.obj_criterion = obj_criterion
        else:
            self.obj_criterion = [False] * len(self.objectives)
        self.metric_weight = metric_weight
        self.metric_criterion = metric_criterion
        self.obj_weight = obj_weight
        self.is_measure = is_measure
        self._accuracy_target = None

    @property
    def baseline(self):
        """Get the actual model performance."""
        return self._baseline

    @baseline.setter
    def baseline(self, val):
        """Get the value of the performance."""
        self._baseline = val

    @property
    def accuracy_target(self):
        """Set the accuracy target."""
        return self._accuracy_target

    @accuracy_target.setter
    def accuracy_target(self, val):
        """Set the value of the accuracy target."""
        self._accuracy_target = val

    def compare(self, last, baseline):
        """The interface of comparing if metric reaches the goal with acceptable accuracy loss.

        Args:
            last (tuple): The tuple of last metric.
            baseline (tuple): The tuple saving FP32 baseline.
        """
        acc, perf = self.val

        if last is not None:
            _, last_measure = last
        else:
            last_measure = 0

        base_acc, _ = baseline

        if not isinstance(acc, list):
            acc = [acc]
            base_acc = [base_acc]

        if self.metric_weight is not None and len(base_acc) > 1:
            base_acc = [np.mean(np.array(base_acc) * self.metric_weight)]

        if self.relative:
            if len(base_acc) == 1:
                acc_target = [
                    (
                        base_acc[0] * (1 - float(self.acc_goal))
                        if self.higher_is_better
                        else base_acc[0] * (1 + float(self.acc_goal))
                    )
                ]
            else:
                # use metric_criterion to replace acc_criterion
                acc_target = [
                    b_acc * (1 - float(self.acc_goal)) if higher_is_better else b_acc * (1 + float(self.acc_goal))
                    for b_acc, higher_is_better in zip(base_acc, self.metric_criterion)
                ]
        else:
            if len(base_acc) == 1:
                acc_target = [
                    base_acc[0] - float(self.acc_goal) if self.higher_is_better else base_acc[0] + float(self.acc_goal)
                ]
            else:
                # use metric_criterion to replace acc_criterion
                acc_target = [
                    b_acc - float(self.acc_goal) if higher_is_better else b_acc + float(self.acc_goal)
                    for b_acc, higher_is_better in zip(base_acc, self.metric_criterion)
                ]

        if last_measure == 0 or all([(x <= y) ^ z for x, y, z in zip(perf, last_measure, self.obj_criterion)]):
            if len(base_acc) == 1:
                return acc[0] >= acc_target[0] if self.higher_is_better else acc[0] < acc_target[0]
            else:
                # use metric_criterion to replace acc_criterion
                return all(
                    [
                        sub_acc >= target if higher_is_better else sub_acc < target
                        for sub_acc, target, higher_is_better in zip(acc, acc_target, self.metric_criterion)
                    ]
                )
        else:
            return False

    def _get_accuracy_target(self):
        assert self._baseline is not None, "Baseline is None"
        base_acc, _ = self._baseline
        if not isinstance(base_acc, list):
            base_acc = [base_acc]
        if self.metric_weight is not None and len(base_acc) > 1:
            base_acc = [np.mean(np.array(base_acc) * self.metric_weight)]

        if self.relative:
            if len(base_acc) == 1:
                acc_target = [
                    (
                        base_acc[0] * (1 - float(self.acc_goal))
                        if self.higher_is_better
                        else base_acc[0] * (1 + float(self.acc_goal))
                    )
                ]
            else:
                # use metric_criterion to replace acc_criterion
                acc_target = [
                    b_acc * (1 - float(self.acc_goal)) if higher_is_better else b_acc * (1 + float(self.acc_goal))
                    for b_acc, higher_is_better in zip(base_acc, self.metric_criterion)
                ]
        else:
            if len(base_acc) == 1:
                acc_target = [
                    base_acc[0] - float(self.acc_goal) if self.higher_is_better else base_acc[0] + float(self.acc_goal)
                ]
            else:
                # use metric_criterion to replace acc_criterion
                acc_target = [
                    b_acc - float(self.acc_goal) if higher_is_better else b_acc + float(self.acc_goal)
                    for b_acc, higher_is_better in zip(base_acc, self.metric_criterion)
                ]
        return acc_target

    def accuracy_meets(self):
        """Verify the new model performance is better than the previous model performance."""
        last_acc, _ = deepcopy(self.val)
        got_better_result = False
        if not isinstance(last_acc, list):
            last_acc = [last_acc]

        if self.metric_weight is not None and len(last_acc) > 1:
            last_acc = [np.mean(np.array(last_acc) * self.metric_weight)]
        if not self._accuracy_target:
            self.accuracy_target = self._get_accuracy_target()
        all_higher = all([_last > _target for _last, _target in zip(last_acc, self.accuracy_target)])
        all_lower = all([_last < _target for _last, _target in zip(last_acc, self.accuracy_target)])
        got_better_result = (all_higher and self.higher_is_better) or (all_lower and not self.higher_is_better)
        return got_better_result

    def accuracy_meet_req(self, last_result: Tuple[float, List[float]]) -> bool:
        """Compare the result of last tuning with baseline to check whether the result meet requirements.

        Args:
            last_result: The evaluation result of the last tuning.

        Returns:
            check_result: Return True if the accuracy meets requirements else False.
        """
        check_result = False
        last_acc, _ = last_result
        if not isinstance(last_acc, list):
            last_acc = [last_acc]

        if self.metric_weight is not None and len(last_acc) > 1:
            last_acc = [np.mean(np.array(last_acc) * self.metric_weight)]
        if not self._accuracy_target:
            self.accuracy_target = self._get_accuracy_target()
        all_higher = all([_last > _target for _last, _target in zip(last_acc, self.accuracy_target)])
        all_lower = all([_last < _target for _last, _target in zip(last_acc, self.accuracy_target)])
        check_result = (all_higher and self.higher_is_better) or (all_lower and not self.higher_is_better)
        return check_result

    def evaluate(self, eval_func, model):
        """The interface of calculating the objective.

        Args:
            eval_func (function): function to do evaluation.
            model (object): model to do evaluation.
        """
        self.reset()
        self.set_model(model)
        if self.is_measure:
            acc = eval_func(model, self.objectives[0])
        else:
            self.start()
            acc = eval_func(model)
            self.end(acc)

        self.val = acc, self.result()
        return self.val

    def reset(self):
        """Reset the objective value."""
        for objective in self.objectives:
            objective.reset()

    def start(self):
        """Start to measure the objective value."""
        for objective in self.objectives:
            objective.start()

    def end(self, acc):
        """Calculate the objective value."""
        for objective in self.objectives:
            if isinstance(objective, Accuracy):
                objective.end(acc)
            else:
                objective.end()

    def result(self):
        """Get the results."""
        return [objective.result() for objective in self.objectives]

    def set_model(self, model):
        """Set the objective model."""
        for objective in self.objectives:
            objective.model = model

    def best_result(self, tune_data, baseline):
        """Calculate the best results.

        Example::

            # metric + multi-objectives case:
            tune_data = [
                [acc1, [obj1, obj2, ...]],
                [acc2, [obj1, obj2, ...]],
                ...
            ]
            # multi-metrics + multi-objectives case:
            tune_data = [
                [[acc1, acc2], [[acc1, acc2], obj1, obj2]],
                [[acc1, acc2], [[acc1, acc2], obj1, obj2]]
            ]
        """
        higher_is_better = self.higher_is_better
        obj_weight = self.obj_weight
        obj_criterion = self.obj_criterion

        base_acc, base_obj = baseline
        acc_data = [i[0] for i in tune_data]
        obj_data = [i[1] for i in tune_data]

        idx = self.representation.index("Accuracy") if "Accuracy" in self.representation else None
        if isinstance(base_acc, list) and len(base_acc) > 1 and idx is not None:
            if self.metric_weight is not None:
                for i in range(len(obj_data)):
                    obj_data[i][idx] = np.mean(np.array(obj_data[i][idx]) * self.metric_weight)
                base_obj[idx] = np.mean(np.array(base_obj[idx]) * self.metric_weight)
                assert obj_criterion[idx] == self.higher_is_better, (
                    "Accuracy criterion and "
                    "objective criterion have conflict on whether weighted accuracy is "
                    "higher-is-better"
                )
            else:
                # use metric_criterion to replace acc_criterion
                higher_is_better = self.metric_criterion
                for i in range(len(obj_data)):
                    tmp = obj_data[i].pop(idx)
                    obj_data[i].extend(tmp)
                tmp = base_obj.pop(idx)
                base_obj.extend(tmp)
                obj_criterion.pop(idx)
                # use metric_criterion to replace acc_criterion in obj_criterion
                obj_criterion.extend(self.metric_criterion)
                if obj_weight:
                    # convert 1 acc_weight to acc_num (acc_weight/acc_num)s
                    tmp = obj_weight.pop(idx)
                    tmp = [tmp / len(base_acc)] * len(base_acc)
                    obj_weight.extend(tmp)

        base_obj = np.array(base_obj)
        base_obj[base_obj == 0] = 1e-20

        acc_data = np.array(acc_data)
        obj_data = np.array(obj_data)
        obj_data[obj_data == 0] = 1e-20

        if not isinstance(base_acc, list):
            base_acc = [base_acc]
        elif len(base_acc) > 1 and self.metric_weight is not None:
            base_acc = [np.mean(np.array(base_acc) * self.metric_weight)]
            acc_data = np.mean(acc_data * self.metric_weight, axis=1)

        if len(base_acc) == 1:
            if self.relative:
                acc_target = [
                    (
                        base_acc[0] * (1 - float(self.acc_goal))
                        if self.higher_is_better
                        else base_acc[0] * (1 + float(self.acc_goal))
                    )
                ]
            else:
                acc_target = [
                    base_acc[0] - float(self.acc_goal) if self.higher_is_better else base_acc[0] + float(self.acc_goal)
                ]
            acc_mask = acc_data >= acc_target if self.higher_is_better else acc_data < acc_target
        else:
            if self.relative:
                acc_target = [
                    b_acc * (1 - float(self.acc_goal)) if higher_is_better else b_acc * (1 + float(self.acc_goal))
                    for b_acc, higher_is_better in zip(base_acc, self.metric_criterion)
                ]
            else:
                acc_target = [
                    b_acc - float(self.acc_goal) if higher_is_better else b_acc + float(self.acc_goal)
                    for b_acc, higher_is_better in zip(base_acc, self.metric_criterion)
                ]
            acc_mask = np.all(
                [
                    [
                        item >= target if higher_is_better else item < target
                        for item, target, higher_is_better in zip(acc_item, acc_target, self.metric_criterion)
                    ]
                    for acc_item in acc_data
                ],
                axis=1,
            )

        obj_data = base_obj / obj_data

        # normalize data
        if idx is not None and not self.relative:
            if not isinstance(higher_is_better, list):
                obj_data[:, idx] = base_obj[idx] - 1 / (obj_data[:, idx] / base_obj[idx])
            else:
                for i in range(len(higher_is_better)):
                    if higher_is_better[i]:
                        k = i - len(higher_is_better)
                        obj_data[:, k] = base_obj[k] - 1 / (obj_data[:, k] / base_obj[k])

        min_val = np.min(obj_data, axis=0)
        max_val = np.max(obj_data, axis=0)
        zero_mask = max_val != min_val

        # convert higher-is-better to lower-is-better
        obj_criterion = np.array(obj_criterion)
        obj_data[:, obj_criterion] = max_val[obj_criterion] + min_val[obj_criterion] - obj_data[:, obj_criterion]

        obj_data[:, zero_mask] = (obj_data[:, zero_mask] - min_val[zero_mask]) / (
            max_val[zero_mask] - min_val[zero_mask]
        )
        if obj_weight:
            weighted_result = np.sum(obj_weight * obj_data, axis=1)
        else:
            weighted_result = np.sum(obj_data, axis=1)

        weighted_result[acc_mask == False] = 0.0  # noqa: E712
        best_trial = np.argmax(weighted_result)
        return best_trial, tune_data[best_trial]
