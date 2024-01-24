#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
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
"""Initialize critetion classes.

Classes includes:
    TensorFlowCrossEntropyLoss,
    TensorFlowSparseCategoricalCrossentropy,
    TensorflowKnowledgeDistillationLoss
"""

from collections import Counter

import numpy as np

from neural_compressor.adaptor.pytorch import pytorch_forward_wrapper
from neural_compressor.utils import logger
from neural_compressor.utils.utility import LazyImport, singleton

torch = LazyImport("torch")
tf = LazyImport("tensorflow")


@singleton
class TensorflowCriterions(object):
    """Record criterions in TensorflowCriterions class."""

    def __init__(self):
        """Initialize the TensorflowCriterions class."""
        self.criterions = {}
        self.criterions.update(TENSORFLOW_CRITERIONS)


framework_criterions = {
    "tensorflow": TensorflowCriterions,
}

# user/model specific criterions will be registered here
TENSORFLOW_CRITERIONS = {}
PYTORCH_CRITERIONS = {}

registry_criterions = {
    "tensorflow": TENSORFLOW_CRITERIONS,
}


class Criterions(object):
    """Integrate criterions of different framework."""

    def __init__(self, framework):
        """Initialize the Criterions class.

        Args:
            framework (string): framework name.
        """
        assert framework in ("tensorflow", "pytorch", "pytorch_fx"), "framework support tensorflow pytorch"
        self.criterions = framework_criterions[framework]().criterions

    def __getitem__(self, criterion_type):
        """Fetch criterion result by criterion type.

        Args:
            criterion_type (string): criterion type.

        Returns:
            cls: criterion class.
        """
        assert (
            criterion_type in self.criterions.keys()
        ), "only support criterions in {} \
            , but got criterion type {}".format(
            self.criterions.keys(), criterion_type
        )

        return self.criterions[criterion_type]

    def register(self, name, criterion_cls):
        """Register criterion name and result in existing criterion class.

        Args:
            name (string): criterion name/type.
            criterion_cls (string): criterion class.
        """
        assert name not in self.criterions.keys(), "registered criterion name already exists."
        self.criterions.update({name: criterion_cls})


def criterion_registry(criterion_type, framework):
    """Use to register criterion classes in registry_criterions.

    Args:
        criterion_type (str): The string of supported criterion.
        framework (str): The string of supported framework.

    Returns:
        cls: The class of register.
    """

    def decorator_criterion(cls):
        """Decorate criterion class to check framework and criterion name."""
        for fw in [fwk.strip() for fwk in framework.split(",")]:
            assert fw in ["tensorflow", "pytorch"], "The framework support tensorflow pytorch"

            if criterion_type in registry_criterions[fw].keys():
                raise ValueError("Cannot have two criterions with the same name")
            registry_criterions[fw][criterion_type] = cls
        return cls

    return decorator_criterion


@criterion_registry("CrossEntropyLoss", "tensorflow")
class TensorFlowCrossEntropyLoss(object):
    """TensorFlow CrossEntropyLoss criterion."""

    def __init__(self, param_dict):
        """Initialize the Datasets class.

        Args:
            param_dict (dict): The dict of parameters setting by user for CrossEntropyLoss criterion.
        """
        assert isinstance(param_dict, dict), "This criterion constructor parameter must be a dict"
        self._param_dict = param_dict

    def _mapping(self):
        _param_map = {"reduction": "reduction", "from_logits": "from_logits"}
        _dict = {}
        for key in self._param_dict:
            if key in _param_map:
                if key == "reduction":
                    assert self._param_dict[key] in [
                        "auto",
                        "none",
                        "sum",
                        "sum_over_batch_size",
                    ], "Supported reduction value for tensorflow is auto, none, sum, sum_over_batch_size"
                _dict.update({_param_map[key]: self._param_dict[key]})
        return _dict

    def __call__(self):
        """Call the TensorFlowCrossEntropyLoss.

        Returns:
            cls: criterion class.
            param_dict(dict): param_dict
        """
        return tf.keras.losses.CategoricalCrossentropy, self._mapping()


@criterion_registry("SparseCategoricalCrossentropy", "tensorflow")
class TensorFlowSparseCategoricalCrossentropy(object):
    """TensorFlow SparseCategoricalCrossentropyLoss criterion."""

    def __init__(self, param_dict):
        """Initialize the Datasets class.

        Args:
            param_dict (string): param_dict.
        """
        assert isinstance(param_dict, dict), "This criterion constructor parameter must be a dict"
        self._param_dict = param_dict

    def _mapping(self):
        _param_map = {"reduction": "reduction", "from_logits": "from_logits"}
        _dict = {}
        for key in self._param_dict:
            if key in _param_map:
                if key == "reduction":
                    assert self._param_dict[key] in [
                        "auto",
                        "none",
                        "sum",
                        "sum_over_batch_size",
                    ], "Supported reduction value for tensorflow is auto, none, sum, sum_over_batch_size"
                _dict.update({_param_map[key]: self._param_dict[key]})
        return _dict

    def __call__(self):
        """Call the TensorFlowSparseCategoricalCrossentropy.

        Returns:
            cls: criterion class.
            param_dict(dict): param_dict
        """
        return tf.keras.losses.SparseCategoricalCrossentropy, self._mapping()


class KnowledgeDistillationFramework(object):
    """Knowledge Distillation Framework."""

    def __init__(self, student_model=None, teacher_model=None):
        """Initialize the KnowledgeDistillationFramework class.

        Args:
            student_model: student_model.
            teacher_model: teacher_model.
        """
        self._student_model = student_model
        self._teacher_model = teacher_model

    @property
    def student_model(self):
        """Return student model."""
        return self._student_model

    @student_model.setter
    def student_model(self, model):
        """Setter of teacher model."""
        self._student_model = model

    @property
    def teacher_model(self):
        """Return teacher model."""
        return self._teacher_model

    @teacher_model.setter
    def teacher_model(self, model):
        """Setter of teacher model."""
        self._teacher_model = model


class KnowledgeDistillationLoss(KnowledgeDistillationFramework):
    """Initialize the KnowledgeDistillationLoss class."""

    def __init__(
        self, temperature=1.0, loss_types=["CE", "CE"], loss_weights=[0.5, 0.5], student_model=None, teacher_model=None
    ):
        """Initialize Knowledge Distillation Loss class.

        Args:
            temperature (float, optional): Hyperparameters that control the entropy
                                            of probability distributions. Defaults to 1.0.
            loss_types (list, optional): loss type. Defaults to ['CE', 'CE'].
            loss_weights (list, optional): loss weights. Defaults to [0.5, 0.5].
            student_model (model, optional): student model. Defaults to None.
            teacher_model (model, optional): teacher model. Defaults to None.
        """
        super(KnowledgeDistillationLoss, self).__init__(student_model=student_model, teacher_model=teacher_model)
        self.teacher_outputs = None
        self.temperature = temperature
        self.loss_weights = loss_weights
        self.loss_types = loss_types
        self.teacher_student_loss = self.student_targets_loss = None
        assert len(loss_weights) == len(loss_types) == 2, (
            "Wrong length for " + "loss_weights or loss_types, should be 2."
        )
        assert sum(loss_weights) == 1.0, "Sum of loss_weights should be 1.0."

    def teacher_model_forward(self, input, teacher_model=None):
        """Define parameters for teacher_model_forward function.

        Args:
            input (tensor, tuple or dict): input data.
            teacher_model (model, optional): teacher model. Defaults to None.

        Raises:
            NotImplementedError: NotImplementedError
        """
        raise NotImplementedError("Function teacher_model_forward " "should be framework related.")

    def teacher_student_loss_cal(self, student_outputs, teacher_outputs):
        """Define parameters for teacher_student_loss_cal function.

        Args:
            student_outputs (tensor): student outputs
            teacher_outputs (tensor): student outputs

        Raises:
            NotImplementedError: NotImplementedError
        """
        raise NotImplementedError("Function teacher_student_loss_cal " "should be framework related.")

    def student_targets_loss_cal(self, student_outputs, targets):
        """Define parameters for student_targets_loss_cal function.

        Args:
            student_outputs (tensor): student outputs
            targets (tensor): groud truth label

        Raises:
            NotImplementedError: NotImplementedError
        """
        raise NotImplementedError("Function student_targets_loss_cal " "should be framework related.")

    def loss_cal(self, student_outputs, targets):
        """Calculate loss of student model.

        Args:
            student_outputs (tensor): student outputs
            targets (tensor): groud truth label

        Returns:
            tensor: loss
        """
        return self.student_targets_loss_cal(student_outputs, targets)

    def loss_cal_sloss(self, student_outputs, teacher_outputs, student_loss):
        """Calculate all losses between student model and teacher model.

        Args:
            student_outputs (tensor): student outputs
            teacher_outputs (tensor): teacher outputs
            student_loss (tensor): student loss

        Returns:
            tensor: loss
        """
        if self.loss_weights[0] > 0:
            origin_loss = student_loss
        else:
            origin_loss = 0

        if self.loss_weights[1] > 0:
            student_out_ = student_outputs / self.temperature
            teacher_out_ = teacher_outputs / self.temperature
            distillation_loss = self.teacher_student_loss_cal(student_out_, teacher_out_)
            distillation_loss *= self.temperature**2
        else:
            distillation_loss = 0

        self.loss = origin_loss * self.loss_weights[0] + distillation_loss * self.loss_weights[1]
        return self.loss

    def __call__(self, student_outputs, targets):
        """Call the class to calculate loss.

        Args:
            student_outputs (tensor): student outputs
            targets (tensor): groud truth label

        Returns:
            tensor: loss
        """
        return self.loss_cal(student_outputs, targets)


class TensorflowKnowledgeDistillationLoss(KnowledgeDistillationLoss):
    """The TensorflowKnowledgeDistillationLoss class inherits from KnowledgeDistillationLoss."""

    def __init__(
        self, temperature=1.0, loss_types=["CE", "CE"], loss_weights=[0.5, 0.5], student_model=None, teacher_model=None
    ):
        """Initialize Tensorflow Knowledge Distillation Loss class.

        Args:
            temperature (float, optional): Hyperparameters that control the entropy
                                            of probability distributions. Defaults to 1.0.
            loss_types (list, optional): loss types. Defaults to ['CE', 'CE'].
            loss_weights (list, optional): loss weights. Defaults to [0.5, 0.5].
            student_model (optional): student model. Defaults to None.
            teacher_model (optional): teacher model. Defaults to None.

        Raises:
            NotImplementedError: NotImplementedError
            NotImplementedError: NotImplementedError
        """
        super(TensorflowKnowledgeDistillationLoss, self).__init__(
            temperature=temperature,
            loss_types=loss_types,
            loss_weights=loss_weights,
            student_model=student_model,
            teacher_model=teacher_model,
        )
        if self.student_targets_loss is None:
            if self.loss_types[0] == "CE":
                self.student_targets_loss = tf.keras.losses.SparseCategoricalCrossentropy()
            else:
                raise NotImplementedError(
                    "Now we only support CrossEntropyLoss " "for loss of student model output with respect to targets."
                )
            logger.info("student_targets_loss: {}, {}".format(self.loss_types[0], self.loss_weights[0]))
        if self.teacher_student_loss is None:
            if self.loss_types[1] == "CE":
                self.teacher_student_loss = self.SoftCrossEntropy
            elif self.loss_types[1] == "KL":
                self.teacher_student_loss = tf.keras.losses.KLDivergence()
            else:
                raise NotImplementedError(
                    "Now we only support CrossEntropyLoss"
                    " for loss of student model output with respect to teacher model output."
                )
            logger.info("teacher_student_loss: {}, {}".format(self.loss_types[1], self.loss_weights[1]))

    def SoftCrossEntropy(self, targets, logits):
        """Return SoftCrossEntropy.

        Args:
            logits (tensor): output logits
            targets (tensor): ground truth label

        Returns:
            tensor: SoftCrossEntropy
        """
        log_prob = tf.math.log(logits)
        targets_prob = targets
        return tf.math.reduce_mean(tf.math.reduce_sum(-targets_prob * log_prob, axis=-1), axis=-1)

    def teacher_model_forward(self, input, teacher_model=None):
        """Teacher model forward.

        Args:
            input (tensor): input data
            teacher_model (optional): teacher model. Defaults to None.
            device (torch.device, optional): device. Defaults to None.

        Returns:
            tensor: output
        """
        outputs = None
        if self.loss_weights[1] > 0 and input is not None:
            model = self.teacher_model if teacher_model is None else teacher_model
            if isinstance(input, list) or isinstance(input, tuple):  # pragma: no cover
                outputs = model(*input, training=True)
            elif isinstance(input, dict):  # pragma: no cover
                outputs = model(**input, training=True)
            else:
                outputs = model(input, training=True)
            self.teacher_outputs = outputs
        return outputs

    def teacher_student_loss_cal(self, student_outputs, teacher_outputs):
        """Calculate loss between student model and teacher model.

        Args:
            student_outputs (tensor): student outputs
            teacher_outputs (tensor): teacher outputs

        Returns:
            tensor: loss
        """
        assert self.teacher_student_loss, "teacher_student_loss not specified."
        return self.teacher_student_loss(teacher_outputs, student_outputs)

    def student_targets_loss_cal(self, student_outputs, targets):
        """Calculate loss of student model.

        Args:
            student_outputs (tensor): student outputs
            targets (tensor): groud truth label

        Returns:
            tensor: loss
        """
        assert self.student_targets_loss, "student_targets_loss not specified."
        return self.student_targets_loss(targets, student_outputs)

    def __call__(self, student_outputs, targets):
        """Return loss of student model.

        Args:
            student_outputs (tensor): student outputs
            targets (tensor): groud truth label

        Returns:
            tensor: loss
        """
        tmp = student_outputs
        student_outputs = targets
        targets = tmp
        return self.loss_cal(student_outputs, targets)


@criterion_registry("KnowledgeDistillationLoss", "tensorflow")
class TensorflowKnowledgeDistillationLossWrapper(object):
    """TensorflowKnowledgeDistillationLossWrapper wraps TensorflowKnowledgeDistillationLoss."""

    def __init__(self, param_dict):
        """Initialize Tensorflow Knowledge Distillation Loss Wrapper.

        Args:
            param_dict (dict): parameter dict
        """
        self.param_dict = param_dict

    def _param_check(self):
        param_dict = self.param_dict
        _params = ["temperature", "loss_types", "loss_weights"]
        assert all(key in param_dict for key in _params), "Keys {} must be in input parameters.".format(_params)
        assert param_dict["temperature"] > 0.0, "Value of temperature must be positive."
        assert len(param_dict["loss_types"]) == len(
            param_dict["loss_weights"]
        ), "Length of loss_types and loss_weights must be the same."
        assert all(
            type(param_dict[k]) in [list, tuple] for k in ["loss_types", "loss_weights"]
        ), "Type of loss_types and loss_weights must be list or tuple."
        assert all(
            any(isinstance(e, t) for t in [str, tf.keras]) for e in param_dict["loss_types"]
        ), "Type of loss_types element must be str or torch Module."
        assert (
            all(0.0 <= e <= 1.0 for e in param_dict["loss_weights"])
            and abs(sum(param_dict["loss_weights"]) - 1.0) < 1e-9
        ), "Element of loss_weights must be in interval [0, 1] and summed to 1.0."
        new_dict = {}
        for k in _params:
            new_dict[k] = param_dict[k]
        return new_dict

    def __call__(self, **kwargs):
        """Return TensorflowKnowledgeDistillationLoss, param dict.

        Returns:
            class: TensorflowKnowledgeDistillationLoss
            param dict (dict): param dict
        """
        return TensorflowKnowledgeDistillationLoss, self._param_check()


class TensorflowKnowledgeDistillationLossExternal(KnowledgeDistillationLoss):
    """TensorflowKnowledgeDistillationLossExternal inherits from KnowledgeDistillationLoss."""

    def __init__(
        self, temperature=1.0, loss_types=["CE", "CE"], loss_weights=[0.5, 0.5], student_model=None, teacher_model=None
    ):
        """Initialize Tensorflow Knowledge Distillation Loss class.

        Args:
            temperature (float, optional): Hyperparameters that control the entropy
                                            of probability distributions. Defaults to 1.0.
            loss_types (list, optional): loss types. Defaults to ['CE', 'CE'].
            loss_weights (list, optional): loss weights. Defaults to [0.5, 0.5].
            student_model (optional): student model. Defaults to None.
            teacher_model (optional): teacher model. Defaults to None.

        Raises:
            NotImplementedError: NotImplementedError
            NotImplementedError: NotImplementedError
        """
        super(TensorflowKnowledgeDistillationLossExternal, self).__init__(
            temperature=temperature,
            loss_types=loss_types,
            loss_weights=loss_weights,
            student_model=student_model,
            teacher_model=teacher_model,
        )
        if self.student_targets_loss is None:
            if self.loss_types[0] == "CE":
                self.student_targets_loss = tf.keras.losses.CategoricalCrossentropy()
            else:
                raise NotImplementedError(
                    "Now we only support CrossEntropyLoss " "for loss of student model output with respect to targets."
                )
            logger.info("student_targets_loss: {}, {}".format(self.loss_types[0], self.loss_weights[0]))
        if self.teacher_student_loss is None:
            if self.loss_types[1] == "CE":
                self.teacher_student_loss = tf.keras.losses.CategoricalCrossentropy()
            elif self.loss_types[1] == "KL":
                self.teacher_student_loss = tf.keras.losses.KLDivergence()
            else:
                raise NotImplementedError(
                    "Now we only support CrossEntropyLoss"
                    " for loss of student model output with respect to teacher model output."
                )
            logger.info("teacher_student_loss: {}, {}".format(self.loss_types[1], self.loss_weights[1]))

    def teacher_model_forward(self, input, teacher_model=None):
        """Teacher model forward.

        Args:
            input (tensor): input data
            teacher_model (optional): teacher model. Defaults to None.
            device (optional): device. Defaults to None.

        Returns:
            tensor: output
        """
        outputs = None
        if self.loss_weights[1] > 0 and input is not None:
            model = self.teacher_model if teacher_model is None else teacher_model
            if isinstance(input, list) or isinstance(input, tuple):  # pragma: no cover
                outputs = model(*input, training=True)
            elif isinstance(input, dict):  # pragma: no cover
                outputs = model(**input, training=True)
            else:
                outputs = model(input, training=True)
            self.teacher_outputs = outputs
        return outputs

    def teacher_student_loss_cal(self, student_outputs, teacher_outputs):
        """Calculate loss between student model and teacher model.

        Args:
            student_outputs (tensor): student outputs
            teacher_outputs (tensor): teacher outputs

        Returns:
            tensor: loss
        """
        assert self.teacher_student_loss, "teacher_student_loss not specified."
        return self.teacher_student_loss(teacher_outputs, student_outputs)

    def student_targets_loss_cal(self, student_outputs, targets):
        """Calculate loss of student model.

        Args:
            student_outputs (tensor): student outputs
            targets (tensor): groud truth label

        Returns:
            tensor: loss
        """
        assert self.student_targets_loss, "student_targets_loss not specified."
        return self.student_targets_loss(targets, student_outputs)

