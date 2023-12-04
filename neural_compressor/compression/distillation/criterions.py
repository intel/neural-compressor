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
"""Initialize critetion classes.

Classes includes:
    TensorFlowCrossEntropyLoss, PyTorchCrossEntropyLoss,
    TensorFlowSparseCategoricalCrossentropy,
    TensorflowKnowledgeDistillationLoss, PyTorchKnowledgeDistillationLoss,
    PyTorchIntermediateLayersKnowledgeDistillationLoss.
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


@singleton
class PyTorchCriterions(object):
    """Record criterions in PyTorchCriterions class."""

    def __init__(self):
        """Initialize the TensorflowCriterions class."""
        self.criterions = {}
        self.criterions.update(PYTORCH_CRITERIONS)


framework_criterions = {
    "tensorflow": TensorflowCriterions,
    "pytorch": PyTorchCriterions,
    "pytorch_fx": PyTorchCriterions,
}

# user/model specific criterions will be registered here
TENSORFLOW_CRITERIONS = {}
PYTORCH_CRITERIONS = {}

registry_criterions = {
    "tensorflow": TENSORFLOW_CRITERIONS,
    "pytorch": PYTORCH_CRITERIONS,
    "pytorch_fx": PYTORCH_CRITERIONS,
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


@criterion_registry("CrossEntropyLoss", "pytorch")
class PyTorchCrossEntropyLoss(object):
    """PyTorch CrossEntropyLoss criterion."""

    def __init__(self, param_dict):
        """Initialize the PyTorchCrossEntropyLoss class.

        Args:
            param_dict (string): param_dict.
        """
        assert isinstance(param_dict, dict), "This criterion constructor parameter must be a dict"
        self._param_dict = param_dict

    def _mapping(self):
        _param_map = {"reduction": "reduction"}
        _dict = {}
        for key in self._param_dict:
            if key in _param_map:
                if key == "reduction":
                    assert self._param_dict[key] in [
                        "none",
                        "mean",
                        "sum",
                    ], "Supported reduction value is none, mean, sum"
                _dict.update({_param_map[key]: self._param_dict[key]})
        return _dict

    def __call__(self):
        """Call the PyTorchCrossEntropyLoss.

        Returns:
            cls: criterion class.
            param_dict(dict): param_dict
        """
        return torch.nn.CrossEntropyLoss, self._mapping()


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


class PyTorchKnowledgeDistillationLoss(KnowledgeDistillationLoss):
    """The PyTorchKnowledgeDistillationLoss class inherits from KnowledgeDistillationLoss."""

    def __init__(
        self, temperature=1.0, loss_types=["CE", "CE"], loss_weights=[0.5, 0.5], student_model=None, teacher_model=None
    ):
        """Initialize PyTorch Knowledge Distillation Loss class.

        Args:
            temperature (float, optional): Hyperparameters that control the entropy
                                            of probability distributions. Defaults to 1.0.
            loss_types (list, optional): loss types. Defaults to ['CE', 'CE'].
            loss_weights (list, optional): loss weights. Defaults to [0.5, 0.5].
            student_model (torch.nn.model, optional): student model. Defaults to None.
            teacher_model (torch.nn.model, optional): teacher model. Defaults to None.

        Raises:
            NotImplementedError: NotImplementedError
            NotImplementedError: NotImplementedError
        """
        super(PyTorchKnowledgeDistillationLoss, self).__init__(
            temperature=temperature,
            loss_types=loss_types,
            loss_weights=loss_weights,
            student_model=student_model,
            teacher_model=teacher_model,
        )
        if self.student_targets_loss is None:
            if self.loss_types[0] == "CE":
                self.student_targets_loss = torch.nn.CrossEntropyLoss()
            elif self.loss_types[0] == "MSE":
                self.student_targets_loss = torch.nn.MSELoss()
            else:
                raise NotImplementedError(
                    "Now we only support CrossEntropyLoss and MSELoss "
                    "for loss of student model output with respect to targets."
                )
            logger.info("student_targets_loss: {}, {}".format(self.loss_types[0], self.loss_weights[0]))
        if self.teacher_student_loss is None:
            if self.loss_types[1] == "CE":
                self.teacher_student_loss = self.SoftCrossEntropy
            elif self.loss_types[1] == "KL":
                self.teacher_student_loss = self.KullbackLeiblerDivergence
            elif self.loss_types[1] == "MSE":
                self.teacher_student_loss = torch.nn.MSELoss()
            else:
                raise NotImplementedError(
                    "Now we only support CrossEntropyLoss KL Divergence"
                    " and MSELoss for loss of student model output with respect to teacher model output."
                )
            logger.info("teacher_student_loss: {}, {}".format(self.loss_types[1], self.loss_weights[1]))

    def SoftCrossEntropy(self, logits, targets):
        """Return SoftCrossEntropy.

        Args:
            logits (tensor): output logits
            targets (tensor): ground truth label

        Returns:
            tensor: SoftCrossEntropy
        """
        log_prob = torch.nn.functional.log_softmax(logits, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        return (-targets_prob * log_prob).sum(dim=-1).mean()

    def KullbackLeiblerDivergence(self, logits, targets):
        """Return KullbackLeiblerDivergence.

        Args:
            logits (tensor): output logits
            targets (tensor): ground truth label

        Returns:
            tensor: KullbackLeiblerDivergence
        """
        log_prob = torch.nn.functional.log_softmax(logits, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        return torch.nn.functional.kl_div(log_prob, targets_prob)

    def teacher_model_forward(self, input, teacher_model=None, device=None):
        """Teacher model forward.

        Args:
            input (tensor): input data
            teacher_model (torch.nn.model, optional): teacher model. Defaults to None.
            device (torch.device, optional): device. Defaults to None.

        Returns:
            tensor: output
        """
        outputs = None
        if self.loss_weights[1] > 0:
            model = self.teacher_model if teacher_model is None else teacher_model
            assert isinstance(model, torch.nn.Module), "Teacher model should be a torch Module instead of {}".format(
                type(model)
            )
            model.eval()
            try:
                model_device = next(model.parameters()).device
            except:
                logger.warning("Cannot get model device, assuming it's in CPU.")
                model_device = "cpu"
            device = model_device if device is None else device
            if device != model_device:
                model.to(device)
            with torch.no_grad():
                outputs = pytorch_forward_wrapper(model, input)
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
        return self.teacher_student_loss(student_outputs, teacher_outputs)

    def student_targets_loss_cal(self, student_outputs, targets):
        """Calculate loss of student model.

        Args:
            student_outputs (tensor): student outputs
            targets (tensor): groud truth label

        Returns:
            tensor: loss
        """
        assert self.student_targets_loss, "student_targets_loss not specified."
        return self.student_targets_loss(student_outputs, targets)


@criterion_registry("KnowledgeDistillationLoss", "pytorch")
class PyTorchKnowledgeDistillationLossWrapper(object):
    """PyTorchKnowledgeDistillationLossWrapper wraps PyTorchKnowledgeDistillationLoss."""

    def __init__(self, param_dict):
        """Initialize PyTorch Knowledge Distillation Loss Wrapper.

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
            any(isinstance(e, t) for t in [str, torch.nn.Module]) for e in param_dict["loss_types"]
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
        """Return PyTorchKnowledgeDistillationLoss, param dict.

        Returns:
            PyTorchKnowledgeDistillationLoss (class): PyTorchKnowledgeDistillationLoss
            param dict (dict): param dict
        """
        return PyTorchKnowledgeDistillationLoss, self._param_check()


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


class IntermediateLayersKnowledgeDistillationLoss(KnowledgeDistillationFramework):
    """The IntermediateLayersKnowledgeDistillationLoss class inherits from KnowledgeDistillationLoss."""

    def __init__(
        self,
        layer_mappings=[],
        loss_types=None,
        loss_weights=None,
        add_origin_loss=False,
        student_model=None,
        teacher_model=None,
    ):
        """Initialize PyTorch Knowledge Distillation Loss class.

        Args:
            temperature (float, optional): Hyperparameters that control the entropy
                                            of probability distributions. Defaults to 1.0.
            loss_types (list, optional): loss types. Defaults to ['CE', 'CE'].
            loss_weights (list, optional): loss weights. Defaults to [0.5, 0.5].
            student_model (torch.nn.model, optional): student model. Defaults to None.
            teacher_model (torch.nn.model, optional): teacher model. Defaults to None.

        Raises:
            NotImplementedError: NotImplementedError
            NotImplementedError: NotImplementedError
        """
        super(IntermediateLayersKnowledgeDistillationLoss, self).__init__(
            student_model=student_model, teacher_model=teacher_model
        )
        self.student_features = {}
        self.teacher_features = {}

        self.layer_mappings = []
        self.layer_output_process = []
        for item in layer_mappings:
            assert len(item) == 1 or len(item) == 2, (
                "Each item in layer_mappings "
                + "should be a list or tuple containing 1 list or 2 lists, with format "
                + "[(layer_name, )] or [(student_layer_name, ), (teacher_layer_name, )], "
                + "first one is the abbreviation for cases that student_layer_name and teacher_layer_name "
                + "are the same. The length of tuples in the list could be either 1 like previous cases, "
                + "or 2, like [(layer_name, layer_output_process)] or "
                + "[(student_layer_name, student_layer_output_process), "
                + "(teacher_layer_name, teacher_layer_output_process)]."
                + "For example, with 2 tuples of length 2, element looks like "
                + "[('student_model.layer1.attention', '1'), ('teacher_model.layer1.attention', '1')], "
                + "where 'student_model.layer1.attention' and 'teacher_model.layer1.attention' "
                + "represent attention module on layer 1 of the student model and the "
                + "teacher model respectively, two '1' represent the index to retrieve the "
                + "desired output from the defined module's outputs, in this case, the above "
                + "two module's outputs are lists, with desired output in index 1 of these "
                + "lists, in cases of dict output, retrieving can be done by defining the "
                + "corresponding key, in cases of module's output is the desired output, "
                + "just adopt the format such as [('student_model.layer1.output"
                + ".output', ), ('teacher_model.layer1.output', )]."
            )
            if len(item) == 1:
                item = [item[0], item[0]]
            for i in range(len(item)):
                if not isinstance(item[i], (list, tuple)):
                    item[i] = [item[i], ""]
                elif len(item[i]) == 1:
                    item[i] = [item[i][0], ""]
                else:
                    assert len(item[i]) == 2, "Expect {} to be a tuple of length 1 or 2.".format(item[i])
            self.layer_mappings.append((item[0][0], item[1][0]))
            self.layer_output_process.append((item[0][1], item[1][1]))
        for student_layer, teacher_layer in self.layer_mappings:
            self.student_features[student_layer] = []
            self.teacher_features[teacher_layer] = []

        self.loss_weights = (
            [1.0 / len(layer_mappings)] * len(layer_mappings)
            if (loss_weights is None or loss_weights == [])
            else loss_weights
        )
        self.loss_types = ["MSE"] * len(layer_mappings) if (loss_types is None or loss_types == []) else loss_types
        self.add_origin_loss = add_origin_loss
        self.loss_funcs = []
        self.feature_matchers = None
        self.init_loss_funcs()
        assert len(self.layer_mappings) == len(self.loss_weights) == len(self.loss_types), (
            f"Wrong length for layer_mappings:{self.layer_mappings}, "
            + f"loss_weights:{self.loss_weights} or loss_types:{self.loss_types}, "
            + "all should be the same."
        )

    def init_loss_funcs(self):
        """Init loss funcs.

        Raises:
            NotImplementedError: NotImplementedError
        """
        raise NotImplementedError("Function init_loss_funcs " "should be framework related.")

    def init_feature_matcher(self, student_feature, teacher_feature):
        """Init feature matcher.

        Raises:
            NotImplementedError: NotImplementedError
        """
        raise NotImplementedError("Function init_feature_matcher " "should be framework related.")

    def teacher_model_forward(self, input, teacher_model=None):
        """Teacher model forward.

        Raises:
            NotImplementedError: NotImplementedError
        """
        raise NotImplementedError("Function teacher_model_forward " "should be framework related.")

    def loss_cal(self):
        """Calculate loss.

        Raises:
            NotImplementedError: NotImplementedError
        """
        raise NotImplementedError("Function loss_cal should be framework related.")

    def loss_cal_sloss(self, student_outputs, teacher_outputs, student_loss):
        """Calculate all losses between student model and teacher model.

        Args:
            student_outputs (tensor): student outputs
            teacher_outputs (tensor): teacher outputs
            student_loss (tensor): student loss

        Returns:
            tensor: loss
        """
        return self.loss_cal()

    def clear_features(self):
        """Clean features in list."""
        for student_layer in self.student_features:
            self.student_features[student_layer] = []
        for teacher_layer in self.teacher_features:
            self.teacher_features[teacher_layer] = []

    def __call__(self, student_outputs, targets):
        """Return 0."""
        return 0


class PyTorchIntermediateLayersKnowledgeDistillationLoss(IntermediateLayersKnowledgeDistillationLoss):
    """PyTorch Intermediate Layers Knowledge Distillation Loss."""

    def __init__(
        self,
        layer_mappings=[],
        loss_types=None,
        loss_weights=None,
        add_origin_loss=False,
        student_model=None,
        teacher_model=None,
    ):
        """Initialize PyTorch Knowledge Distillation Loss class.

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
        super(PyTorchIntermediateLayersKnowledgeDistillationLoss, self).__init__(
            layer_mappings=layer_mappings,
            loss_types=loss_types,
            loss_weights=loss_weights,
            add_origin_loss=add_origin_loss,
            student_model=student_model,
            teacher_model=teacher_model,
        )
        self.register_hooks_for_models()

    def register_hooks_for_models(self):
        """Register hooks for models to record module output.

        Raises:
            AttributeError: AttributeError
        """
        from neural_compressor.compression.distillation import utility

        def register_model_forward_hook(model, path, output_process="", student=False):
            module = model
            if path != "":
                nodes = path.split(".")
                for node in nodes:
                    try:
                        module = module.__getattr__(node)
                    except:
                        raise AttributeError("There is no path {} in the model.".format(path))
            return module.register_forward_hook(utility.get_activation(path, output_process, student))

        assert isinstance(self.student_model, torch.nn.Module) and isinstance(self.teacher_model, torch.nn.Module), (
            "Expect student_model and teacher_model to be an torch.nn.Module object, "
            + "got student_model:{} and teacher_model:{}".format(type(self.student_model), type(self.teacher_model))
        )
        self.hook_handles = []
        for idx in range(len(self.layer_mappings)):
            student_layer, teacher_layer = self.layer_mappings[idx]
            student_output_process, teacher_output_process = self.layer_output_process[idx]
            st_handle = register_model_forward_hook(self.student_model, student_layer, student_output_process, True)
            te_handle = register_model_forward_hook(self.teacher_model, teacher_layer, teacher_output_process)
            utility.STUDENT_FEATURES = self.student_features
            utility.TEACHER_FEATURES = self.teacher_features
            self.hook_handles.extend([st_handle, te_handle])

    def remove_all_hooks(self):
        """Remove all hooks."""
        for hook in self.hook_handles:
            hook.remove()

    def init_loss_funcs(self):
        """Init loss funcs."""
        for loss_type in self.loss_types:
            if loss_type == "MSE":
                loss_func = torch.nn.MSELoss()
            elif loss_type == "KL":
                loss_func = torch.nn.KLDivLoss()
            elif loss_type == "L1":
                loss_func = torch.nn.L1Loss()
            else:
                raise NotImplementedError(
                    f"Unsupported loss type {loss_type}, supported loss is "
                    "MSE for mean squared error, KL for Kullback-Leibler divergence and "
                    "L1 for L1 loss."
                )
            self.loss_funcs.append(loss_func)

    def init_feature_matcher(self, student_feature, teacher_feature):
        """Init feature matcher.

        Args:
            student_feature (tensor): student feature
            teacher_feature (tensor): teacher feature

        Returns:
            pytorch_linear_feature_matcher
        """

        class pytorch_linear_feature_matcher(torch.nn.Module):
            def __init__(self, src_shape, dst_shape):
                super().__init__()
                shape_diff = [abs(i - j) for i, j in zip(dst_shape, src_shape)]
                assert shape_diff.count(0) == len(shape_diff) - 1, (
                    "Expect only one " + "different dimension between student_feature and teacher_feature."
                )
                self.dim_idx = np.argmax(shape_diff)
                self.dense = torch.nn.Linear(src_shape[self.dim_idx], dst_shape[self.dim_idx])

            def forward(self, input):
                output = torch.transpose(input, self.dim_idx, -1)
                if input.device != next(self.parameters()).device:
                    self.to(input.device)
                output = self.dense(output)
                output = torch.transpose(output, self.dim_idx, -1)
                return output

        assert isinstance(student_feature, (torch.Tensor, np.ndarray)) and isinstance(
            teacher_feature, (torch.Tensor, np.ndarray)
        ), (
            "Expect student_feature and teacher_feature to be torch.Tensor or np.ndarray "
            + "objects, got student_feature a {st} object, teacher_feature a {tt} object.".format(
                st=type(student_feature), tt=type(teacher_feature)
            )
        )
        assert len(student_feature.shape) == len(teacher_feature.shape), (
            "Expect student_feature and teacher_feature to have the same length of shape, "
            + "got student_feature of {}, teacher_feature of {}.".format(student_feature.shape, teacher_feature.shape)
        )
        if sum([abs(i - j) for i, j in zip(student_feature.shape, teacher_feature.shape)]) == 0:
            return lambda x: x
        return pytorch_linear_feature_matcher(student_feature.shape, teacher_feature.shape)

    def teacher_model_forward(self, input, teacher_model=None, device=None):
        """Define parameters for teacher_model_forward function.

        Args:
            input (tensor, tuple or dict): input data.
            teacher_model (model, optional): teacher model. Defaults to None.

        Raises:
            NotImplementedError: NotImplementedError
        """
        model = self.teacher_model if teacher_model is None else teacher_model
        assert isinstance(model, torch.nn.Module), "Teacher model should be a torch Module instead of {}".format(
            type(model)
        )
        model.eval()
        try:
            model_device = next(model.parameters()).device
        except:
            logger.warning("Cannot get model device, assuming it's in CPU.")
            model_device = "cpu"
        device = model_device if device is None else device
        if device != model_device:
            model.to(device)
        with torch.no_grad():
            outputs = pytorch_forward_wrapper(model, input)
        return outputs

    def loss_cal_sloss(self, student_outputs, teacher_outputs, student_loss):
        """Calculate all losses between student model and teacher model.

        Args:
            student_outputs (tensor): student outputs
            teacher_outputs (tensor): teacher outputs
            student_loss (tensor): student loss

        Returns:
            tensor: loss
        """
        loss = self.loss_cal()
        if self.add_origin_loss:
            loss += student_loss
        return loss

    def loss_cal(self):
        """Calculate loss of student model.

        Returns:
            tensor: loss
        """
        self.loss = 0
        init_feature_matchers = False
        if self.feature_matchers is None:
            init_feature_matchers = True
            self.feature_matchers = {}
        for idx in range(len(self.layer_mappings)):
            student_layer, teacher_layer = self.layer_mappings[idx]
            student_feature = self.student_features[student_layer]
            teacher_feature = self.teacher_features[teacher_layer]
            assert len(student_feature) == len(teacher_feature) and len(student_feature) > 0, (
                "Lengths of student_feature and teacher_feature should be the same and larger than 0, "
                + "instead of {} and {}, ".format(len(student_feature), len(teacher_feature))
                + "please run student and teacher model forward properly before calculating the loss."
            )

            def device2feature_gen(features):
                devices_count = Counter([f.device for f in features])
                assert [1] * len(devices_count) == [
                    _ for _ in devices_count.values()
                ], "Currently only support 1 feature tensor per device, " + "got {}.".format(devices_count)
                return {feat.device: feat for feat in features}

            student_feature = device2feature_gen(student_feature)
            teacher_feature = device2feature_gen(teacher_feature)
            assert student_feature.keys() == teacher_feature.keys(), (
                "Features from student model have different devices with that of "
                + "teacher model, got student: {}, teacher: {}.".format(student_feature.keys(), teacher_feature.keys())
            )
            output_device = (
                torch.device("cuda:0") if torch.device("cuda:0") in student_feature.keys() else torch.device("cpu")
            )
            if init_feature_matchers:
                feature_matcher = self.init_feature_matcher(
                    student_feature[output_device], teacher_feature[output_device]
                )
                self.feature_matchers[student_layer] = feature_matcher

            tmp_loss = 0
            for device in student_feature.keys():
                student_feature[device] = student_feature[device].to(output_device)
                teacher_feature[device] = teacher_feature[device].to(output_device)
                stfeat, tefeat = student_feature[device], teacher_feature[device]
                stfeat = self.feature_matchers[student_layer](stfeat)
                if self.loss_types[idx] == "KL":
                    check_is_not_prob = lambda x: (torch.abs(x.sum(dim=-1) - 1.0) > 0.2).any().item()
                    if isinstance(self.feature_matchers[student_layer], torch.nn.Module):
                        stfeat = torch.nn.LogSoftmax(dim=-1)(stfeat)
                    else:
                        if check_is_not_prob(stfeat):
                            stfeat = torch.softmax(stfeat, dim=-1)
                        stfeat = torch.log(stfeat + 1e-9)
                    if check_is_not_prob(tefeat):
                        tefeat = torch.softmax(tefeat, dim=-1)
                tmp_loss += self.loss_funcs[idx](stfeat, tefeat) * self.loss_weights[idx]
            self.loss += tmp_loss
        self.clear_features()
        return self.loss


@criterion_registry("IntermediateLayersKnowledgeDistillationLoss", "pytorch")
class PyTorchIntermediateLayersKnowledgeDistillationLossWrapper(object):
    """PyTorch Intermediate Layers Knowledge Distillation Loss Wrapper."""

    def __init__(self, param_dict):
        """Initialize PyTorchIntermediateLayersKnowledgeDistillationLossWrapper class.

        Args:
            param_dict (dict): param dict
        """
        self.param_dict = param_dict

    def _param_check(self):
        param_dict = self.param_dict
        _params = ["layer_mappings", "loss_types", "loss_weights", "add_origin_loss"]
        layer_mappings = param_dict["layer_mappings"]
        if "loss_types" not in param_dict or param_dict["loss_types"] == []:
            param_dict["loss_types"] = ["MSE"] * len(layer_mappings)
        if "loss_weights" not in param_dict or param_dict["loss_weights"] == []:
            param_dict["loss_weights"] = [1.0 / len(layer_mappings)] * len(layer_mappings)
        if "add_origin_loss" not in param_dict:
            param_dict["add_origin_loss"] = False
        assert "layer_mappings" in param_dict, "Key layer_mappings must be in input parameters."
        assert all(
            type(param_dict[k]) in [list, tuple] for k in ["layer_mappings", "loss_types", "loss_weights"]
        ), "Type of loss_types and loss_weights must be list or tuple."
        assert isinstance(param_dict["add_origin_loss"], bool), "Type of add_origin_loss should be bool."
        assert (
            len(param_dict["layer_mappings"]) == len(param_dict["loss_types"]) == len(param_dict["loss_weights"])
        ), "Length of layer_mappings, loss_types and loss_weights must be the same."
        assert all(
            type(it) in [list, tuple] and (len(it) == 1 or len(it) == 2) for it in param_dict["layer_mappings"]
        ), (
            "Each item in layer_mappings should be a list containing 1 tuple or 2 tuples, with format "
            + "[(layer_name, )] or [(student_layer_name, ), (teacher_layer_name, )], "
            + "first one is the abbreviation for cases that student_layer_name and teacher_layer_name "
            + "are the same. The length of tuples in the list could be either 1 like previous cases, "
            + "or 2, like [(layer_name, layer_output_process)] or "
            + "[(student_layer_name, student_layer_output_process), "
            + "(teacher_layer_name, teacher_layer_output_process)]."
            + "For example, with 2 tuples of length 2, element looks like "
            + "[('student_model.layer1.attention', '1'), ('teacher_model.layer1.attention', '1')], "
            + "where 'student_model.layer1.attention' and 'teacher_model.layer1.attention' "
            + "represent attention module on layer 1 of the student model and the "
            + "teacher model respectively, two '1' represent the index to retrieve the "
            + "desired output from the defined module's outputs, in this case, the above "
            + "two module's outputs are lists, with desired output in index 1 of these "
            + "lists, in cases of dict output, retrieving can be done by defining the "
            + "corresponding key, in cases of module's output is the desired output, "
            + "just adopt the format such as [('student_model.layer1.output"
            + ".output', ), ('teacher_model.layer1.output', )]."
        )
        assert all(
            any(isinstance(e, t) for t in [str, torch.nn.Module]) for e in param_dict["loss_types"]
        ), "Type of loss_types element must be str or torch Module."
        assert all(
            0.0 <= e <= 1.0 for e in param_dict["loss_weights"]
        ), "Element of loss_weights must be in interval [0, 1]."
        new_dict = {}
        for k in _params:
            new_dict[k] = param_dict[k]
        return new_dict

    def __call__(self, **kwargs):
        """Return PyTorchIntermediateLayersKnowledgeDistillationLossWrapper, param dict.

        Returns:
            class: PyTorchIntermediateLayersKnowledgeDistillationLoss
            param dict (dict): param dict
        """
        return PyTorchIntermediateLayersKnowledgeDistillationLoss, self._param_check()


class SelfKnowledgeDistillationLoss(KnowledgeDistillationFramework):
    """SelfKnowledge Distillation Loss."""

    def __init__(
        self,
        layer_mappings=[],
        loss_types=None,
        loss_weights=None,
        temperature=1.0,
        add_origin_loss=False,
        student_model=None,
        teacher_model=None,
    ):
        """Initialize SelfKnowledge Distillation Loss class.

        Args:
            layer_mappings (list): layers of distillation.Format like
                 [[[student1_layer_name1, teacher_layer_name1],[student2_layer_name1, teacher_layer_name1]],
                 [[student1_layer_name2, teacher_layer_name2],[student2_layer_name2, teacher_layer_name2]]]
            loss_types (list, optional): loss types. Defaults to ['CE'] * len(layer_mappings).
            loss_weights (list, optional): loss weights. Defaults to [1.0 / len(layer_mappings)] *
                len(layer_mappings).temperature (float, optional): use to calculate the soft label CE.
            temperature (optional): temperature. Defaults to 1.0.
            add_origin_loss (bool, optional): whether to add origin loss for hard label loss.
            student_model (optional): student model. Defaults to None.
            teacher_model (optional): teacher model. Defaults to None.
        """
        super(SelfKnowledgeDistillationLoss, self).__init__(student_model=student_model, teacher_model=teacher_model)
        self.temperature = temperature
        self.layer_mappings = []
        for items in layer_mappings:
            for value in items:
                assert len(value) == 2, (
                    "Each item in layer_mappings "
                    + "should be a list or tuple of length 2, with format "
                    + "[student_layer_name, teacher_layer_name]."
                )
            self.layer_mappings.append(items)

        self.loss_weights = (
            [1.0 / len(self.layer_mappings)] * len(self.layer_mappings) if loss_weights is None else loss_weights
        )
        self.loss_types = ["CE"] * len(self.layer_mappings) if loss_types is None else loss_types
        self.add_origin_loss = add_origin_loss
        self.loss_funcs = []
        self.init_loss_funcs()
        assert len(self.layer_mappings) == len(self.loss_weights) == len(self.loss_types), (
            f"Wrong length for layer_mappings:{self.layer_mappings}, "
            + f"loss_weights:{self.loss_weights} or loss_types:{self.loss_types}, "
            + "all should be the same."
        )

    def init_loss_funcs(self):
        """Init loss funcs.

        Raises:
            NotImplementedError: NotImplementedError
        """
        raise NotImplementedError("Function init_loss_funcs " "should be framework related.")

    def teacher_model_forward(self, input, teacher_model=None):
        """Teacher model forward.

        Raises:
            NotImplementedError: NotImplementedError
        """
        raise NotImplementedError("Function teacher_model_forward " "should be framework related.")

    def loss_cal(self, student_outputs):
        """Calculate loss.

        Raises:
            NotImplementedError: NotImplementedError
        """
        raise NotImplementedError("Function loss_cal should be framework related.")

    def loss_cal_sloss(self, student_outputs, teacher_outputs, student_loss):
        """Calculate all losses between student model and teacher model.

        Args:
            student_outputs (dict): student outputs
            teacher_outputs (dict): teacher outputs
            student_loss (tensor): student loss

        Returns:
            tensor: loss
        """
        loss = self.loss_cal(student_outputs)
        if self.add_origin_loss:
            loss += student_loss
        return loss

    def __call__(self, student_outputs, targets):
        """Return 0."""
        return 0


class PyTorchSelfKnowledgeDistillationLoss(SelfKnowledgeDistillationLoss):
    """PyTorch SelfKnowledge Distillation Loss."""

    def __init__(
        self,
        layer_mappings=[],
        loss_types=None,
        loss_weights=None,
        temperature=1.0,
        add_origin_loss=False,
        student_model=None,
        teacher_model=None,
    ):
        """Initialize PyTorch SelfKnowledge Distillation Loss class.

        Args:
            layer_mappings (list): layers of distillation.Format like
                 [[[student1_layer_name1, teacher_layer_name1],[student2_layer_name1, teacher_layer_name1]],
                 [[student1_layer_name2, teacher_layer_name2],[student2_layer_name2, teacher_layer_name2]]]
            loss_types (list, optional): loss types. Defaults to ['CE'] * len(layer_mappings).
            loss_weights (list, optional): loss weights. Defaults to [1.0 / len(layer_mappings)] *
                len(layer_mappings).temperature (float, optional): use to calculate the soft label CE.
            temperature (optional): temperature. Defaults to 1.0.
            add_origin_loss (bool, optional): whether to add origin loss for hard label loss.
            student_model (optional): student model. Defaults to None.
            teacher_model (optional): teacher model. Defaults to None.
        """
        super(PyTorchSelfKnowledgeDistillationLoss, self).__init__(
            layer_mappings=layer_mappings,
            loss_types=loss_types,
            loss_weights=loss_weights,
            temperature=temperature,
            add_origin_loss=add_origin_loss,
            student_model=student_model,
            teacher_model=teacher_model,
        )

    def SoftCrossEntropy(self, logits, targets):
        """Return SoftCrossEntropy.

        Args:
            logits (tensor): output logits
            targets (tensor): ground truth label

        Returns:
            tensor: SoftCrossEntropy
        """
        log_prob = torch.nn.functional.log_softmax(logits, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        return (-targets_prob * log_prob).sum(dim=-1).mean()

    def KullbackLeiblerDivergence(self, logits, targets):
        """Return KullbackLeiblerDivergence.

        Args:
            logits (tensor): output logits
            targets (tensor): ground truth label

        Returns:
            tensor: KullbackLeiblerDivergence
        """
        log_prob = torch.nn.functional.log_softmax(logits, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        return torch.nn.functional.kl_div(log_prob, targets_prob)

    def L2Divergence(self, feature1, feature2):
        """Return L2Divergence.

        Args:
            feature1 (tensor): feature1 value
            feature2 (tensor): feature2 value

        Returns:
            tensor: L2Divergence between feature1 and feature2
        """
        return torch.dist(feature1, feature2)

    def init_loss_funcs(self):
        """Init loss funcs."""
        for loss_type in self.loss_types:
            if loss_type == "CE":
                loss_func = self.SoftCrossEntropy
            elif loss_type == "KL":
                loss_func = self.KullbackLeiblerDivergence
            elif loss_type == "L2":
                loss_func = self.L2Divergence
            else:
                raise NotImplementedError(
                    f"Unsupported loss type {loss_type}, supported loss is"
                    " CE for software CE, KL for Kullback-Leibler divergence and"
                    " L2 for L2 distance."
                )
            self.loss_funcs.append(loss_func)

    def loss_cal(self, student_outputs):
        """Calculate loss of student model.

        Args:
            student_outputs (dict): student outputs

        Returns:
            tensor: loss
        """
        self.loss = torch.FloatTensor([0.0])
        tmp_loss = 0
        temperature = self.temperature
        for loss_idx in range(len(self.layer_mappings)):
            items = self.layer_mappings[loss_idx]
            for idx in range(len(items)):
                student_layer, teacher_layer = items[idx]
                student_feature = student_outputs[student_layer]
                teacher_feature = student_outputs[teacher_layer]
                if loss_idx == 1:  # soft logit
                    tmp_loss += (
                        self.loss_funcs[loss_idx](student_feature / temperature, teacher_feature / temperature)
                        * self.loss_weights[loss_idx]
                    )
                else:  # feature learning
                    tmp_loss += (
                        self.loss_funcs[loss_idx](student_feature, teacher_feature) * self.loss_weights[loss_idx]
                    )
            if tmp_loss.device != self.loss.device:
                self.loss = self.loss.to(tmp_loss.device)
            self.loss += tmp_loss
        return self.loss

    def teacher_model_forward(self, input, teacher_model=None, device=None):
        """Teacher model forward.

        Args:
            input (tensor): input data
            teacher_model (torch.nn.model, optional): teacher model. Defaults to None.
            device (torch.device, optional): device. Defaults to None.

        Returns:
            tensor: output
        """
        outputs = None
        if self.loss_weights[1] > 0:
            model = self.teacher_model if teacher_model is None else teacher_model
            assert isinstance(model, torch.nn.Module), "Teacher model should be a torch Module instead of {}".format(
                type(model)
            )
            model.eval()
            try:
                model_device = next(model.parameters()).device
            except:
                logger.warning("Cannot get model device, assuming it's in CPU.")
                model_device = "cpu"
            device = model_device if device is None else device
            if device != model_device:
                model.to(device)
            with torch.no_grad():
                outputs = pytorch_forward_wrapper(model, input)
            self.teacher_outputs = outputs
        return outputs


@criterion_registry("SelfKnowledgeDistillationLoss", "pytorch")
class PyTorchSelfKnowledgeDistillationLossWrapper(object):
    """PyTorch SelfKnowledge Distillation Loss Wrapper."""

    def __init__(self, param_dict):
        """Initialize PyTorchSelfKnowledgeDistillationLossWrapper class.

        Args:
            param_dict (dict): param dict
        """
        self.param_dict = param_dict

    def _param_check(self):
        param_dict = self.param_dict
        _params = ["temperature", "layer_mappings", "loss_types", "loss_weights", "add_origin_loss"]
        layer_mappings = param_dict["layer_mappings"]
        if "loss_types" not in param_dict:
            param_dict["loss_types"] = ["CE"] * len(layer_mappings)
        if "loss_weights" not in param_dict:
            param_dict["loss_weights"] = [1.0 / len(layer_mappings)] * len(layer_mappings)
        if "add_origin_loss" not in param_dict:
            param_dict["add_origin_loss"] = False
        if "temperature" not in param_dict:
            param_dict["temperature"] = 1.0
        assert "layer_mappings" in param_dict, "Key layer_mappings must be in input parameters."
        assert all(
            type(param_dict[k]) in [list, tuple] for k in ["layer_mappings", "loss_types", "loss_weights"]
        ), "Type of loss_types and loss_weights must be list or tuple."
        assert isinstance(param_dict["add_origin_loss"], bool), "Type of add_origin_loss should be bool."
        assert (
            len(param_dict["layer_mappings"]) == len(param_dict["loss_types"]) == len(param_dict["loss_weights"])
        ), "Length of layer_mappings, loss_types and loss_weights must be the same."
        assert param_dict["temperature"] > 0.0, "Value of temperature must be positive."
        for items in param_dict["layer_mappings"]:
            assert all(type(it) in [list, tuple] and (len(it) == 2) for it in items), (
                "Elements of layer_mappings must be list or tuple and with length of 2."
                + "element looks like ['resblock.1.feature.output,"
                + "'resblock.deepst.feature.output'], where "
                + "'resblock.1.feature.output' and 'resblock.deepst.feature.output' "
                + "represent resblock feature output of the student model and feature output of the"
                + "teacher model respectively."
            )
        assert all(
            any(isinstance(e, t) for t in [str]) for e in param_dict["loss_types"]
        ), "Type of loss_types element must be str."
        assert all(
            0.0 <= e <= 1.0 for e in param_dict["loss_weights"]
        ), "Element of loss_weights must be in interval [0, 1]."
        new_dict = {}
        for k in _params:
            new_dict[k] = param_dict[k]
        return new_dict

    def __call__(self, **kwargs):
        """Return PyTorchSelfKnowledgeDistillationLoss, param dict.

        Returns:
            class: PyTorchSelfKnowledgeDistillationLoss
            param dict (dict): param dict
        """
        return PyTorchSelfKnowledgeDistillationLoss, self._param_check()
