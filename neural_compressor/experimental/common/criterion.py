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
from collections import UserDict
from neural_compressor.utils.utility import LazyImport, singleton

torch = LazyImport('torch')
tf = LazyImport('tensorflow')

@singleton
class TensorflowCriterions(object):
    def __init__(self):
        self.criterions = {}
        self.criterions.update(TENSORFLOW_CRITERIONS)

@singleton
class PyTorchCriterions(object):
    def __init__(self):
        self.criterions = {}
        self.criterions.update(PYTORCH_CRITERIONS)

framework_criterions = {"tensorflow": TensorflowCriterions,
                        "pytorch":    PyTorchCriterions,
                        "pytorch_fx": PyTorchCriterions}

# user/model specific criterions will be registered here
TENSORFLOW_CRITERIONS = {}
PYTORCH_CRITERIONS= {}

registry_criterions = {"tensorflow": TENSORFLOW_CRITERIONS,
                       "pytorch":    PYTORCH_CRITERIONS,
                       "pytorch_fx": PYTORCH_CRITERIONS}

class Criterions(object):
    def __init__(self, framework):
        assert framework in ("tensorflow", "pytorch", "pytorch_fx"), \
                             "framework support tensorflow pytorch"
        self.criterions = framework_criterions[framework]().criterions

    def __getitem__(self, criterion_type):
        assert criterion_type in self.criterions.keys(), "only support criterions in {}".\
            format(self.criterions.keys())

        return self.criterions[criterion_type]

    def register(self, name, criterion_cls):
        assert name not in self.criterions.keys(), 'registered criterion name already exists.'
        self.criterions.update({name: criterion_cls})

def criterion_registry(criterion_type, framework):
    """The class decorator used to register all criterion classes.

    Args:
        criterion_type (str): The string of supported criterion.
        framework (str): The string of supported framework. 

    Returns:
        cls: The class of register.
    """
    def decorator_criterion(cls):
        for fw in [fwk.strip() for fwk in framework.split(',')]:
            assert fw in [
                "tensorflow",
                "pytorch"], "The framework support tensorflow pytorch"

            if criterion_type in registry_criterions[fw].keys():
                raise ValueError('Cannot have two criterions with the same name')
            registry_criterions[fw][criterion_type] = cls
        return cls
    return decorator_criterion

@criterion_registry('CrossEntropyLoss', 'tensorflow')
class TensorFlowCrossEntropyLoss(object):
    """TensorFlow CrossEntropyLoss criterion.

    Args:
        param_dict (dict): The dict of parameters setting by user for CrossEntropyLoss criterion.
    """
    def __init__(self, param_dict):
        assert isinstance(param_dict, dict), 'This criterion constructor parameter must be a dict'
        self._param_dict = param_dict

    def _mapping(self):
        _param_map = {'reduction': 'reduction'}
        _dict = {}
        for key in self._param_dict:
            if key in _param_map:
                if key == 'reduction':
                    assert self._param_dict[key] in ['none', 'mean', 'sum'], \
                        'Supported reduction value is none, mean, sum'
                _dict.update({_param_map[key]: self._param_dict[key]})
        return _dict

    def __call__(self, **kwargs):
        return tf.keras.losses.CategoricalCrossentropy, self._mapping(**kwargs)

@criterion_registry('CrossEntropyLoss', 'pytorch')
class PyTorchCrossEntropyLoss(object):
    """PyTorch CrossEntropyLoss criterion.

    Args:
        param_dict (dict): The dict of parameters setting by user for SGD criterion
    """
    def __init__(self, param_dict):
        assert isinstance(param_dict, dict), 'This criterion constructor parameter must be a dict'
        self._param_dict = param_dict

    def _mapping(self):
        _param_map = {'reduction': 'reduction'}
        _dict = {}
        for key in self._param_dict:
            if key in _param_map:
                if key == 'reduction':
                    assert self._param_dict[key] in ['none', 'mean', 'sum'], \
                        'Supported reduction value is none, mean, sum'
                _dict.update({_param_map[key]: self._param_dict[key]})
        return _dict

    def __call__(self, **kwargs):
        return torch.nn.CrossEntropyLoss, self._mapping(**kwargs)

class KnowledgeDistillationLoss(object):
    def __init__(self, temperature=1.0, loss_types=['CE', 'CE'], 
                 loss_weights=[0.5, 0.5]):
        self._teacher_model = None
        self.teacher_outputs = None
        self.temperature = temperature
        self.loss_weights = loss_weights
        self.loss_types = loss_types
        self.teacher_student_loss = self.student_targets_loss = None
        assert len(loss_weights) == len(loss_types) == 2, 'Wrong length for ' + \
                                    'loss_weights or loss_types, should be 2.'
        assert sum(loss_weights) == 1.0, 'Sum of loss_weights should be 1.0.'

    @property
    def teacher_model(self):
        """ Getter of teacher model """
        return self._teacher_model

    @teacher_model.setter
    def teacher_model(self, model):
        """ Setter of teacher model """
        self._teacher_model = model

    def teacher_model_forward(self, input, teacher_model=None):
        raise NotImplementedError('Function teacher_model_forward '
                                  'should be framework related.')
    
    def teacher_student_loss_cal(self, student_outputs, teacher_outputs):
        raise NotImplementedError('Function teacher_student_loss_cal '
                                  'should be framework related.')

    def student_targets_loss_cal(self, student_outputs, targets):
        raise NotImplementedError('Function student_targets_loss_cal '
                                  'should be framework related.')

    def loss_cal(self, student_outputs, targets):
        if self.loss_weights[0] > 0:
            origin_loss = self.student_targets_loss_cal(student_outputs, targets)
        else:
            origin_loss = 0

        if self.loss_weights[1] > 0:
            assert self.teacher_outputs is not None, 'Output of teacher model is required, ' \
                'please run teacher_model_forward to get output of teacher model first.'
            student_out_ = student_outputs / self.temperature
            teacher_out_ = self.teacher_outputs / self.temperature
            distillation_loss = self.teacher_student_loss_cal(student_out_, teacher_out_)
            distillation_loss *=  self.temperature ** 2
        else:
            distillation_loss = 0

        self.loss = origin_loss * self.loss_weights[0] + \
                    distillation_loss * self.loss_weights[1]
        return self.loss

    def __call__(self, student_outputs, targets):
        return self.loss_cal(student_outputs, targets)

class PyTorchKnowledgeDistillationLoss(KnowledgeDistillationLoss):
    def __init__(self, temperature=1.0, loss_types=['CE', 'CE'], 
                 loss_weights=[0.5, 0.5]):
        super(PyTorchKnowledgeDistillationLoss, self).__init__(temperature=temperature, 
                                                               loss_types=loss_types,
                                                               loss_weights=loss_weights)
        if self.student_targets_loss is None:
            if self.loss_types[0] == 'CE':
                self.student_targets_loss = torch.nn.CrossEntropyLoss()
            else:
                raise NotImplementedError('Now we only support CrossEntropyLoss '
                 'for loss of student model output with respect to targets.')
            print('student_targets_loss: {}, {}'.format(self.loss_types[0], \
                                                        self.loss_weights[0]))
        if self.teacher_student_loss is None:
            if self.loss_types[1] == 'CE':
                self.teacher_student_loss = self.SoftCrossEntropy
            elif self.loss_types[1] == 'KL':
                self.teacher_student_loss = self.KullbackLeiblerDivergence
            else:
                raise NotImplementedError('Now we only support CrossEntropyLoss'
                ' for loss of student model output with respect to teacher model ouput.')
            print('teacher_student_loss: {}, {}'.format(self.loss_types[1], \
                                                        self.loss_weights[1]))

    def SoftCrossEntropy(self, logits, targets):
        log_prob = torch.nn.functional.log_softmax(logits, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        return (- targets_prob * log_prob).sum(dim=-1).mean()

    def KullbackLeiblerDivergence(self, logits, targets):
        log_prob = torch.nn.functional.log_softmax(logits, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        return torch.nn.functional.kl_div(log_prob, targets_prob)
    
    def teacher_model_forward(self, input, teacher_model=None):
        if self.loss_weights[1] > 0:
            model = self.teacher_model if teacher_model is None else teacher_model
            assert isinstance(model, torch.nn.Module), \
            'Teacher model should be a torch Module instead of {}'.format(type(model))
            model.eval()
            with torch.no_grad():
                if isinstance(input, dict) or isinstance(input, UserDict):
                    outputs = model(**input)
                elif isinstance(input, list) or isinstance(input, tuple):
                    outputs = model(*input)
                else:
                    outputs = model(input)
                self.teacher_outputs = outputs
        
    def teacher_student_loss_cal(self, student_outputs, teacher_outputs):
        assert self.teacher_student_loss, 'teacher_student_loss not specified.'
        return self.teacher_student_loss(student_outputs, teacher_outputs)
    
    def student_targets_loss_cal(self, student_outputs, targets):
        assert self.student_targets_loss, 'student_targets_loss not specified.'
        return self.student_targets_loss(student_outputs, targets)

@criterion_registry('KnowledgeDistillationLoss', 'pytorch')
class PyTorchKnowledgeDistillationLossWrapper(object):
    def __init__(self, param_dict):
        self.param_dict = param_dict

    def _param_check(self):
        param_dict = self.param_dict
        _params = ['temperature', 'loss_types', 'loss_weights']
        assert all(key in param_dict for key in _params),\
            'Keys {} must be in input parameters.'.format(_params)
        assert param_dict['temperature'] > 0.0,\
            'Value of temperature must be positive.'
        assert len(param_dict['loss_types']) == len(param_dict['loss_weights']),\
            'Length of loss_types and loss_weights must be positive.'
        assert all(type(param_dict[k]) in [list, tuple] \
            for k in ['loss_types', 'loss_weights']),\
            'Type of loss_types and loss_weights must be list or tuple.'
        assert all(any(isinstance(e, t) for t in [str, torch.nn.Module]) \
            for e in param_dict['loss_types']), \
            'Type of loss_types element must be str or torch Module.'
        assert all(0. <= e <= 1. for e in param_dict['loss_weights']) and \
            sum(param_dict['loss_weights']) == 1.0, \
            'Element of loss_weights must be in interval [0, 1] and summed to 1.0.'
        new_dict = {}
        for k in _params:
            new_dict[k] = param_dict[k]
        return new_dict
    
    def __call__(self, **kwargs):
        return PyTorchKnowledgeDistillationLoss, self._param_check()

@criterion_registry('KnowledgeDistillationLoss', 'tensorflow')
class TensorflowKnowledgeDistillationLoss(KnowledgeDistillationLoss):
    def __init__(self, temperature=1.0, loss_types=['CE', 'CE'], 
                 loss_weights=[0.5, 0.5]):
        super(TensorflowKnowledgeDistillationLoss, self).__init__(temperature=temperature, 
                                                                  loss_types=loss_types,
                                                                  loss_weights=loss_weights)
        if self.student_targets_loss is None:
            if self.loss_types[0] == 'CE':
                self.student_targets_loss = tf.keras.losses.CategoricalCrossentropy()
            else:
                raise NotImplementedError('Now we only support CrossEntropyLoss '
                 'for loss of student model output with respect to targets.')
        if self.teacher_student_loss is None:
            if self.loss_types[1] == 'CE':
                self.teacher_student_loss = tf.keras.losses.CategoricalCrossentropy()
            else:
                raise NotImplementedError('Now we only support CrossEntropyLoss'
                ' for loss of student model output with respect to teacher model ouput.')
    
    def teacher_model_forward(self, input, teacher_model=None):
        pass
    
    def teacher_student_loss_cal(self, student_outputs, teacher_outputs):
        assert self.teacher_student_loss, 'teacher_student_loss not specified.'
        return self.teacher_student_loss(student_outputs, teacher_outputs)
    
    def student_targets_loss_cal(self, student_outputs, targets):
        assert self.student_targets_loss, 'student_targets_loss not specified.'
        return self.teacher_student_loss(student_outputs, targets)           
   