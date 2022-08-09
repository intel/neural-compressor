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
from collections import UserDict, Counter
from neural_compressor.utils.utility import LazyImport, singleton
from neural_compressor.utils import logger
from neural_compressor.adaptor.pytorch import pytorch_forward_wrapper

import numpy as np

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
        _param_map = {'reduction': 'reduction',
                      'from_logits':'from_logits'}
        _dict = {}
        for key in self._param_dict:
            if key in _param_map:
                if key == 'reduction':
                    assert self._param_dict[key] in ['auto', 'none', 'sum', 'sum_over_batch_size'], \
                        'Supported reduction value for tensorflow is auto, none, sum, sum_over_batch_size'
                _dict.update({_param_map[key]: self._param_dict[key]})
        return _dict

    def __call__(self, **kwargs):
        return tf.keras.losses.CategoricalCrossentropy, self._mapping(**kwargs)

@criterion_registry('SparseCategoricalCrossentropy', 'tensorflow')
class TensorFlowSparseCategoricalCrossentropy(object):
    """TensorFlow SparseCategoricalCrossentropyLoss criterion.

    Args:
        param_dict (dict): The dict of parameters setting by user for SparseCategoricalCrossentropy criterion.
    """
    def __init__(self, param_dict):
        assert isinstance(param_dict, dict), 'This criterion constructor parameter must be a dict'
        self._param_dict = param_dict

    def _mapping(self):
        _param_map = {'reduction': 'reduction',
                      'from_logits':'from_logits'}
        _dict = {}
        for key in self._param_dict:
            if key in _param_map:
                if key == 'reduction':
                    assert self._param_dict[key] in ['auto', 'none', 'sum', 'sum_over_batch_size'], \
                        'Supported reduction value for tensorflow is auto, none, sum, sum_over_batch_size'
                _dict.update({_param_map[key]: self._param_dict[key]})
        return _dict

    def __call__(self, **kwargs):
        return tf.keras.losses.SparseCategoricalCrossentropy, self._mapping(**kwargs)

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

class KnowledgeDistillationFramework(object):
    def __init__(self, student_model=None, teacher_model=None):
        self._student_model = student_model
        self._teacher_model = teacher_model

    @property
    def student_model(self):
        """ Getter of student model """
        return self._student_model

    @student_model.setter
    def student_model(self, model):
        """ Setter of teacher model """
        self._student_model = model

    @property
    def teacher_model(self):
        """ Getter of teacher model """
        return self._teacher_model

    @teacher_model.setter
    def teacher_model(self, model):
        """ Setter of teacher model """
        self._teacher_model = model

class KnowledgeDistillationLoss(KnowledgeDistillationFramework):
    def __init__(self, temperature=1.0, loss_types=['CE', 'CE'], 
                 loss_weights=[0.5, 0.5], student_model=None, teacher_model=None):
        super(KnowledgeDistillationLoss, self).__init__(student_model=student_model,
                                                        teacher_model=teacher_model)
        self.teacher_outputs = None
        self.temperature = temperature
        self.loss_weights = loss_weights
        self.loss_types = loss_types
        self.teacher_student_loss = self.student_targets_loss = None
        assert len(loss_weights) == len(loss_types) == 2, 'Wrong length for ' + \
                                    'loss_weights or loss_types, should be 2.'
        assert sum(loss_weights) == 1.0, 'Sum of loss_weights should be 1.0.'

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
        return self.student_targets_loss_cal(student_outputs, targets)

    def loss_cal_sloss(self, student_outputs, teacher_outputs, student_loss):
        if self.loss_weights[0] > 0:
            origin_loss = student_loss
        else:
            origin_loss = 0

        if self.loss_weights[1] > 0:
            student_out_ = student_outputs / self.temperature
            teacher_out_ = teacher_outputs / self.temperature
            distillation_loss = self.teacher_student_loss_cal(student_out_, teacher_out_)
            distillation_loss *= self.temperature ** 2
        else:
            distillation_loss = 0

        self.loss = origin_loss * self.loss_weights[0] + \
                    distillation_loss * self.loss_weights[1]
        return self.loss

    def __call__(self, student_outputs, targets):
        return self.loss_cal(student_outputs, targets)

class PyTorchKnowledgeDistillationLoss(KnowledgeDistillationLoss):
    def __init__(self, temperature=1.0, loss_types=['CE', 'CE'], 
                 loss_weights=[0.5, 0.5], student_model=None, teacher_model=None):
        super(PyTorchKnowledgeDistillationLoss, self).__init__(temperature=temperature, 
                                                               loss_types=loss_types,
                                                               loss_weights=loss_weights,
                                                               student_model=student_model,
                                                               teacher_model=teacher_model)
        if self.student_targets_loss is None:
            if self.loss_types[0] == 'CE':
                self.student_targets_loss = torch.nn.CrossEntropyLoss()
            else:
                raise NotImplementedError('Now we only support CrossEntropyLoss '
                 'for loss of student model output with respect to targets.')
            logger.info('student_targets_loss: {}, {}'.format(self.loss_types[0], \
                                                        self.loss_weights[0]))
        if self.teacher_student_loss is None:
            if self.loss_types[1] == 'CE':
                self.teacher_student_loss = self.SoftCrossEntropy
            elif self.loss_types[1] == 'KL':
                self.teacher_student_loss = self.KullbackLeiblerDivergence
            else:
                raise NotImplementedError('Now we only support CrossEntropyLoss'
                ' for loss of student model output with respect to teacher model ouput.')
            logger.info('teacher_student_loss: {}, {}'.format(self.loss_types[1], \
                                                        self.loss_weights[1]))

    def SoftCrossEntropy(self, logits, targets):
        log_prob = torch.nn.functional.log_softmax(logits, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        return (- targets_prob * log_prob).sum(dim=-1).mean()

    def KullbackLeiblerDivergence(self, logits, targets):
        log_prob = torch.nn.functional.log_softmax(logits, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        return torch.nn.functional.kl_div(log_prob, targets_prob)

    def teacher_model_forward(self, input, teacher_model=None, device=None):
        outputs = None
        if self.loss_weights[1] > 0:
            model = self.teacher_model if teacher_model is None else teacher_model
            assert isinstance(model, torch.nn.Module), \
                'Teacher model should be a torch Module instead of {}'.format(type(model))
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
                outputs = pytorch_forward_wrapper(model, input, device=device)
            self.teacher_outputs = outputs
        return outputs

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
            'Length of loss_types and loss_weights must be the same.'
        assert all(type(param_dict[k]) in [list, tuple] \
            for k in ['loss_types', 'loss_weights']),\
            'Type of loss_types and loss_weights must be list or tuple.'
        assert all(any(isinstance(e, t) for t in [str, torch.nn.Module]) \
            for e in param_dict['loss_types']), \
            'Type of loss_types element must be str or torch Module.'
        assert all(0. <= e <= 1. for e in param_dict['loss_weights']) and \
            abs(sum(param_dict['loss_weights']) - 1.0) < 1e-9, \
            'Element of loss_weights must be in interval [0, 1] and summed to 1.0.'
        new_dict = {}
        for k in _params:
            new_dict[k] = param_dict[k]
        return new_dict

    def __call__(self, **kwargs):
        return PyTorchKnowledgeDistillationLoss, self._param_check()

class TensorflowKnowledgeDistillationLoss(KnowledgeDistillationLoss):
    def __init__(self, temperature=1.0, loss_types=['CE', 'CE'], 
                 loss_weights=[0.5, 0.5], student_model=None, teacher_model=None):
        super(TensorflowKnowledgeDistillationLoss, self).__init__(temperature=temperature, 
                                                                  loss_types=loss_types,
                                                                  loss_weights=loss_weights,
                                                                  student_model=student_model,
                                                                  teacher_model=teacher_model)
        if self.student_targets_loss is None:
            if self.loss_types[0] == 'CE':
                self.student_targets_loss = tf.keras.losses.SparseCategoricalCrossentropy()
            else:
                raise NotImplementedError('Now we only support CrossEntropyLoss '
                 'for loss of student model output with respect to targets.')
            logger.info('student_targets_loss: {}, {}'.format(self.loss_types[0], \
                                                        self.loss_weights[0]))            
        if self.teacher_student_loss is None:
            if self.loss_types[1] == 'CE':
                self.teacher_student_loss = self.SoftCrossEntropy
            elif self.loss_types[1] == 'KL':
                self.teacher_student_loss = tf.keras.losses.KLDivergence()
            else:
                raise NotImplementedError('Now we only support CrossEntropyLoss'
                ' for loss of student model output with respect to teacher model ouput.')
            logger.info('teacher_student_loss: {}, {}'.format(self.loss_types[1], \
                                                        self.loss_weights[1]))
    def SoftCrossEntropy(self, targets, logits):
        log_prob = tf.math.log(logits)
        targets_prob = targets
        return tf.math.reduce_mean(tf.math.reduce_sum(- targets_prob * log_prob, axis=-1), axis=-1)

    def teacher_model_forward(self, input, teacher_model=None):
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
        assert self.teacher_student_loss, 'teacher_student_loss not specified.'
        return self.teacher_student_loss(teacher_outputs, student_outputs)

    def student_targets_loss_cal(self, student_outputs, targets):
        assert self.student_targets_loss, 'student_targets_loss not specified.'
        return self.student_targets_loss(targets, student_outputs)

    def __call__(self, student_outputs, targets):
        tmp = student_outputs
        student_outputs = targets
        targets = tmp
        return self.loss_cal(student_outputs, targets)

@criterion_registry('KnowledgeDistillationLoss', 'tensorflow')
class TensorflowKnowledgeDistillationLossWrapper(object):
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
            'Length of loss_types and loss_weights must be the same.'
        assert all(type(param_dict[k]) in [list, tuple] \
            for k in ['loss_types', 'loss_weights']),\
            'Type of loss_types and loss_weights must be list or tuple.'
        assert all(any(isinstance(e, t) for t in [str, tf.keras]) \
            for e in param_dict['loss_types']), \
            'Type of loss_types element must be str or torch Module.'
        assert all(0. <= e <= 1. for e in param_dict['loss_weights']) and \
            abs(sum(param_dict['loss_weights']) - 1.0) < 1e-9, \
            'Element of loss_weights must be in interval [0, 1] and summed to 1.0.'
        new_dict = {}
        for k in _params:
            new_dict[k] = param_dict[k]
        return new_dict

    def __call__(self, **kwargs):
        return TensorflowKnowledgeDistillationLoss, self._param_check()

class TensorflowKnowledgeDistillationLossExternal(KnowledgeDistillationLoss):
    def __init__(self, temperature=1.0, loss_types=['CE', 'CE'], 
                 loss_weights=[0.5, 0.5], student_model=None, teacher_model=None):
        super(TensorflowKnowledgeDistillationLossExternal, self).__init__(
                                                                temperature=temperature, 
                                                                loss_types=loss_types,
                                                                loss_weights=loss_weights,
                                                                student_model=student_model,
                                                                teacher_model=teacher_model)
        if self.student_targets_loss is None:
            if self.loss_types[0] == 'CE':
                self.student_targets_loss = tf.keras.losses.CategoricalCrossentropy()
            else:
                raise NotImplementedError('Now we only support CrossEntropyLoss '
                 'for loss of student model output with respect to targets.')
            logger.info('student_targets_loss: {}, {}'.format(self.loss_types[0], \
                                                        self.loss_weights[0]))            
        if self.teacher_student_loss is None:
            if self.loss_types[1] == 'CE':
                self.teacher_student_loss = tf.keras.losses.CategoricalCrossentropy()
            elif self.loss_types[1] == 'KL':
                self.teacher_student_loss = tf.keras.losses.KLDivergence()
            else:
                raise NotImplementedError('Now we only support CrossEntropyLoss'
                ' for loss of student model output with respect to teacher model ouput.')
            logger.info('teacher_student_loss: {}, {}'.format(self.loss_types[1], \
                                                        self.loss_weights[1]))

    def teacher_model_forward(self, input, teacher_model=None):
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
        assert self.teacher_student_loss, 'teacher_student_loss not specified.'
        return self.teacher_student_loss(teacher_outputs, student_outputs)

    def student_targets_loss_cal(self, student_outputs, targets):
        assert self.student_targets_loss, 'student_targets_loss not specified.'
        return self.student_targets_loss(targets, student_outputs)


class IntermediateLayersKnowledgeDistillationLoss(KnowledgeDistillationFramework):
    def __init__(self, layer_mappings=[], loss_types=None, loss_weights=None, 
                 add_origin_loss=False, student_model=None, teacher_model=None):
        super(IntermediateLayersKnowledgeDistillationLoss, self).__init__(
            student_model=student_model,
            teacher_model=teacher_model
            )
        self.student_features = {}
        self.teacher_features = {}

        self.layer_mappings = []
        self.layer_output_process = []
        for item in layer_mappings:
            assert len(item) == 4 or len(item) == 2, 'Each item in layer_mappings ' + \
                'should be a list or tuple of length 2 or 4, with format ' + \
                '[student_layer_name, teacher_layer_name] or ' + \
                '[student_layer_name, student_layer_output_process, ' + \
                'teacher_layer_name, teacher_layer_output_process].' + \
                'For example, with length of 4, element looks like [\'student_model.layer1' + \
                '.attention\', \'1\', \'teacher_model.layer1.attention\', \'1\'], where ' + \
                '\'student_model.layer1.attention\' and \'teacher_model.layer1.attention\' ' + \
                'represent attention module on layer 1 of the student model and the ' + \
                'teacher model respectively, two \'1\' represent the index to retrieve the ' + \
                'desired output from the defined module\'s outputs, in this case, the above ' + \
                'two module\'s outputs are lists, with desired output in index 1 of these ' + \
                'lists, in cases of dict output, retrieving can be done by define the ' + \
                'corresponding key, in cases of module\'s output is the desired output, ' + \
                'just adopt an element of length 2 such as [\'student_model.layer1.output' + \
                '.output\', \'teacher_model.layer1.output\'].'
            self.layer_mappings.append((item[0], item[2]) if len(item)==4 else item)
            self.layer_output_process.append((item[1], item[3]) if len(item)==4 else ('', ''))
        for student_layer, teacher_layer in self.layer_mappings:
            self.student_features[student_layer] = []
            self.teacher_features[teacher_layer] = []

        self.loss_weights = [1.0 / len(layer_mappings)] * len(layer_mappings) \
                            if loss_weights is None else loss_weights
        self.loss_types = ['MSE'] * len(layer_mappings) \
                          if loss_types is None else loss_types
        self.add_origin_loss = add_origin_loss
        self.loss_funcs = []
        self.feature_matchers = None
        self.init_loss_funcs()
        assert len(self.layer_mappings) == len(self.loss_weights) == len(self.loss_types), \
            'Wrong length for layer_mappings:{}, loss_weights:{} or loss_types:{}, ' + \
            'all should be the same.'.format(
                len(self.layer_mappings), len(self.loss_weights), len(self.loss_types)
                )

    def init_loss_funcs(self):
        raise NotImplementedError('Function init_loss_funcs '
                                  'should be framework related.')

    def init_feature_matcher(self, student_feature, teacher_feature):
        raise NotImplementedError('Function init_feature_matcher '
                                  'should be framework related.')

    def teacher_model_forward(self, input, teacher_model=None):
        raise NotImplementedError('Function teacher_model_forward '
                                  'should be framework related.')

    def loss_cal(self):
        raise NotImplementedError('Function loss_cal should be framework related.')

    def loss_cal_sloss(self, student_outputs, teacher_outputs, student_loss):
        return self.loss_cal()

    def clear_features(self):
        for student_layer in self.student_features:
            self.student_features[student_layer] = []
        for teacher_layer in self.teacher_features:
            self.teacher_features[teacher_layer] = []

    def __call__(self, student_outputs, targets):
        return 0


class PyTorchIntermediateLayersKnowledgeDistillationLoss(
                IntermediateLayersKnowledgeDistillationLoss
                ):
    def __init__(self, layer_mappings=[], loss_types=None, loss_weights=None, 
                 add_origin_loss=False, student_model=None, teacher_model=None):
        super(PyTorchIntermediateLayersKnowledgeDistillationLoss, self).__init__(
                                                layer_mappings=layer_mappings, 
                                                loss_types=loss_types,
                                                loss_weights=loss_weights,
                                                add_origin_loss=add_origin_loss, 
                                                student_model=student_model,
                                                teacher_model=teacher_model)
        self.register_hooks_for_models()

    def register_hooks_for_models(self):
        def get_activation(name, model_activation, output_process=''):
            def hook(model, input, output):
                if output_process != '':
                    if isinstance(output, dict) and output_process in output:
                        output = output[output_process]
                    elif isinstance(output, (tuple, list)) and str.isnumeric(output_process):
                        output = output[int(output_process)]
                    elif callable(output_process):
                        output = output_process(output)
                    else:
                        raise NotImplementedError('Current only support get the data with ' + \
                            'integer index in case the output is tuple or list and only ' + \
                            'need one item or with key in case the output is dict,  ' + \
                            'or output_process is a function.')
                model_activation[name].append(output)
            return hook

        def register_model_forward_hook(model, path, model_activation, output_process=''):
            nodes = path.split('.')
            module = model
            for node in nodes:
                try:
                    if str.isnumeric(node):
                        module = module[int(node)]
                    else:
                        module = module.__getattr__(node)
                except:
                    raise AttributeError('There is no path {} in the model.'.format(path))
            return module.register_forward_hook(
                get_activation(path, model_activation, output_process)
                )

        assert isinstance(self.student_model, torch.nn.Module) and \
               isinstance(self.teacher_model, torch.nn.Module), \
               'Expect student_model and teacher_model to be an torch.nn.Module object, ' + \
               'got student_model:{} and teacher_model:{}'.format(
                   type(self.student_model), type(self.teacher_model)
                   )
        self.hook_handles = []
        for idx in range(len(self.layer_mappings)):
            student_layer, teacher_layer = self.layer_mappings[idx]
            student_output_process, teacher_output_process = self.layer_output_process[idx]
            st_handle = register_model_forward_hook(self.student_model, student_layer, 
                                        self.student_features, student_output_process)
            te_handle = register_model_forward_hook(self.teacher_model, teacher_layer, 
                                        self.teacher_features, teacher_output_process)
            self.hook_handles.extend([st_handle, te_handle])

    def remove_all_hooks(self):
        for hook in self.hook_handles:
            hook.remove()

    def init_loss_funcs(self):
        for loss_type in self.loss_types:
            if loss_type == 'MSE':
                loss_func = torch.nn.MSELoss()
            elif loss_type == 'KL':
                loss_func = torch.nn.KLDivLoss()
            elif loss_type == 'L1':
                loss_func = torch.nn.L1Loss()
            else:
                raise NotImplementedError('Unsupported loss type {}, supported loss is ' + \
                    'MSE for mean squared error, KL for Kullback-Leibler divergence and ' + \
                    'L1 for L1 loss.'.format(loss_type))
            self.loss_funcs.append(loss_func)

    def init_feature_matcher(self, student_feature, teacher_feature):
        class pytorch_linear_feature_matcher(torch.nn.Module):
            def __init__(self, src_shape, dst_shape):
                super().__init__()
                shape_diff = [abs(i - j) for i, j in zip(dst_shape, src_shape)]
                assert shape_diff.count(0) == len(shape_diff) - 1, 'Expect only one ' + \
                        'different dimension between student_feature and teacher_feature.'
                self.dim_idx = np.argmax(shape_diff)
                self.dense = torch.nn.Linear(src_shape[self.dim_idx], dst_shape[self.dim_idx])

            def forward(self, input):
                output = torch.transpose(input, self.dim_idx, -1)
                if input.device != next(self.parameters()).device:
                    self.to(input.device)
                output = self.dense(output)
                output = torch.transpose(output, self.dim_idx, -1)
                return output

        assert isinstance(student_feature, (torch.Tensor, np.ndarray)) and \
            isinstance(teacher_feature, (torch.Tensor, np.ndarray)), \
            'Expect student_feature and teacher_feature to be torch.Tensor or np.ndarray ' + \
            'objects, got student_feature a {st} object, teacher_feature a {tt} object.'.format(
                st=type(student_feature), tt=type(teacher_feature)
            )
        assert len(student_feature.shape) == len(teacher_feature.shape), \
            'Expect student_feature and teacher_feature to have the same length of shape, ' + \
            'got student_feature of {}, teacher_feature of {}.'.format(
                student_feature.shape, teacher_feature.shape
                )
        if sum([abs(i - j) for i, j in zip(student_feature.shape, teacher_feature.shape)]) == 0:
            return lambda x:x
        return pytorch_linear_feature_matcher(student_feature.shape, teacher_feature.shape)

    def teacher_model_forward(self, input, teacher_model=None, device=None):
        model = self.teacher_model if teacher_model is None else teacher_model
        assert isinstance(model, torch.nn.Module), \
            'Teacher model should be a torch Module instead of {}'.format(type(model))
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
            return pytorch_forward_wrapper(model, input, device=device)

    def loss_cal(self):
        self.loss = 0
        init_feature_matchers = False
        if self.feature_matchers is None:
            init_feature_matchers = True
            self.feature_matchers = {}
        for idx in range(len(self.layer_mappings)):
            student_layer, teacher_layer = self.layer_mappings[idx]
            student_feature = self.student_features[student_layer]
            teacher_feature = self.teacher_features[teacher_layer]
            assert len(student_feature) == len(teacher_feature) and len(student_feature) > 0, \
                'student_feature or teacher_feature is empty, please run student and ' + \
                'teacher model forward before calculating the loss.'
            def device2feature_gen(features):
                devices_count = Counter([f.device for f in features])
                assert [1] * len(devices_count) == [_ for _ in devices_count.values()], \
                    'Currently only support 1 feature tensor per device, ' + \
                    'got {}.'.format(devices_count)
                return {feat.device:feat for feat in features}

            student_feature = device2feature_gen(student_feature)
            teacher_feature = device2feature_gen(teacher_feature)
            assert student_feature.keys() == teacher_feature.keys(), \
                'Features from student model have different devices with that of ' + \
                'teacher model, got student: {}, teacher: {}.'.format(student_feature.keys(),
                                                                    teacher_feature.keys())
            output_device = torch.device('cuda:0') \
                if torch.device('cuda:0') in student_feature.keys() else torch.device('cpu')
            if init_feature_matchers:
                feature_matcher = self.init_feature_matcher(student_feature[output_device], 
                                                            teacher_feature[output_device])
                self.feature_matchers[student_layer] = feature_matcher

            tmp_loss = 0
            for device in student_feature.keys():
                student_feature[device] = student_feature[device].to(output_device)
                teacher_feature[device] = teacher_feature[device].to(output_device)
                stfeat, tefeat = student_feature[device], teacher_feature[device]
                stfeat = self.feature_matchers[student_layer](stfeat)
                if self.loss_types[idx] == 'KL':
                    check_is_not_prob = \
                        lambda x:(torch.abs(x.sum(dim=-1) - 1.0) > 0.2).any().item()
                    if isinstance(self.feature_matchers[student_layer], torch.nn.Module):
                        stfeat = torch.nn.LogSoftmax(dim=-1)(stfeat)
                    else:
                        if check_is_not_prob(stfeat):
                            stfeat = torch.softmax(stfeat, dim=-1)
                        stfeat = torch.log(stfeat+1e-9)
                    if check_is_not_prob(tefeat):
                        tefeat = torch.softmax(tefeat, dim=-1)
                tmp_loss += self.loss_funcs[idx](stfeat, tefeat) * self.loss_weights[idx]
            self.loss += tmp_loss
            self.student_features[student_layer] = []
            self.teacher_features[teacher_layer] = []
        return self.loss

@criterion_registry('IntermediateLayersKnowledgeDistillationLoss', 'pytorch')
class PyTorchIntermediateLayersKnowledgeDistillationLossWrapper(object):
    def __init__(self, param_dict):
        self.param_dict = param_dict

    def _param_check(self):
        param_dict = self.param_dict
        _params = ['layer_mappings', 'loss_types', 'loss_weights', 'add_origin_loss']
        layer_mappings = param_dict['layer_mappings']
        if 'loss_types' not in param_dict:
            param_dict['loss_types'] = ['MSE'] * len(layer_mappings)
        if 'loss_weights' not in param_dict:
            param_dict['loss_weights'] = [1.0 / len(layer_mappings)] * len(layer_mappings)
        if 'add_origin_loss' not in param_dict:
            param_dict['add_origin_loss'] = False
        assert 'layer_mappings' in param_dict, \
            'Key layer_mappings must be in input parameters.'
        assert all(type(param_dict[k]) in [list, tuple] \
            for k in ['layer_mappings', 'loss_types', 'loss_weights']), \
            'Type of loss_types and loss_weights must be list or tuple.'
        assert isinstance(param_dict['add_origin_loss'], bool), \
            'Type of add_origin_loss should be bool.'
        assert len(param_dict['layer_mappings']) == \
            len(param_dict['loss_types']) == len(param_dict['loss_weights']),\
            'Length of layer_mappings, loss_types and loss_weights must be the same.'
        assert all(type(it) in [list, tuple] and (len(it) == 2 or len(it) == 4) \
            for it in param_dict['layer_mappings']), \
            'Elements of layer_mappings must be list or tuple and with length of 2 or 4.' + \
            'For example, with length of 4, element looks like [\'student_model.layer1' + \
            '.attention\', \'1\', \'teacher_model.layer1.attention\', \'1\'], where ' + \
            '\'student_model.layer1.attention\' and \'teacher_model.layer1.attention\' ' + \
            'represent attention module on layer 1 of the student model and the ' + \
            'teacher model respectively, two \'1\' represent the index to retrieve the ' + \
            'desired output from the defined module\'s outputs, in this case, the above ' + \
            'two module\'s outputs are lists, with desired output in index 1 of these ' + \
            'lists, in cases of dict output, retrieving can be done by define the ' + \
            'corresponding key, in cases of module\'s output is the desired output, ' + \
            'just adopt an element of length 2 such as [\'student_model.layer1.output' + \
            '.output\', \'teacher_model.layer1.output\'].'
        assert all(any(isinstance(e, t) for t in [str, torch.nn.Module]) \
            for e in param_dict['loss_types']), \
            'Type of loss_types element must be str or torch Module.'
        assert all(0. <= e <= 1. for e in param_dict['loss_weights']) and \
            abs(sum(param_dict['loss_weights']) - 1.0) < 1e-9, \
            'Element of loss_weights must be in interval [0, 1] and summed to 1.0.'
        new_dict = {}
        for k in _params:
            new_dict[k] = param_dict[k]
        return new_dict

    def __call__(self, **kwargs):
        return PyTorchIntermediateLayersKnowledgeDistillationLoss, self._param_check()
