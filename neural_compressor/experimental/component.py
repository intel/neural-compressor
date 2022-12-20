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

"""This is a module for Component class.

The Component class will be inherited by the class 'Quantization', 'Pruning' and 'Distillation'.
"""

from ..conf.config import Conf
from ..utils import logger
from ..utils.utility import required_libs
from ..utils.create_obj_from_config import create_dataloader, create_train_func, create_eval_func
from ..model import BaseModel
from .common import Model
from ..adaptor import FRAMEWORKS
from ..model.model import get_model_fwk_name
import importlib
from deprecated import deprecated


class Component(object):
    """This is base class of Neural Compressor Component.

    This class will be inherited by the class 'Quantization', 'Pruning' and 'Distillation'.
    This design is mainly for one-shot optimization for pruning/distillation/quantization-aware training.
    In this class will apply all hooks for 'Quantization', 'Pruning' and 'Distillation'.
    """

    def __init__(self, conf_fname_or_obj=None, combination=None):
        """Construct all the necessary attributes for the Component object.

        Args:
            conf_fname_or_obj: A YAML configuration file path or a Config object which definds the compressor behavior.
            combination: What components to be combined in this object.
        """
        self.conf = None
        self.cfg = None
        self.combination = combination
        self.framework = None
        self._model = None
        self._train_func = None
        self._train_dataloader = None
        self._eval_func = None
        self._eval_dataloader = None
        self._train_distributed = False
        self._evaluation_distributed = False
        self.adaptor = None
        self._metric = None
        self.hooks = {
            'on_train_begin': self.on_train_begin,
            'on_train_end': self.on_train_end,
            'on_epoch_begin': self.on_epoch_begin,
            'on_epoch_end': self.on_epoch_end,
            'on_step_begin': self.on_step_begin,
            'on_step_end': self.on_step_end,
            'on_after_compute_loss': self.on_after_compute_loss,
            'on_before_optimizer_step': self.on_before_optimizer_step,
            'on_after_optimizer_step': self.on_after_optimizer_step,
            'on_before_eval': self.on_before_eval,
            'on_after_eval': self.on_after_eval
        }
        self.hooks_dict = {
            'on_train_begin': [],
            'on_train_end': [],
            'on_epoch_begin': [],
            'on_epoch_end': [],
            'on_step_begin': [],
            'on_step_end': [],
            'on_after_compute_loss': [],
            'on_before_optimizer_step': [],
            'on_after_optimizer_step': [],
            'on_before_eval': [],
            'on_after_eval': []
        }
        if conf_fname_or_obj is not None:  # pragma: no cover
            if isinstance(conf_fname_or_obj, str):
                self.conf = Conf(conf_fname_or_obj)
            elif isinstance(conf_fname_or_obj, Conf):
                self.conf = conf_fname_or_obj
            else:
                assert False, \
                    "Please pass a YAML configuration file path and name or \
                    Conf class to Component"
            self._init_with_conf()

    def _init_with_conf(self):
        """Initialize some attributers."""
        self.cfg = self.conf.usr_cfg
        if self.cfg.model.framework != 'NA':
            self.framework = self.cfg.model.framework.lower()
            if self.framework in required_libs:
                for lib in required_libs[self.framework]:
                    try:
                        importlib.import_module(lib)
                    except Exception as e:
                        logger.error("{}.".format(e))
                        raise RuntimeError("{} is not correctly installed. " \
                            "Please check your environment".format(lib))

    def prepare(self):
        """Register Quantization Aware Training hooks."""
        if self.combination is not None and 'Quantization' in self.combination:
            if self.adaptor is None:
                framework_specific_info = {'device': self.cfg.device,
                                        'random_seed': self.cfg.tuning.random_seed,
                                        'workspace_path': self.cfg.tuning.workspace.path,
                                        'q_dataloader': None}
                if self.cfg.quantization.approach is not None:
                    framework_specific_info['approach'] = self.cfg.quantization.approach

                if 'tensorflow' in self.framework:
                    framework_specific_info.update(
                        {"inputs": self.cfg.model.inputs, "outputs": self.cfg.model.outputs})
                self.adaptor = FRAMEWORKS[self.framework](framework_specific_info)
                self.adaptor.model = self.model
            self.register_hook('on_train_begin', self.adaptor._pre_hook_for_qat)
            self.register_hook('on_train_end', self.adaptor._post_hook_for_qat)

    def prepare_qat(self):
        """Register Quantization Aware Training hooks."""
        if self.adaptor is None:
            framework_specific_info = {'device': self.cfg.device,
                                    'random_seed': self.cfg.tuning.random_seed,
                                    'workspace_path': self.cfg.tuning.workspace.path,
                                    'q_dataloader': None,
                                    'backend': self.cfg.model.get('backend', 'default'),
                                    'format': self.cfg.model.get('quant_format', 'default')}
            if self.cfg.quantization.approach is not None:
                framework_specific_info['approach'] = self.cfg.quantization.approach

            if 'tensorflow' in self.framework:
                framework_specific_info.update(
                    {"inputs": self.cfg.model.inputs, "outputs": self.cfg.model.outputs})
            self.adaptor = FRAMEWORKS[self.framework](framework_specific_info)
            self.adaptor.model = self.model
        self.register_hook('on_train_begin', self.adaptor._pre_hook_for_qat)
        self.register_hook('on_train_end', self.adaptor._post_hook_for_qat)

    def pre_process(self):
        """Initialize some attributes, such as the adaptor, the dataloader and train/eval functions from yaml config.

        Component base class provides default function to initialize dataloaders and functions
        from user config. And for derived classes(Pruning, Quantization, etc.), an override
        function is required.
        """
        if self.adaptor is None:
            # create adaptor
            framework_specific_info = {'device': self.cfg.device,
                                       'random_seed': self.cfg.tuning.random_seed,
                                       'workspace_path': self.cfg.tuning.workspace.path,
                                       'q_dataloader': None}
            if self.cfg.quantization.approach is not None:
                framework_specific_info['approach'] = self.cfg.quantization.approach

            if 'tensorflow' in self.framework:
                framework_specific_info.update(
                    {"inputs": self.cfg.model.inputs, "outputs": self.cfg.model.outputs})

            self.adaptor = FRAMEWORKS[self.framework](framework_specific_info)
            self.adaptor.model = self.model

        # create dataloaders
        if self._train_dataloader is None and self._train_func is None:
            train_dataloader_cfg = self.cfg.train.dataloader
            assert train_dataloader_cfg is not None, \
                   'No training dataloader setting in current component. Please check ' \
                   'dataloader field of train field in yaml file. Or manually pass ' \
                   'dataloader to component.'

            self._train_dataloader = create_dataloader(self.framework, train_dataloader_cfg)
        if self._eval_dataloader is None and self._eval_func is None:
            if self._eval_dataloader is None:
                eval_dataloader_cfg = self.cfg.evaluation.accuracy.dataloader
                assert eval_dataloader_cfg is not None, \
                   'No evaluation dataloader setting in current component. Please check ' \
                   'dataloader field of evaluation field in yaml file. Or manually pass ' \
                   'dataloader to component.'
                self._eval_dataloader = create_dataloader(self.framework, eval_dataloader_cfg)

        # create functions
        if self._train_func is None:
            self._train_func = create_train_func(self.framework,
                                                    self._train_dataloader,
                                                    self.adaptor,
                                                    self.cfg.train,
                                                    hooks=self.hooks)
        if self._eval_func is None:
            metric = [self._metric] if self._metric else self.cfg.evaluation.accuracy.metric
            self._eval_func = create_eval_func(self.framework,
                                               self._eval_dataloader,
                                               self.adaptor,
                                               metric,
                                               self.cfg.evaluation.accuracy.postprocess,
                                               fp32_baseline = False)

        self.prepare()
        # strategy will be considered in future
        if getattr(self.train_dataloader, 'distributed', False):
            self.register_hook('on_train_begin', self.adaptor._pre_hook_for_hvd)

    def execute(self):
        """Execute the processing of this compressor.

        Component base class provides compressing processing. And for derived classes(Pruning, Quantization, etc.),
        an override function is required.
        """
        # TODO: consider strategy sync during combination
        if self._train_func is not None:
            modified_model = self._train_func(self._model \
                    if getattr(self._train_func, 'builtin', None) else self._model.model)
            # for the cases that model is changed not inplaced during training, for example,
            # oneshot with torch_fx QAT interfaces. Needs to reset model afterwards.
            if modified_model is not None:
                self._model.model = modified_model
        if self._eval_func is not None:
            score = self._eval_func(self._model \
                    if getattr(self._eval_func, 'builtin', None) else self._model.model)
            logger.info("Evaluated model score is {}.".format(str(score)))
        return self._model

    def post_process(self):
        """Post process after execution.

        For derived classes(Pruning, Quantization, etc.), an override function is required.
        """
        pass

    def on_train_begin(self, dataloader=None):
        """Be called before the beginning of epochs."""
        for on_train_begin_hook in self.hooks_dict['on_train_begin']:
            on_train_begin_hook(dataloader)

    def on_train_end(self):
        """Be called after the end of epochs."""
        for on_train_end_hook in self.hooks_dict['on_train_end']:
            on_train_end_hook()

    @deprecated(version='2.0', reason="please use `on_train_begin` instead")
    def pre_epoch_begin(self, dataloader=None):
        """Be called before the beginning of epochs."""
        for on_train_begin_hook in self.hooks_dict['on_train_begin']:
            on_train_begin_hook(dataloader)

    @deprecated(version='2.0', reason="please use `on_train_end` instead")
    def post_epoch_end(self):
        """Be called after the end of epochs."""
        for on_train_end_hook in self.hooks_dict['on_train_end']:
            on_train_end_hook()

    def on_epoch_begin(self, epoch):
        """Be called on the beginning of epochs."""
        for on_epoch_begin_hook in self.hooks_dict['on_epoch_begin']:
            on_epoch_begin_hook(epoch)

    def on_step_begin(self, batch_id):
        """Be called on the beginning of batches."""
        res_list = []
        for on_step_begin_hook in self.hooks_dict['on_step_begin']:
            res_list.append(on_step_begin_hook(batch_id))
        return res_list

    @deprecated(version='2.0', reason="please use `on_step_begin` instead")
    def on_batch_begin(self, batch_id):
        """Be called on the beginning of batches."""
        return self.on_step_begin(batch_id)

    def on_after_compute_loss(self, input, student_output, student_loss, teacher_output=None):
        """Be called on the end of loss computation."""
        loss = student_loss
        for on_after_compute_loss_hook in self.hooks_dict['on_after_compute_loss']:
            loss = on_after_compute_loss_hook(input, student_output, loss, teacher_output)
        return loss

    def on_before_optimizer_step(self):
        """Be called before optimizer step."""
        for on_before_optimizer_step_hook in self.hooks_dict['on_before_optimizer_step']:
            on_before_optimizer_step_hook()

    def on_after_optimizer_step(self):
        """Be called after optimizer step."""
        for on_after_optimizer_step_hook in self.hooks_dict['on_after_optimizer_step']:
            on_after_optimizer_step_hook()

    def on_before_eval(self):
        """Be called before evaluation."""
        for on_before_eval_hook in self.hooks_dict['on_before_eval']:
            on_before_eval_hook()

    def on_after_eval(self):
        """Be called after evaluation."""
        for on_after_eval_hook in self.hooks_dict['on_after_eval']:
            on_after_eval_hook()

    @deprecated(version='2.0', reason="please use `on_before_optimizer_step` instead")
    def on_post_grad(self):
        """Be called before optimizer step."""
        return self.on_before_optimizer_step()

    def on_step_end(self):
        """Be called on the end of batches."""
        res_list = []
        for on_step_end_hook in self.hooks_dict['on_step_end']:
            res_list.append(on_step_end_hook())
        return res_list

    @deprecated(version='2.0', reason="please use `on_step_end` instead")
    def on_batch_end(self):
        """Be called on the end of batches."""
        return self.on_step_end()

    def on_epoch_end(self):
        """Be called on the end of epochs."""
        res_list = []

        for on_epoch_end_hook in self.hooks_dict['on_epoch_end']:
            res_list.append(on_epoch_end_hook())

        return res_list

    def register_hook(self, scope, hook, input_args=None, input_kwargs=None):
        """Register hook for component.

        Input_args and input_kwargs are reserved for user registered hooks.
        """
        if hook not in self.hooks_dict[scope]:
            self.hooks_dict[scope].append(hook)

    def __call__(self):
        """Execute this class.

        For derived classes(Pruning, Quantization, etc.), an override function is required.
        """
        self.pre_process()
        results = self.execute()
        self.post_process()
        return results

    def __repr__(self):
        """Represent this class."""
        if self.combination:
            return 'Combination of ' + ','.join(self.combination)
        else:
            return 'Base Component'

    @property
    def train_func(self):
        """Not support get train_func."""
        assert False, 'Should not try to get the value of `train_func` attribute.'
        return None

    @train_func.setter
    def train_func(self, user_train_func):
        """Training function.

        Args:
            user_train_func: This function takes "model" as input parameter
                         and executes entire training process with self
                         contained training hyper-parameters. If training_func set,
                         an evaluation process must be triggered and user should
                         set eval_dataloader with metric configured or directly eval_func
                         to make evaluation of the model executed. training_func will return
                         a trained model.
        """
        self._train_func = user_train_func

    @property
    def eval_func(self):
        """Not support get eval_func."""
        assert False, 'Should not try to get the value of `eval_func` attribute.'
        return None

    @eval_func.setter
    def eval_func(self, user_eval_func):
        """Eval function for component.

        Args:
            user_eval_func: This function takes "model" as input parameter
                         and executes entire evaluation process with self
                         contained metrics. If eval_func set,
                         an evaluation process must be triggered
                         to make evaluation of the model executed.
        """
        self._eval_func = user_eval_func

    @property
    def train_dataloader(self):
        """Getter to train dataloader."""
        return self._train_dataloader

    @train_dataloader.setter
    def train_dataloader(self, dataloader):
        """Set Data loader for training for Component.

        It is iterable and the batched data should consists of a tuple like
        (input, label) if the training dataset containing label, or yield (input, _)
        for label-free train dataset, the input in the batched data will be used for
        model inference, so it should satisfy the input format of specific model.
        In train process, label of data loader will not be used and
        neither the postprocess and metric. User only need to set
        train_dataloader when train_dataloader can not be configured from yaml file.

        Args:
            dataloader(generator): user are supported to set a user defined dataloader
                                   which meet the requirements that can yield tuple of
                                   (input, label)/(input, _) batched data. Another good
                                   practice is to use neural_compressor.experimental.common.DataLoader
                                   to initialize a neural_compressor dataloader object. Notice
                                   neural_compressor.experimental.common.DataLoader is just a wrapper of the
                                   information needed to build a dataloader, it can't yield
                                   batched data and only in this setter method
                                   a 'real' train_dataloader will be created,
                                   the reason is we have to know the framework info
                                   and only after the Component object created then
                                   framework information can be known.
                                   Future we will support creating iterable dataloader
                                   from neural_compressor.experimental.common.DataLoader.
        """
        from .common import _generate_common_dataloader
        self._train_dataloader = _generate_common_dataloader(
            dataloader, self.framework, self._train_distributed)

    @property
    def eval_dataloader(self):
        """Getter to eval dataloader."""
        return self._eval_dataloader

    @eval_dataloader.setter
    def eval_dataloader(self, dataloader):
        """Set Data loader for evaluation of component.

        It is iterable and the batched data should consists of yield (input, _).
        the input in the batched data will be used for model inference, so it
        should satisfy the input format of specific model.
        User only need to set eval_dataloader when eval_dataloader can not be
        configured from yaml file.

        Args:
            dataloader(generator): user are supported to set a user defined dataloader
                                   which meet the requirements that can yield tuple of
                                   (input, label)/(input, _) batched data. Another good
                                   practice is to use neural_compressor.experimental.common.DataLoader
                                   to initialize a neural_compressor dataloader object. Notice
                                   neural_compressor.experimental.common.DataLoader is just a wrapper of the
                                   information needed to build a dataloader, it can't yield
                                   batched data and only in this setter method
                                   a 'real' train_dataloader will be created,
                                   the reason is we have to know the framework info
                                   and only after the Component object created then
                                   framework information can be known.
                                   Future we will support creating iterable dataloader
                                   from neural_compressor.experimental.common.DataLoader.
        """
        from .common import _generate_common_dataloader
        self._eval_dataloader = _generate_common_dataloader(
            dataloader, self.framework, self._evaluation_distributed)

    @property
    def model(self):
        """Getter of model in neural_compressor.model."""
        return self._model

    @model.setter
    def model(self, user_model):
        """Set the user model and dispatch to framework specific internal model object.

        Args:
            user_model: user are supported to set model from original framework model format
                        (eg, tensorflow frozen_pb or path to a saved model),
                        but not recommended. Best practice is to set from a initialized
                        neural_compressor.experimental.common.Model.
                        If tensorflow model is used, model's inputs/outputs will be
                        auto inferenced, but sometimes auto inferenced
                        inputs/outputs will not meet your requests,
                        set them manually in config yaml file.
                        Another corner case is slim model of tensorflow,
                        be careful of the name of model configured in yaml file,
                        make sure the name is in supported slim model list.

        """
        if self.cfg.model.framework == 'NA':
            assert not isinstance(user_model, BaseModel), \
                "Please pass an original framework model but not neural compressor model!"
            self.framework = get_model_fwk_name(user_model)
            if self.framework == "tensorflow":
                from ..model.tensorflow_model import get_model_type
                if get_model_type(user_model) == 'keras' and self.cfg.model.backend == 'itex':
                    self.framework = 'keras'
            if self.framework == "pytorch":
                if self.cfg.model.backend == "default":
                    self.framework = "pytorch_fx"
                elif self.cfg.model.backend == "ipex":
                    self.framework = "pytorch_ipex"
            self.cfg.model.framework = self.framework

        if not isinstance(user_model, BaseModel):
            logger.warning("Force convert framework model to neural_compressor model.")
            self._model = Model(user_model, framework=self.framework)
        else:
            # It is config of neural_compressor version < 2.0, no need in 2.0
            if self.cfg.model.framework == "pytorch_ipex":
                from neural_compressor.model.torch_model import IPEXModel
                if not isinstance(user_model, IPEXModel):
                    self._model = Model(user_model.model, framework=self.cfg.model.framework)
                    return

            self._model = user_model

        if 'tensorflow' in self.framework:
            self._model.name = self.cfg.model.name
            self._model.output_tensor_names = self.cfg.model.outputs
            self._model.input_tensor_names = self.cfg.model.inputs
            self._model.workspace_path = self.cfg.tuning.workspace.path
