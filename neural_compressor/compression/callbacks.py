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

The Component class will be inherited by the class 'QuantizationAwareTrainingCallbacks',
'PruningCallbacks' and 'DistillationCallbacks'.
"""

from ..model import BaseModel, Model
from ..model.model import MODELS
from ..utils import logger
from ..utils.utility import LazyImport
from .distillation.criterions import Criterions
from .pruner.pruners import PRUNERS, get_pruner
from .pruner.utils import get_sparsity_ratio, get_sparsity_ratio_tf, parse_to_prune, parse_to_prune_tf, process_config

LazyImport("torch.nn")
torch = LazyImport("torch")
tf = LazyImport("tensorflow")


class BaseCallbacks(object):
    """This is base class of Neural Compressor Callbacks.

    This class will be inherited by the class 'QuantizationAwareTrainingCallbacks',
    'PruningCallbacks' and 'DistillationCallbacks'.
    This design is mainly for pruning/distillation/quantization-aware training.
    In this class will apply all hooks for 'Quantization', 'Pruning' and 'Distillation'.
    """

    def __init__(self, conf=None, model=None):
        """Construct all the necessary attributes for the callbacks object.

        Args:
            conf: A Config object which definds the compressor behavior.
                  Just like: QuantizationAwareTrainingConfig, WeightPruningConfig \
                    and DistillationConfig.
            model: Model to be compressed in this object. It should be neural compressor model.
        """
        assert model is None or isinstance(model, BaseModel), "The model should be a instanceof BaseModel"
        self.conf = conf
        self.framework = None
        self.model = model
        self.adaptor = None
        self.hooks = {
            "on_train_begin": self.on_train_begin,
            "on_train_end": self.on_train_end,
            "on_epoch_begin": self.on_epoch_begin,
            "on_epoch_end": self.on_epoch_end,
            "on_step_begin": self.on_step_begin,
            "on_step_end": self.on_step_end,
            "on_after_compute_loss": self.on_after_compute_loss,
            "on_before_optimizer_step": self.on_before_optimizer_step,
            "on_after_optimizer_step": self.on_after_optimizer_step,
            "on_before_eval": self.on_before_eval,
            "on_after_eval": self.on_after_eval,
        }
        self.hooks_dict = {
            "on_train_begin": [],
            "on_train_end": [],
            "on_epoch_begin": [],
            "on_epoch_end": [],
            "on_step_begin": [],
            "on_step_end": [],
            "on_after_compute_loss": [],
            "on_before_optimizer_step": [],
            "on_after_optimizer_step": [],
            "on_before_eval": [],
            "on_after_eval": [],
        }

    def on_train_begin(self, dataloader=None):
        """Be called before the beginning of training."""
        for on_train_begin_hook in self.hooks_dict["on_train_begin"]:
            on_train_begin_hook(dataloader)

    def on_train_end(self):
        """Be called after the end of training."""
        for on_train_end_hook in self.hooks_dict["on_train_end"]:
            on_train_end_hook()

    def on_epoch_begin(self, epoch):
        """Be called on the beginning of epochs."""
        for on_epoch_begin_hook in self.hooks_dict["on_epoch_begin"]:
            on_epoch_begin_hook(epoch)

    def on_step_begin(self, batch_id):
        """Be called on the beginning of batches."""
        if len(self.hooks_dict["on_step_begin"]) > 0:
            res_list = []
            for on_step_begin_hook in self.hooks_dict["on_step_begin"]:
                res_list.append(on_step_begin_hook(batch_id))
            return res_list
        else:
            return None

    def on_after_compute_loss(self, input, student_output, student_loss, teacher_output=None):
        """Be called on the end of loss computation."""
        if len(self.hooks_dict["on_after_compute_loss"]) > 0:
            loss = student_loss
            for on_after_compute_loss_hook in self.hooks_dict["on_after_compute_loss"]:
                loss = on_after_compute_loss_hook(input, student_output, loss, teacher_output)
            return loss
        else:
            return None

    def on_before_optimizer_step(self):
        """Be called before optimizer step."""
        for on_before_optimizer_step_hook in self.hooks_dict["on_before_optimizer_step"]:
            on_before_optimizer_step_hook()

    def on_after_optimizer_step(self):
        """Be called after optimizer step."""
        for on_after_optimizer_step_hook in self.hooks_dict["on_after_optimizer_step"]:
            on_after_optimizer_step_hook()

    def on_before_eval(self):
        """Be called before evaluation."""
        for on_before_eval_hook in self.hooks_dict["on_before_eval"]:
            on_before_eval_hook()

    def on_after_eval(self):
        """Be called after evaluation."""
        for on_after_eval_hook in self.hooks_dict["on_after_eval"]:
            on_after_eval_hook()

    def on_step_end(self):
        """Be called on the end of batches."""
        if len(self.hooks_dict["on_step_end"]) > 0:
            res_list = []
            for on_step_end_hook in self.hooks_dict["on_step_end"]:
                res_list.append(on_step_end_hook())
            return res_list
        else:
            return None

    def on_epoch_end(self):
        """Be called on the end of epochs."""
        res_list = []

        for on_epoch_end_hook in self.hooks_dict["on_epoch_end"]:
            res_list.append(on_epoch_end_hook())

        return res_list

    def register_hook(self, scope, hook, input_args=None, input_kwargs=None):
        """Register hook for component.

        Input_args and input_kwargs are reserved for user registered hooks.
        """
        if hook not in self.hooks_dict[scope]:
            self.hooks_dict[scope].append(hook)

    def __repr__(self):
        """Represent this class."""
        pass

    def remove_hook(self, scope, hook):
        """Remove hooks if user want to tune accuracy with train_func."""
        for registed_hook in self.hooks_dict[scope]:
            if type(hook) == type(registed_hook):
                self.hooks_dict[scope].remove(registed_hook)


class QuantizationAwareTrainingCallbacks(BaseCallbacks):
    """This is the class for callbacks of quantization aware training.

    This design is mainly for Quantization-Aware Training.
    In this class will apply all hooks for Quantization-Aware Training.
    """

    def __init__(self, conf=None, model=None, adaptor=None):
        """Construct all the necessary attributes for the callbacks object.

        Args:
            conf: A QuantizationAwareTrainingConfig object which definds the compressor behavior.
            model: Model to be quantized in this object. It should be neural compressor model.
        """
        super(QuantizationAwareTrainingCallbacks, self).__init__(conf=conf, model=model)
        self.register_hook("on_train_begin", adaptor._pre_hook_for_qat)
        self.register_hook("on_train_end", adaptor._post_hook_for_qat)

    def __repr__(self):
        """Represent this class."""
        return "Quantization Aware Training Callbacks"


class PruningCallbacks(BaseCallbacks):
    """This is the class for callbacks of pruning object.

    In this class will apply all hooks for Pruning.
    """

    def __init__(self, conf=None, model=None):
        """Construct all the necessary attributes for the callbacks object.

        Args:
            conf: A WeightPruningConfig object which definds the compressor behavior.
            model: Model to be Pruning in this object. It should be neural compressor model.
        """
        super(PruningCallbacks, self).__init__(conf=conf, model=model)
        self.pruners_info = process_config(self.conf)
        self.pruners = []
        self._generate_pruners()
        self.generate_hooks()

    def on_train_end(self):
        """Be called after the end of training."""
        for on_train_end_hook in self.hooks_dict["on_train_end"]:
            on_train_end_hook()
        if self.conf.framework == "pytorch" and isinstance(self.model.model, torch.nn.Module):
            get_sparsity_ratio(self.pruners, self.model)
        elif self.conf.framework == "keras" and isinstance(self.model.model, tf.keras.Model):
            get_sparsity_ratio_tf(self.pruners, self.model)

    def __repr__(self):
        """Return the class's string representation."""
        return "Pruning Callbacks"

    def generate_hooks(self):
        """Register hooks for pruning."""
        for pruner in self.pruners:
            for key in self.hooks.keys():
                if hasattr(pruner, key):
                    self.register_hook(key, getattr(pruner, key))

    def _generate_pruners(self):
        """Obtain Pruner objects."""
        if self.conf.framework == "pytorch" and isinstance(self.model.model, torch.nn.Module):
            # model auto slim related
            from .pruner.model_slim.pattern_analyzer import SelfMHASearcher

            for info in self.pruners_info:
                if "mha" in info["pattern"]:
                    # head pruning
                    pa_obj = SelfMHASearcher(self.model.model)
                    modules, _ = pa_obj.search(split_qkv_ffn=False)
                    modules = pa_obj.obtain_mha_module(modules)
                    modules = pa_obj.from_layer_name_to_object(modules)
                    if len(modules) == 0:
                        logger.warning("one pruner hooks no mha modules, please have a check")
                    self.pruners.append(get_pruner(info, modules))
                else:
                    # original pruning types, e.g NxM or N:M
                    modules = parse_to_prune(info, self.model.model)
                    if modules == {}:
                        logger.warning("one pruner hooks no layers, please have a check")

                    self.pruners.append(get_pruner(info, modules))
                    info["modules"] = [key for key in modules.keys()]
                    info["len_of_modules"] = len(info["modules"])
                    logger.info(info)
        elif self.conf.framework == "keras" and isinstance(self.model.model, tf.keras.Model):
            from tensorflow.python.ops.numpy_ops import np_config

            np_config.enable_numpy_behavior()
            for info in self.pruners_info:
                # original pruning types, e.g NxM or N:M
                modules = parse_to_prune_tf(info, self.model.model)
                if modules == {}:
                    logger.warning("one pruner hooks no layers, please have a check")

                self.pruners.append(get_pruner(info, modules, "keras"))
                info["modules"] = [key for key in modules.keys()]
                info["len_of_modules"] = len(info["modules"])
                logger.info(info)
        else:
            assert False, "now only support {}".format(PRUNERS.keys())


class DistillationCallbacks(BaseCallbacks):
    """Distillation class derived from Component class.

    Distillation class abstracted the pipeline of knowledge distillation,
    transfer the knowledge of the teacher model to the student model.

    Args:
        conf: Distillation_Conf containing teacher model, distillation criterion etc.
        model: Student model. It should be neural compressor model.

    Attributes:
        _epoch_ran: A integer indicating how much epochs ran.
        eval_frequency: The frequency for doing evaluation of the student model
            in terms of epoch.
        best_score: The best metric of the student model in the training.
        best_model: The best student model found in the training.
    """

    def __init__(self, conf=None, model=None):
        """Initialize the attributes."""
        super(DistillationCallbacks, self).__init__(conf=conf, model=model)

        self.framework = list(MODELS.keys())[list(MODELS.values()).index(self._parse_model_class(model))]
        self._teacher_model = None
        self._criterion = None
        self._epoch_ran = 0
        self._train_cfg = None
        self.eval_frequency = 1
        self.best_score = 0
        self.best_model = None
        self.hooks_registered = False
        assert hasattr(self.conf, "teacher_model"), "Please assign teacher model in DistillationConfig."
        self.teacher_model = self.conf.teacher_model
        self.generate_hooks()
        self.create_criterion()

    def _parse_model_class(self, model):
        """Parse model class for getting framework."""
        from neural_compressor.model.tensorflow_model import TensorflowBaseModel, TensorflowModel, TensorflowQATModel

        if isinstance(model, TensorflowQATModel):
            return type(model)
        if isinstance(model, TensorflowBaseModel):
            return TensorflowModel
        return type(model)

    def _on_step_begin(self, batch_id):
        """Operations called on the beginning of batches."""
        if self.criterion is not None and hasattr(self.criterion, "clear_features"):
            self.criterion.clear_features()

    def _on_after_compute_loss(self, input, student_output, student_loss, teacher_output=None):
        """Set or compute output of teacher model.

        Called after student model forward, calculate the output of the teacher model
        with the same input of the student model.

        Args:
            input (tensor or list or dict): The input of the student model.
            student_output (tensor): The output logits of the student model.
            student_loss (tensor or float): The original loss of the student model.
            teacher_output (tensor, optional): The output logits of the teacher model.
        """
        if self.criterion is None:
            self.create_criterion()
        assert self.criterion, "criterion must be set in yaml config file."
        if teacher_output is None:
            assert self.teacher_model, "teacher_model must be set."
            teacher_output = self.criterion.teacher_model_forward(input, teacher_model=self.teacher_model._model)
        return self.criterion.loss_cal_sloss(student_output, teacher_output, student_loss)

    def init_train_cfg(self):
        """Initialize the training configuration."""
        if self._train_cfg is None:
            # train section of distillation section in yaml file should be configured.
            self._train_cfg = self.conf.criterion
        assert self._train_cfg, (
            "train field of distillation section in yaml file must "
            "be configured for distillation if train_func is NOT set."
        )

    def create_criterion(self):
        """Create the criterion for training."""
        self.init_train_cfg()
        if self.criterion is None:
            assert self._train_cfg.config is not None, (
                "criterion part in train field of distillation section in yaml file "
                "must be configured for distillation if criterion is NOT set."
            )
            criterion_cfg = self._train_cfg.config
            assert (
                len(criterion_cfg) == 1
            ), "There must be exactly one loss in " "criterion part, instead got {} loss.".format(len(criterion_cfg))
            loss = [i for i in criterion_cfg.keys()][0]
            loss_cfg = criterion_cfg[loss]
            criterion_builder = Criterions(self.framework)[loss](loss_cfg)
            criterion_tuple = criterion_builder()
            if self.teacher_model and self.student_model:
                if self.framework == "tensorflow":  # new, for tf
                    teacher_model = self.teacher_model._model
                    student_model = self.student_model._model
                else:  # for pytorch and other frameworks
                    teacher_model = self.teacher_model.model
                    student_model = self.student_model.model
                criterion_tuple[1]["student_model"] = student_model
                criterion_tuple[1]["teacher_model"] = teacher_model
            self.criterion = criterion_tuple[0](**criterion_tuple[1])
        else:
            logger.warning("Use user defined criterion.")

        self._train_cfg.criterion = self.criterion

    def generate_hooks(self):
        """Register hooks for distillation.

        Register necessary hooks for distillation pipeline.
        """
        if not self.hooks_registered:
            self.register_hook("on_step_begin", self._on_step_begin)
            self.register_hook("on_after_compute_loss", self._on_after_compute_loss)
            self.hooks_registered = True

    @property
    def criterion(self):
        """Getter of criterion.

        Returns:
            The criterion used in the distillation process.
        """
        return self._criterion

    @criterion.setter
    def criterion(self, user_criterion):
        """Setter of criterion used in the distillation process.

        Set the user defined criterion. When using built-in train_func, user can
         specify the customized criterion through this setter.

        Args:
            user_criterion (criterion object): User defined criterion.
        """
        self._criterion = user_criterion

    @property
    def teacher_model(self):
        """Getter of the teacher model.

        Returns:
            The teacher model used in the distillation process.
        """
        return self._teacher_model

    @teacher_model.setter
    def teacher_model(self, user_model):
        """Set the user model and dispatch to framework specific internal model object.

        Args:
           user_model: user are supported to set model from original framework model format
                       (eg, tensorflow frozen_pb or path to a saved model),
                       but not recommended. Best practice is to set from a initialized
                       neural_compressor.Model. If tensorflow model is used, model's inputs/outputs will be
                       auto inferenced, but sometimes auto inferenced
                       inputs/outputs will not meet your requests,
                       set them manually in config yaml file.
                       Another corner case is slim model of tensorflow,
                       be careful of the name of model configured in yaml file,
                       make sure the name is in supported slim model list.
        """
        if not isinstance(user_model, BaseModel):
            logger.warning("Force convert framework model to neural_compressor model.")
            self._teacher_model = Model(user_model)
        else:
            self._teacher_model = user_model

    @property
    def student_model(self):
        """Getter of the student model.

        Returns:
            The student model used in the distillation process.
        """
        return self.model

    @property
    def train_cfg(self):
        """Getter of the train configuration.

        Returns:
            The train configuration used in the distillation process.
        """
        return self._train_cfg

    def __repr__(self):
        """Class representation."""
        return "Distillation Callbacks"
