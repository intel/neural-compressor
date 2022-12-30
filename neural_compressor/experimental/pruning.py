"""pruning module."""
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

from .component import Component
from ..pruner.pruner_legacy import PRUNERS
from ..utils import logger
from ..utils.utility import GLOBAL_STATE, MODE
from ..utils.create_obj_from_config import create_dataloader, create_train_func, create_eval_func
from ..model import BaseModel
from ..adaptor import FRAMEWORKS
from ..conf.config import PruningConf
from ..conf.pythonic_config import Config

from deprecated import deprecated


class Pruning(Component):
    """This is base class of pruning object.

       Since DL use cases vary in the accuracy metrics (Top-1, MAP, ROC etc.), loss criteria
       (<1% or <0.1% etc.) and pruning objectives (performance, memory footprint etc.).
       Pruning class provides a flexible configuration interface via YAML for users to specify
       these parameters.

    Args:
        conf_fname_or_obj (string or obj): The path to the YAML configuration file or
            PruningConf class containing accuracy goal, pruning objective and related
            dataloaders etc.

    Attributes:
        conf: A config dict object. Contains pruning setting parameters.
        pruners: A list of Pruner object.

    """

    def __init__(self, conf_fname_or_obj=None):
        """Initiailize."""
        super(Pruning, self).__init__()
        if isinstance(conf_fname_or_obj, PruningConf):
            self.conf = conf_fname_or_obj
        elif isinstance(conf_fname_or_obj, Config):
            self.conf = PruningConf()
            self.conf.map_pyconfig_to_cfg(conf_fname_or_obj)
        else:
            self.conf = PruningConf(conf_fname_or_obj)
        self._init_with_conf()

        self.pruners = []
        self.callbacks = dict(tf_pruning=TfPruningCallback)

    def update_items_for_all_pruners(self, **kwargs):
        """Functions which add User-defined arguments to the original configurations.

        The original config of pruning is read from a file. 
        However, users can still modify configurations by passing key-value arguments in this function.
        Please note that the key-value arguments' keys are analysable in current configuration.
        """
        for item in self.cfg.pruning.approach.weight_compression_pytorch.pruners:
            for key in kwargs:
                if hasattr(item, key):
                    setattr(item, key, kwargs[key])

    def _on_train_begin(self, dataloader=None):
        """Functions called before training."""
        for pruner in self.pruners:
            pruner.on_train_begin(dataloader)

    def _on_epoch_begin(self, epoch):
        """Functions called on the beginning of epochs."""
        for pruner in self.pruners:
            pruner.on_epoch_begin(epoch)

    def _on_step_begin(self, batch_id):
        """Functions called on the beginning of batches."""
        res = []
        for pruner in self.pruners:
            res.append(pruner.on_step_begin(batch_id))
        return res

    def _on_before_optimizer_step(self):
        """Functions called after gradient computed, usually for getting gradients."""
        for pruner in self.pruners:
            pruner.on_before_optimizer_step()

    def _on_after_optimizer_step(self):
        """Functions called after optimzier step."""
        for pruner in self.pruners:
            pruner.on_after_optimizer_step()

    def _on_step_end(self):
        """Functions called on the end of batches."""
        res = []
        for pruner in self.pruners:
            res.append(pruner.on_step_end())
        return res

    def _on_epoch_end(self):
        """Functions called on the end of epochs."""
        res = []
        for pruner in self.pruners:
            res.append(pruner.on_epoch_end())
        if hasattr(self, "_model"):
            stats, sparsity = self._model.report_sparsity()
            logger.info(stats)
            logger.info(sparsity)
        return res

    def _on_train_end(self):
        """Functions called after training."""
        for pruner in self.pruners:
            pruner.on_train_end()

    def _on_before_eval(self):
        """Implement at the beginning of evaluation phase."""
        for pruner in self.pruners:
            pruner.on_before_eval()

    def _on_after_eval(self):
        """Implement at the end of evaluation phase."""
        for pruner in self.pruners:
            pruner.on_after_eval()

    def prepare(self):
        """Functions prepare for generate_hooks, generate_pruners."""
        self.generate_hooks()
        self.generate_pruners()

    def pre_process(self):
        """Functions called before pruning begins, usually set up pruners."""
        assert isinstance(self._model, BaseModel), 'need set neural_compressor Model for pruning....'

        GLOBAL_STATE.STATE = MODE.PRUNING
        framework_specific_info = {'device': self.cfg.device,
                                   'random_seed': self.cfg.tuning.random_seed,
                                   'workspace_path': self.cfg.tuning.workspace.path,
                                   'q_dataloader': None,
                                   'format': 'default',
                                   'backend': 'default'}

        if self.framework == 'tensorflow':
            framework_specific_info.update(
                {"inputs": self.cfg.model.inputs, "outputs": self.cfg.model.outputs})

        self.adaptor = FRAMEWORKS[self.framework](framework_specific_info)

        self.prepare()

        if self._train_dataloader is None and self._train_func is None:
            train_dataloader_cfg = self.cfg.pruning.train.dataloader
            assert train_dataloader_cfg is not None, \
                   'dataloader field of train field of pruning section ' \
                   'in yaml file should be configured as train_dataloader property is NOT set!'
            train_dataloader_cfg.distributed = self.train_distributed
            self._train_dataloader = create_dataloader(self.framework, train_dataloader_cfg)

        if self._eval_dataloader is None and self._eval_func is None:
            eval_dataloader_cfg = self.cfg.evaluation.accuracy.dataloader
            assert eval_dataloader_cfg is not None, \
                   'dataloader field of evaluation ' \
                   'in yaml file should be configured as eval_dataloader property is NOT set!'
            eval_dataloader_cfg.distributed = self.evaluation_distributed
            self._eval_dataloader = create_dataloader(self.framework, eval_dataloader_cfg)

        if self._train_func is None:
            # train section of pruning section in yaml file should be configured.
            train_cfg = self.cfg.pruning.train
            assert train_cfg, "train field of pruning section in yaml file must " \
                              "be configured for pruning if pruning_func is NOT set."
            self._train_func = create_train_func(self.framework, \
                                                   self.train_dataloader, \
                                                   self.adaptor, \
                                                   train_cfg, \
                                                   hooks=self.hooks, \
                                                   callbacks=self.callbacks)
        if self._eval_func is None:
            # eval section in yaml file should be configured.
            eval_cfg = self.cfg.evaluation
            assert eval_cfg, "eval field of pruning section in yaml file must " \
                              "be configured for pruning if eval_func is NOT set."
            self._eval_func = create_eval_func(self.framework, \
                                               self.eval_dataloader, \
                                               self.adaptor, \
                                               eval_cfg.accuracy.metric, \
                                               eval_cfg.accuracy.postprocess, \
                                               fp32_baseline = False)
        if getattr(self.train_dataloader, 'distributed', False):
            self.register_hook('on_train_begin', self.adaptor._pre_hook_for_hvd)

    def execute(self):
        """Functions that execute the pruning process.

        Follow the working flow: evaluate the dense model -> train/prune the model, evaluate the sparse model.
        """
        logger.info("Start to get the baseline model's score before pruning.")
        self.baseline_score = self._eval_func(self._model if getattr(self._eval_func, 'builtin', None) \
                        else self._model.model)
        logger.info("Baseline model's score is {}.".format(str(self.baseline_score)))
        logger.info("Model pruning begins.")
        self._train_func(self._model if getattr(self._train_func, 'builtin', None) \
                        else self._model.model)
        logger.info("Model pruning is done. Start to evaluate the pruned model.")
        self.last_score = self._eval_func(self._model if getattr(self._eval_func, 'builtin', None) \
                        else self._model.model)
        logger.info("Pruned model score is {}.".format(str(self.last_score)))
        return self._model

    def generate_hooks(self):
        """Register hooks for pruning."""
        self.register_hook('on_train_begin', self._on_train_begin)
        self.register_hook('on_train_end', self._on_train_end)
        self.register_hook('on_epoch_begin', self._on_epoch_begin)
        self.register_hook('on_epoch_end', self._on_epoch_end)
        self.register_hook('on_step_begin', self._on_step_begin)
        self.register_hook('on_step_end', self._on_step_end)
        self.register_hook('on_before_optimizer_step', self._on_before_optimizer_step)
        self.register_hook('on_after_optimizer_step', self._on_after_optimizer_step)
        self.register_hook('on_before_eval', self._on_before_eval)
        self.register_hook('on_after_eval', self._on_after_eval)

    def generate_pruners(self):
        """Functions that generate pruners and set up self.pruners."""
        for name in self.cfg.pruning.approach:
            assert name == 'weight_compression' or name == "weight_compression_pytorch", \
                'now we only support weight_compression and weight_compression_pytorch'

            if self.cfg.pruning.approach.weight_compression_pytorch != None:
                from .pytorch_pruner.pruning import Pruning as PytorchPruning
                self.pytorch_pruner = PytorchPruning(self.cfg)
                self.pytorch_pruner.model = self.model._model # extract their pytorch model
                self.pytorch_pruner.prepare()
                self.pytorch_pruner.on_train_begin()
                self.pruners += self.pytorch_pruner.pruners

            if self.cfg.pruning.approach.weight_compression != None:
                for pruner in self.cfg.pruning.approach.weight_compression.pruners:
                    if pruner.prune_type == 'basic_magnitude':
                        self.pruners.append(PRUNERS['BasicMagnitude'](\
                                                self._model, \
                                                pruner,
                                                self.cfg.pruning.approach.weight_compression))
                    elif pruner.prune_type == 'pattern_lock':
                        self.pruners.append(PRUNERS['PatternLock'](\
                                                self._model, \
                                                pruner,
                                                self.cfg.pruning.approach.weight_compression))
                    elif pruner.prune_type == 'gradient_sensitivity':
                        self.pruners.append(PRUNERS['GradientSensitivity'](\
                                                self._model, \
                                                pruner,
                                                self.cfg.pruning.approach.weight_compression))
                    elif pruner.prune_type == 'group_lasso':
                        self.pruners.append(PRUNERS['GroupLasso'](\
                                                self._model, \
                                                pruner,
                                                self.cfg.pruning.approach.weight_compression))
                    else:
                        ##print(pruner.prune_type)
                        assert False, 'now only support {}'.format(PRUNERS.keys())

    def __call__(self):
        """Entry point of pruning.

           This interface currently only works on pytorch
           and provides three usages:
           a) Fully yaml configuration: User specifies all the info through yaml,
              including dataloaders used in training and evaluation phases
              and pruning tuning settings.

              For this usage, only model parameter is mandatory.

           b) Partial yaml configuration: User specifies dataloaders used in training
              and evaluation phase by code.
              The tool provides built-in dataloaders and evaluators, user just need provide
              a dataset implemented __iter__ or __getitem__ methods and invoke dataloader()
              with dataset as input parameter to create neural_compressor dataloader before calling this
              function.

              After that, User specifies fp32 "model", training dataset "p_dataloader"
              and evaluation dataset "eval_dataloader".

              For this usage, model, p_dataloader and eval_dataloader parameters are mandatory.

           c) Partial yaml configuration: User specifies dataloaders used in training phase
              by code.
              This usage is quite similar with b), just user specifies a custom "eval_func"
              which encapsulates the evaluation dataset by itself.
              The trained and pruned model is evaluated with "eval_func".
              The "eval_func" tells the tuner whether the pruned model meets
              the accuracy criteria. If not, the Tuner starts a new training and tuning flow.

              For this usage, model, q_dataloader and eval_func parameters are mandatory.

        Returns:
            pruned model: best pruned model found, otherwise return None

        """
        return super(Pruning, self).__call__()

    """This makes pruning.fit() equals to pruning()."""
    fit = __call__

    @property
    def pruning_func(self):
        """Not support get pruning_func."""
        assert False, 'Should not try to get the value of `pruning_func` attribute.'
        return None

    @pruning_func.setter
    @deprecated(version='2.0', reason="please use `train_func` instead")
    def pruning_func(self, user_pruning_func):
        """Training function for pruning.

        Args:
            user_pruning_func: This function takes "model" as input parameter
                         and executes entire training process with self
                         contained training hyper-parameters. If pruning_func set,
                         an evaluation process must be triggered and user should
                         set eval_dataloader with metric configured or directly eval_func
                         to make evaluation of the model executed.
        """
        self._train_func = user_pruning_func

    @property
    def evaluation_distributed(self):
        """Getter to know whether need distributed evaluation dataloader."""
        eval_dataloader_cfg = self.cfg.evaluation.accuracy.dataloader
        yaml_distributed = eval_dataloader_cfg.get('distributed', False)
        return self._evaluation_distributed or yaml_distributed

    @evaluation_distributed.setter
    def evaluation_distributed(self, distributed):
        """Work with the former function."""
        self._evaluation_distributed = distributed

    @property
    def train_distributed(self):
        """Getter to know whether need distributed training dataloader."""
        train_dataloader_cfg = self.cfg.pruning.train.dataloader
        yaml_distributed = train_dataloader_cfg.get('distributed', False)
        return self._train_distributed or yaml_distributed

    @train_distributed.setter
    def train_distributed(self, distributed):
        """Work with the former function."""
        self._train_distributed = distributed

    def __repr__(self):
        """Return the class's string representation."""
        return 'Pruning'

    def get_sparsity_ratio(self):
        """Functions that calculate a modules/layers sparsity.

        Returns:
            Three floats.
            elementwise_over_matmul_gemm_conv refers to zero elements' ratio in pruning layers.
            elementwise_over_all refers to zero elements' ratio in all layers in the model.
            blockwise_over_matmul_gemm_conv refers to all-zero blocks' ratio in pruning layers.
        """
        import torch
        pattern_sparsity_cnt = 0
        element_sparsity_cnt = 0
        for pruner in self.pruners:
            modules = pruner.modules
            sparsity_ratio = pruner.pattern.get_sparsity_ratio(pruner.masks)
            cnt = 0
            for key in modules.keys():
                cnt += modules[key].weight.numel()
            pattern_sparsity_cnt += int(cnt * sparsity_ratio)
            for key in pruner.masks.keys():
                element_sparsity_cnt += torch.sum(pruner.masks[key] == 0).data.item()

        linear_conv_cnt = 0
        param_cnt = 0
        for name, module in self.model.named_modules():
            if type(module).__name__ in ["Linear"] or "Conv" in type(module).__name__:
                linear_conv_cnt += module.weight.numel()

        for n, param in self.model.named_parameters():
            param_cnt += param.numel()
        blockwise_over_matmul_gemm_conv = float(pattern_sparsity_cnt) / linear_conv_cnt
        elementwise_over_matmul_gemm_conv = float(element_sparsity_cnt) / linear_conv_cnt
        elementwise_over_all = float(
            element_sparsity_cnt) / param_cnt

        return elementwise_over_matmul_gemm_conv, elementwise_over_all, blockwise_over_matmul_gemm_conv


class TfPruningCallback(object):
    """Class that contains callback functions.

    Args:
        nc_model: A neural compression model object.
        hooks: A dict. Contains pure-defined hooks.
    """

    def __init__(self, nc_model, input_model, hooks):
        """Initialize."""
        self.hooks = hooks
        self.nc_model = nc_model
        self.model = input_model

    def __getitem__(self, func):
        """Get the class's function."""
        return getattr(self, func)

    def _set_weights(self):
        """Copy the input model's weight to the nc_model."""
        res = {}
        for index, layer in enumerate(self.model.layers):
            if len(layer.weights):
                res[index] = layer.get_weights()[0]
        self.nc_model.weights = res

    def on_train_begin(self, logs=None, dataloader=None):
        """Call the same-name function from hooks."""
        self.hooks['on_train_begin'](dataloader)

    def on_train_end(self, logs=None):
        """Call the same-name function from hooks."""
        self.hooks['on_train_end']()

    @deprecated(version='2.0', reason="please use `on_train_begin` instead")
    def pre_epoch_begin(self, logs=None, dataloader=None):  # pragma: no cover
        """Call the same-name function from hooks."""
        self.on_train_begin(logs, dataloader)

    @deprecated(version='2.0', reason="please use `on_train_end` instead")
    def post_epoch_end(self, logs=None):  # pragma: no cover
        """Call the same-name function from hooks."""
        self.on_train_end(logs)

    def on_epoch_begin(self, epoch, logs=None):
        """Call the same-name function from hooks."""
        self._set_weights()
        self.hooks['on_epoch_begin'](epoch)

    def on_epoch_end(self, logs=None):
        """Call the same-name function from hooks."""
        self._set_weights()
        res = self.hooks['on_epoch_end']()
        for layer_index, weights in res[0][0].items():
            get_weights = self.model.layers[layer_index].get_weights()
            get_weights[0] = weights
            self.model.layers[layer_index].set_weights(get_weights)

    def on_step_begin(self, batch, logs=None):
        """Call the same-name function from hooks."""
        self._set_weights()
        res = self.hooks['on_step_begin'](batch)
        for layer_index, weights in res[0][0].items():
            get_weights = self.model.layers[layer_index].get_weights()
            get_weights[0] = weights
            self.model.layers[layer_index].set_weights(get_weights)

    @deprecated(version='2.0', reason="please use `on_step_begin` instead")
    def on_batch_begin(self, batch, logs=None):  # pragma: no cover
        """Call the same-name function from hooks."""
        self.on_step_begin(batch, logs)

    def on_after_compute_loss(self, input, s_outputs, s_loss, t_outputs=None):
        """Call the same-name function from hooks."""
        return self.hooks['on_after_compute_loss'](input, s_outputs, s_loss, t_outputs)

    def on_step_end(self, logs=None):
        """Call the same-name function from hooks."""
        self._set_weights()
        res = self.hooks['on_step_end']()
        for layer_index, weights in res[0][0].items():
            get_weights = self.model.layers[layer_index].get_weights()
            get_weights[0] = weights
            self.model.layers[layer_index].set_weights(get_weights)

    @deprecated(version='2.0', reason="please use `on_step_end` instead")
    def on_batch_end(self, logs=None):  # pragma: no cover
        """Call the same-name function from hooks."""
        self.on_step_end(logs)
