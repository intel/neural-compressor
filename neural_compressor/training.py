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

import copy
from .conf.pythonic_config import Config, DistillationConfig, Options, \
                                  PruningConfig, QuantizationAwareTrainingConfig
from .experimental.distillation import Distillation
from .experimental.pruning import Pruning
from .experimental.quantization import Quantization
from .experimental.scheduler import Scheduler
from .utils import logger
from typing import Callable, List, Union


class CompressionManager:
    """CompressionManager is uesd in train loop for what user want to deal with additional.

    arguments:
        commponent: one instance of Distillation, Quantization, Pruning, Scheduler

    examples:
        import neural_compressor.training.prepare_compression
        compression_manager = prepare_compression(conf, model)
        train_loop:
            compression_manager.on_train_begin()
            for epoch in range(epochs):
                compression_manager.on_epoch_begin(epoch)
                for i, batch in enumerate(dataloader):
                    compression_manager.on_step_begin(i)
                    ......
                    output = compression_manager.model(batch)
                    loss = ......
                    loss = compression_manager.on_after_compute_loss(batch, output, loss)
                    loss.backward()
                    compression_manager.on_before_optimizer_step()
                    optimizer.step()
                    compression_manager.on_step_end()
                compression_manager.on_epoch_end()
            compression_manager.on_train_end()
        compression_manager.save("path_to_save")
    """
    def __init__(self, component):
        self.callbacks = \
            component.components[0] if isinstance(component, Scheduler) else component
        self.model = component.model
        try:
            # TODO: export to ONNX model need original fp32 model now, will remove it
            #  when int8 model can be exported to ONNX model.
            self.fp32_model = copy.deepcopy(component.model)
        except Exception as e:  # pragma: no cover
            logger.warning("Fail to deep copy the model due to {}.".format(repr(e)))
            self.fp32_model = None

    def save(self, root=None):
        """Save compressed model.

        Args:
            root (str): path to save the model
        """
        self.model.save(root)

    def export(
        self,
        save_path: str,
        input,
        target_model_type: str = 'ONNX',
        quant_mode: str = 'QDQ',
        opset_version: int = 14,
        *args,
        **kwargs
    ):
        """Convert the model to another type model, like `onnx` model and so on.

        Args:

        """
        if target_model_type == "ONNX":
            if self.model.q_config is not None:
                assert self.fp32_model is not None, "Can't deepcopy fp32 model, so we can't " \
                    "export to onnx model now, this is a limitation, will remove in furture."
                self.model.export_to_int8_onnx(
                    save_path, input, opset_version=opset_version, fp32_model=self.fp32_model
                )
            else:
                self.model.export_to_fp32_onnx(save_path, input, opset_version=opset_version)
        else:
            assert False, "Unsupport export for {} model".format(type(self.model))


def prepare_compression(model: Callable, confs: Union[Callable, List], options=None, **kwargs):
    """_summary_

    Args:
        model (Callable, optional):    model to optimize.
        confs (Union[Callable, List]): config of Distillation, Quantization, Pruning,
                                       or list of config for orchestration optimization
        options (Options, optional):   The configure for random_seed, workspace,
                                       resume path and tensorboard flag.

    Returns:
        CompressionManager

    examples:
        import neural_compressor.training.prepare_compression
        compression_manager = prepare_compression(conf, model)
        train_loop:
            compression_manager.on_train_begin()
            for epoch in range(epochs):
                compression_manager.on_epoch_begin(epoch)
                for i, batch in enumerate(dataloader):
                    compression_manager.on_step_begin(i)
                    ......
                    output = model(batch)
                    loss = ......
                    loss = compression_manager.on_after_compute_loss(batch, output, loss)
                    loss.backward()
                    compression_manager.on_before_optimizer_step()
                    optimizer.step()
                    compression_manager.on_step_end()
                compression_manager.on_epoch_end()
            compression_manager.on_train_end()
    """

    if options is None:
        options = Options()
    if isinstance(confs, List):
        from .experimental.scheduler import Scheduler
        comps = []
        for conf in confs:
            if isinstance(conf, QuantizationAwareTrainingConfig):
                conf_ = Config(quantization=conf, options=options)
                com = Quantization(conf_)
            elif isinstance(conf, PruningConfig):
                conf_ = Config(pruning=conf, options=options)
                com = Pruning(conf_)
            elif isinstance(conf, DistillationConfig):
                conf_ = Config(distillation=conf, options=options)
                com = Distillation(conf_)
                assert conf.teacher_model is not None, \
                    "Please set teacher_model in DistillationConfig"
                com.teacher_model = conf.teacher_model
            else:
                assert False, "Unsupported configure: {}".format(type(conf))

            comps.append(com)
        scheduler = Scheduler()
        scheduler.model = model
        comp = scheduler.combine(*comps)
        comp.prepare()
        scheduler.append(comp)
        component = scheduler
    else:
        if isinstance(confs, QuantizationAwareTrainingConfig):
            conf = Config(quantization=confs, options=options)
            component = Quantization(conf)
        elif type(confs) == PruningConfig:
            conf = Config(pruning=confs, options=options)
            component = Pruning(conf)
        elif type(confs) == DistillationConfig:
            conf = Config(distillation=confs, options=options)
            component = Distillation(conf)
            assert confs.teacher_model is not None, \
                    "Please set teacher_model in DistillationConfig"
            component.teacher_model = confs.teacher_model
        else:
            assert False, logger.error(
                "confs should be one of QuantizationAwareTrainingConfig, "
                "PruningConfig, DistillationConfig. not {}".format(type(confs))
            )

        component.model = model
        if isinstance(confs, QuantizationAwareTrainingConfig):
            component.prepare_qat()
    compression_manager = CompressionManager(component)

    return compression_manager
