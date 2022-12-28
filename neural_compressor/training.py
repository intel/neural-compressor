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

"""The configuration of the training loop."""

import copy
from .conf.pythonic_config import Config
from .config import DistillationConfig, QuantizationAwareTrainingConfig, WeightPruningConfig
from .experimental.distillation import Distillation
from .experimental.quantization import Quantization
from .experimental.scheduler import Scheduler
from .utils import logger
from typing import Callable, List, Union
from neural_compressor.experimental.pruning_v2 import Pruning

class CompressionManager:
    """CompressionManager is uesd in train loop for what user want to deal with additional.

    Arguments:
        commponent: one instance of Distillation, Quantization, Pruning, Scheduler

    Examples:
        import neural_compressor.training.prepare_compression
        compression_manager = prepare_compression(conf, model)
        compression_manager.callbacks.on_train_begin()
        model = compression_manager.model
        train_loop:
            for epoch in range(epochs):
                compression_manager.callbacks.on_epoch_begin(epoch)
                for i, batch in enumerate(dataloader):
                    compression_manager.callbacks.on_step_begin(i)
                    ......
                    output = model(batch)
                    loss = ......
                    loss = compression_manager.callbacks.on_after_compute_loss(batch, output, loss)
                    loss.backward()
                    compression_manager.callbacks.on_before_optimizer_step()
                    optimizer.step()
                    compression_manager.callbacks.on_step_end()
                compression_manager.callbacks.on_epoch_end()
        compression_manager.callbacks.on_train_end()
        compression_manager.save("path_to_save")
    """
    def __init__(self, component):
        """Training loop initialization."""
        self.callbacks = self.CallBacks(component)
        self.model = component.model
        try:
            # TODO: export to ONNX model need original fp32 model now, will remove it
            #  when int8 model can be exported to ONNX model.
            self.fp32_model = copy.deepcopy(component.model)
        except Exception as e:  # pragma: no cover
            logger.warning("Fail to deep copy the model due to {}.".format(repr(e)))
            self.fp32_model = None

    class CallBacks:
        """Define the basic command for the training loop."""
        def __init__(self, component):
            """Callbacks are used for execute the training procedure."""
            self.callbacks = \
                component.components[0] if isinstance(component, Scheduler) else component

        def on_train_begin(self, dataloader=None):
            """Called before the beginning of epochs."""
            self.callbacks.on_train_begin(dataloader)

        def on_train_end(self):
            """Called after the end of epochs."""
            self.callbacks.on_train_end()
            logger.info("Training finished!")

        def on_epoch_begin(self, epoch):
            """Called on the beginning of epochs."""
            self.callbacks.on_epoch_begin(epoch)

        def on_step_begin(self, batch_id):
            """Called on the beginning of batches."""
            self.callbacks.on_step_begin(batch_id)

        def on_after_compute_loss(self, input, student_output, student_loss, teacher_output=None):
            """Called on the end of loss computation."""
            return self.callbacks.on_after_compute_loss(
                input, student_output, student_loss, teacher_output=None
            )

        def on_before_optimizer_step(self):
            """Called on the end of backward."""
            self.callbacks.on_before_optimizer_step()

        def on_after_optimizer_step(self):
            """Called on the end of backward."""
            self.callbacks.on_after_optimizer_step()

        def on_step_end(self):
            """Called on the end of batches."""
            return self.callbacks.on_step_end()

        def on_epoch_end(self):
            """Called on the end of epochs."""
            return self.callbacks.on_epoch_end()

    def save(self, root=None):
        """Save compressed model.

        Args:
            root (str): path to save the model
        """
        self.model.save(root)

    def export(
        self,
        save_path: str,
        conf,
    ):
        """Convert the model to another type model, like `onnx` model and so on.

        Args:
            save_path (str): The path to save the model
            conf (Union[Callable, List]) : The configure for onnx exportation.
        """
        self.model.export(save_path, conf)


def prepare_compression(model: Callable, confs: Union[Callable, List], **kwargs):
    """Summary.

    Args:
        model (Callable, optional):    The model to optimize.
        confs (Union[Callable, List]): Config of Distillation, Quantization, Pruning,
                                       or list of config for orchestration optimization
        options (Options, optional):   The configure for random_seed, workspace,
                                       resume path and tensorboard flag.

    Returns:
        CompressionManager

    Examples:
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
    if isinstance(confs, List) and len(confs) > 1:
        from .experimental.scheduler import Scheduler
        comps = []
        for conf in confs:
            if isinstance(conf, QuantizationAwareTrainingConfig):
                conf_ = Config(quantization=conf,
                               benchmark=None,
                               pruning=None,
                               distillation=None,
                               nas=None)
                com = Quantization(conf_)
                com.model = model
            elif isinstance(conf, WeightPruningConfig):
                conf_ = Config(pruning=conf,
                               benchmark=None,
                               quantization=None,
                               distillation=None,
                               nas=None)
                com = Pruning(conf_)
                com.model = model
            elif isinstance(conf, DistillationConfig):
                conf_ = Config(distillation=conf,
                               benchmark=None,
                               quantization=None,
                               pruning=None,
                               nas=None)
                com = Distillation(conf_)
                com.model = model
                if conf.teacher_model is not None:
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
        if isinstance(confs, List):
            confs = confs[0]
        if isinstance(confs, QuantizationAwareTrainingConfig):
            conf = Config(quantization=confs,
                          benchmark=None,
                          pruning=None,
                          distillation=None,
                          nas=None)
            component = Quantization(conf)
        elif type(confs) == WeightPruningConfig:
            conf = Config(pruning=confs,
                          benchmark=None,
                          quantization=None,
                          distillation=None,
                          nas=None)
            component = Pruning(conf)
        elif type(confs) == DistillationConfig:
            conf = Config(distillation=confs,
                          benchmark=None,
                          quantization=None,
                          pruning=None,
                          nas=None)
            component = Distillation(conf)
            if confs.teacher_model is not None:
                component.teacher_model = confs.teacher_model
        else:
            assert False, logger.error(
                "confs should be one of QuantizationAwareTrainingConfig, "
                "PruningConfig, DistillationConfig. not {}".format(type(confs))
            )

        component.model = model
        if isinstance(confs, QuantizationAwareTrainingConfig):
            component.prepare_qat()
        else:
            component.prepare()
    compression_manager = CompressionManager(component)

    return compression_manager
