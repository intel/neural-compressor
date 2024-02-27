#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

try:
    from neural_compressor.utils.utility import LazyImport

    torch = LazyImport("torch")
    from ...utils import logger
except:  # pragma: no cover
    import logging

    import torch

    logger = logging.getLogger()

import transformers

from neural_compressor.adaptor.torch_utils.waq import get_module, set_module

from .model_wrapper import MulLinear, TEQLinearFakeQuant
from .weight_only import quant_weight


class TEQuantizer:
    """Weight-only quantization, Trainable Equivalent Transformation (TEQ): linear wrapper to apply scale to input."""

    def __init__(self, model, weight_config={}, absorb_to_layer={}, extra_config={}, example_inputs=None):
        """
        :param model: the model for quantization
        :param weight_config (dict, optional): contains all info required by GPTQ. Defaults to {}.
        :param example_inputs: inputs for trace
        """
        self.model = model
        self.weight_config = weight_config
        self.folding = extra_config.get("folding", True)
        self.example_inputs = example_inputs
        self.device, self.dtype = self._get_device()
        self.model.eval()
        self.trained_alphas = {}
        self.absorb_to_layer = absorb_to_layer

    def _get_device(self):
        """Get the model device
        :return:Model device."""
        for _, p in self.model.named_parameters():
            return p.data.device, p.data.dtype

    def add_tuning_scale(self, sqrt_w_init=False):
        """The main entry of smooth quant
        to the paper for more details
        :param sqrt_w_init: use sqrt weight to init."""

        # freeze model.
        for n, p in self.model.named_parameters():
            p.requires_grad = False

        for layer_norm in self.absorb_to_layer:
            layer_0_name = self.absorb_to_layer[layer_norm][0]

            module = get_module(self.model, layer_0_name)

            if sqrt_w_init:  # pragma: no cover
                weights = []
                for layer_name in self.absorb_to_layer[layer_norm]:
                    module = get_module(self.model, layer_name)
                    weights.append(module.weight)

                weights = torch.cat(weights, dim=0)
                max_value = torch.sqrt(torch.max(torch.abs(weights), dim=0).values)
                max_value[max_value == 0] = 1.0
                max_value = 1.0 / max_value

                alpha = torch.nn.Parameter(max_value)
                alpha = alpha.to(self.device)
            else:
                alpha = torch.nn.Parameter(torch.ones(module.weight.shape[1], device=self.device))

            self.trained_alphas[layer_norm] = alpha
            for layer_name in self.absorb_to_layer[layer_norm]:
                if self.weight_config.get(layer_name) is None:  # pragma: no cover
                    logger.info(f"layer {layer_name} not in weight config, skip.")
                    continue
                num_bits = self.weight_config[layer_name]["bits"]
                group_size = self.weight_config[layer_name]["group_size"]
                scheme = self.weight_config[layer_name]["scheme"]

                module = get_module(self.model, layer_name)
                wrapper_module = TEQLinearFakeQuant(
                    orig_layer=module, alpha=alpha, num_bits=num_bits, group_size=group_size, scheme=scheme
                )
                set_module(self.model, layer_name, wrapper_module)

        for n, m in self.model.named_modules():
            if isinstance(m, torch.nn.Linear) and "orig_layer" not in n:
                if self.weight_config.get(n) is None:  # pragma: no cover
                    logger.info(f"out of absorbed layer {n} not in weight config, skip.")
                    continue
                num_bits = self.weight_config[layer_name]["bits"]
                group_size = self.weight_config[layer_name]["group_size"]
                scheme = self.weight_config[layer_name]["scheme"]

                alpha = torch.nn.Parameter(torch.ones(m.weight.shape[1], device=self.device))
                alpha.requires_grad_(False)
                wrapper_module = TEQLinearFakeQuant(
                    orig_layer=m, alpha=alpha, num_bits=num_bits, group_size=group_size, scheme=scheme
                )
                set_module(self.model, n, wrapper_module)

    @torch.no_grad()
    def _absorb_scales(self, layer, scale, layer_name=""):
        """Absorb the scale to the layer at output channel
        :param layer: The module
        :param scale: The scale to be absorbed
        :param layer_name: The layer name."""
        # for insert mul
        if not self.folding:  # pragma: no cover
            if isinstance(layer, MulLinear):
                set_module(self.model, layer_name, layer.linear)  ##recover
            else:
                new_module = MulLinear(layer, scale)
                set_module(self.model, layer_name, new_module)
            self.weight_config[layer_name + ".linear"] = self.weight_config[layer_name]
            return

        if (
            isinstance(layer, torch.nn.BatchNorm2d)
            or isinstance(layer, torch.nn.GroupNorm)
            or isinstance(layer, torch.nn.InstanceNorm2d)
        ):
            if layer.affine:  # pragma: no cover
                layer.weight *= scale
                layer.bias *= scale
            else:  # pragma: no cover
                layer.affine = True
                weight = torch.ones(layer.num_features, device=self.device, dtype=self.dtype) * scale
                layer.weight = torch.nn.Parameter(weight, requires_grad=False)
                bias = torch.zeros(layer.num_features, device=self.device, dtype=self.dtype)
                layer.bias = torch.nn.Parameter(bias, requires_grad=False)
        elif isinstance(layer, torch.nn.LayerNorm):
            if layer.elementwise_affine:
                layer.weight *= scale
                layer.bias *= scale
            else:  # pragma: no cover
                layer.elementwise_affine = True
                weight = torch.ones(layer.num_features, device=self.device, dtype=self.dtype) * scale
                layer.weight = torch.nn.Parameter(torch.ones(weight, requires_grad=False))
                bias = torch.zeros(layer.num_features, device=self.device, dtype=self.dtype)
                layer.bias = torch.nn.Parameter(bias, requires_grad=False)

        elif isinstance(layer, torch.nn.Conv2d):  # pragma: no cover
            ## the order could not be changed
            if hasattr(layer, "bias") and (layer.bias is not None):
                layer.bias *= scale
            scale = scale.view(scale.shape[0], 1, 1, 1)
            layer.weight *= scale

        elif isinstance(layer, torch.nn.Linear):  # pragma: no cover
            if hasattr(layer, "bias") and (layer.bias is not None):
                layer.bias *= scale
            scale = scale.view(scale.shape[0], 1)
            layer.weight *= scale

        elif layer.__class__.__name__ == "LlamaRMSNorm" or layer.__class__.__name__ == "T5LayerNorm":  ##quite tricky
            layer.weight *= scale

        else:  # pragma: no cover
            logger.info(
                f"found unsupported layer {type(layer)}, try to multiply scale to "
                f"weight and bias directly, this may introduce accuracy issue, please have a check "
            )
            if hasattr(layer, "weight") and layer.weight is not None:
                layer.weight *= scale
            if hasattr(layer, "bias") and layer.bias is not None:
                layer.bias *= scale

    @torch.no_grad()
    def _scale_layer_weight(self, layer, scale):  ##input channel
        """Scale the layer weights at input channel, depthwise conv output channel
        :param layer_name: The layer name
        :param scale: The scale to be multiplied
        :return:"""
        if layer.__class__.__name__ == "MulLinear":
            layer = layer.linear

        if layer.__class__.__name__ == "TEQLinearFakeQuant":
            layer = layer.orig_layer

        scale = scale.view(1, scale.shape[0])
        layer.weight = torch.nn.Parameter(layer.weight * scale)
        return scale

    @torch.no_grad()
    def transform(self):
        """Apply alpha/scale."""
        for ln_name, layer_names in self.absorb_to_layer.items():
            module = get_module(self.model, ln_name)
            scale = self.trained_alphas[ln_name]
            scale = torch.clip(scale, 1e-5)
            input_scale = 1.0 / scale
            if hasattr(module, "orig_layer"):
                module = module.orig_layer

            self._absorb_scales(module, input_scale, layer_name=ln_name)
            weight_scale = scale
            for layer_name in layer_names:
                layer_module = get_module(self.model, layer_name)
                self._scale_layer_weight(layer_module, weight_scale)

        # for Folding = True
        for n, m in self.model.named_modules():
            if isinstance(m, TEQLinearFakeQuant):
                set_module(self.model, n, m.orig_layer)

    def train(
        self,
        dataloader,
        train_steps=1000,
        lr=1e-3,
        warmup_ratio=0.05,
        gradient_accumulation_steps=1,
        logging_steps=10,
        betas=[0.9, 0.9],
        weight_decay=0,
        lr_scheduler_type="linear",
    ):
        """Train function."""
        trained_alphas_list = []
        for item in self.trained_alphas.items():
            trained_alphas_list.append(item[1])
        optimizer = torch.optim.Adam(trained_alphas_list, lr=lr, weight_decay=weight_decay, betas=betas)

        lr_scheduler = transformers.get_scheduler(  # pylint: disable=E1111
            name=lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=int(train_steps * warmup_ratio) // gradient_accumulation_steps,
            num_training_steps=train_steps // gradient_accumulation_steps,
        )

        logger.info("start training")
        self.model.train()
        global_steps = 0

        while global_steps <= train_steps:
            for inputs in dataloader:
                if isinstance(inputs, torch.Tensor):
                    input_id = inputs
                elif isinstance(inputs, dict):
                    input_id = inputs["input_ids"]
                else:
                    input_id = inputs[0]

                input_id = input_id.to(self.device)
                output = self.model(input_id, labels=input_id)
                loss = output[0] / gradient_accumulation_steps
                loss.backward()
                global_steps += 1

                if global_steps % logging_steps == 0:
                    logger.info("steps: {}, loss: {}".format(global_steps, loss.detach().cpu().item()))

                if global_steps % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

                if global_steps >= train_steps:  # pragma: no cover
                    break

        logger.info("finish training")
        self.model.eval()
        return None

    @torch.no_grad()
    def quantize(self):
        """quantization."""

        for n, m in self.model.named_modules():
            if self.weight_config.get(n) is None:  # pragma: no cover
                logger.info(f"quantize layer {n} not in weight config, skip.")
                continue
            num_bits = self.weight_config[n]["bits"]
            group_size = self.weight_config[n]["group_size"]
            scheme = self.weight_config[n]["scheme"]
            if isinstance(m, torch.nn.Linear):  # pragma: no cover
                quant_weight(m.weight.data, num_bits=num_bits, group_size=group_size, scheme=scheme)

    def save(self, save_scale_file="", save_state_dict_file=""):
        """
        save alpha/scale or model weight
        :param save_scale_file: save alpha/scale with torch.save
        :param save_state_dict_file: save model state_dict
        """
        if save_scale_file:  # pragma: no cover
            torch.save(self.trained_alphas, save_scale_file)

        if save_state_dict_file:  # pragma: no cover
            torch.save(self.model.state_dict(), save_state_dict_file)
