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


from .smooth_quant import GraphTrace, get_module, set_module
from .weight_only import quant_weight
from .model_wrapper import TEQLinearFakeQuant, TEQMulLinear
import torch
from torch.functional import F
from torch.autograd import Function
import logging
from transformers import get_scheduler

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

logger = logging.getLogger(__name__)


class FakeAffineTensorQuantFunction(Function):
    """Fake version of affine quantization
    """

    @staticmethod
    def forward(ctx, inputs, num_bits=4, group_size=1024):
        """

        As it will be only applied on activation with per tensor granularity, broadcast is not needed.

        Args:
            ctx: Pytorch convention.
            inputs: A Tensor of type float32.
            min_range: A float.
            max_range: A float.
            num_bits: An integer

        Returns:
            outputs: A Tensor of type output_dtype
        """
        return quant_weight(inputs, num_bits, group_size)

    @staticmethod
    def backward(ctx, grad_outputs):
        """
        Args:
            ctx: Pytorch convention.
            grad_output: A tensor of gradient of outputs

        Returns:
            grad_inputs: A tensor of gradient
        """
        return grad_outputs, None, None


class TorchTEQ:
    """
    Weight-only quantization, Trainable Equivalent Transformation (TEQ): linear wrapper to apply scale to input
    """

    def __init__(self, model, dataloader, example_inputs=None, traced_model=None,
            num_bits=4, group_size=-1, scheme="asym"):
        """
        :param model: the model for quantization
        :param dataloader: train dataloader
        :param example_inputs: inputs for trace 
        :param traced_model: traced model for trace layers
        :param num_bits: quantization level
        :param group_size: for fine-grained quantization
        :param scheme: asym or sym
        """

        self.model = model
        self.num_bits = num_bits
        self.group_size = group_size
        self.scheme = scheme
        self.device, self.dtype = self._get_device()
        self.model.eval()
        self.dataloader = dataloader
        self.example_inputs = example_inputs
        self.traced_model = traced_model
        if self.traced_model == None:
            self.traced_model = self.model
        self.trained_alphas = {}

    def _get_device(self):
        """
        Get the model device
        :return:Model device
        """
        for _, p in self.model.named_parameters():
            return p.data.device, p.data.dtype

    def add_tuning_scale(self, folding=True,
            op_types=['Linear', 'Conv2d'], excluded_name="lm_head",
            excluded_key=None, sqrt_w_init=False):
        """
        The main entry of smooth quant
        :param alpha: Alpha value to balance the quantization difficulty of activation and weight, please refer
        to the paper for more details
        :param folding: whether insert mul(False) or just allow foldable layers(True) for SmoothQuant
        :param op_types: The op typed to be smooth quantized
        :param excluded_name: exclude layer
        :param excluded_key: exclude key
        :param sqrt_w_init: use sqrt weight to init
        """
        if folding:
            self.insert_mul = False
        else:
            self.insert_mul = True

        with torch.no_grad():
            if self.insert_mul:
                self.absorb_to_layer = self._get_all_layer_names()  # TODO: only support linear now.
            else:
                self.absorb_to_layer, no_absorb_layers = self._trace(
                        op_types)  ##TODO we need to insert mul layer for no_absorb_layers later
                if self.absorb_to_layer == None and no_absorb_layers == None:
                    logger.warning("sorry, could not trace the model, smooth quant is skipped")
                    logger.warning("if you are using huggingface model,"
                                       "you could set torchscript to True "
                                       "when loading the model or set the return_dict to False")
                elif self.absorb_to_layer == {}:
                    logger.warning("could not find any layer to be absorbed")
                else:
                    to_absorb_cnt = 0
                    for key, item in self.absorb_to_layer.items():
                        to_absorb_cnt += len(item)

                    logger.info(
                            f" {to_absorb_cnt} out of {to_absorb_cnt + len(no_absorb_layers)} "
                            f"layers could be absorbed in smooth quant")

        for n, p in self.model.named_parameters():
            p.requires_grad = False

        for key, item in self.absorb_to_layer.items():
            if len(item) == 1 and excluded_name in item[0]:
                excluded_key = key
                break

        if excluded_key != None:
            self.absorb_to_layer.pop(excluded_key)  ## remove

        for layer_norm in self.absorb_to_layer:
            if excluded_name in self.absorb_to_layer[layer_norm][0]:
                continue

            layer_0_name = self.absorb_to_layer[layer_norm][0]

            module = get_module(self.model, layer_0_name)

            if sqrt_w_init:
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
                module = get_module(self.model, layer_name)
                wrapper_module = TEQLinearFakeQuant(orig_layer=module, alpha=alpha,
                        num_bits=self.num_bits, group_size=self.group_size)
                set_module(self.model, layer_name, wrapper_module)

        for n, m in self.model.named_modules():
            if isinstance(m, torch.nn.Linear) and excluded_name not in n and "orig_layer" not in n:
                alpha = torch.nn.Parameter(torch.ones(m.weight.shape[1], device=self.device))
                alpha.requires_grad_(False)
                wrapper_module = TEQLinearFakeQuant(orig_layer=m, alpha=alpha,
                        num_bits=self.num_bits, group_size=self.group_size)
                set_module(self.model, n, wrapper_module)

    def _get_all_layer_names(self, op_types=['Linear']):
        """
        Try the model to find the layers which can be smooth quantized.
        :param op_types: The op types to be smooth quantized
        :return:
        """
        self_absorb_layer = {}
        for name, module in self.model.named_modules():
            for op_type in op_types:
                if op_type == str(module.__class__.__name__):
                    self_absorb_layer[name] = [name]
        return self_absorb_layer

    def _get_example_input(self):
        if self.dataloader == None and self.example_inputs == None:
            return None
        if self.example_inputs is None:
            ##assert self.dataloader, "Please provide dataloader or example_inputs"
            for idx, x in enumerate(self.dataloader):
                self.example_inputs = x
                break

        return self.example_inputs

    @torch.no_grad()
    def _absorb_scales(self, layer, scale, layer_name=""):
        """
        Absorb the scale to the layer at output channel
        :param layer: The module
        :param scale: The scale to be absorbed
        :param layer_name: The layer name
        """
        # for insert mul
        if self.insert_mul:
            if isinstance(layer, TEQMulLinear):
                set_module(self.model, layer_name, layer.sq_linear)  ##recover
            else:
                new_module = TEQMulLinear(layer, scale)
                set_module(self.model, layer_name, new_module)
            return

        if isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.GroupNorm) or \
                isinstance(layer, torch.nn.InstanceNorm2d):
            if layer.affine:
                layer.weight *= scale
                layer.bias *= scale
            else:
                layer.affine = True
                weight = torch.ones(layer.num_features, device=self.device, dtype=self.dtype) * scale
                layer.weight = torch.nn.Parameter(
                    weight, requires_grad=False)
                bias = torch.zeros(layer.num_features, device=self.device, dtype=self.dtype)
                layer.bias = torch.nn.Parameter(bias, requires_grad=False
                                                )
        elif isinstance(layer, torch.nn.LayerNorm):
            if layer.elementwise_affine:
                layer.weight *= scale
                layer.bias *= scale
            else:
                layer.elementwise_affine = True
                weight = torch.ones(layer.num_features, device=self.device, dtype=self.dtype) * scale
                layer.weight = torch.nn.Parameter(
                    torch.ones(weight, requires_grad=False))
                bias = torch.zeros(layer.num_features, device=self.device, dtype=self.dtype)
                layer.bias = torch.nn.Parameter(
                    bias, requires_grad=False)

        elif isinstance(layer, torch.nn.Conv2d):
            ## the order could not be changed
            if hasattr(layer, "bias") and (layer.bias != None):
                layer.bias *= scale
            scale = scale.view(scale.shape[0], 1, 1, 1)
            layer.weight *= scale

        elif isinstance(layer, torch.nn.Linear):
            if hasattr(layer, "bias") and (layer.bias != None):
                layer.bias *= scale
            scale = scale.view(scale.shape[0], 1)
            layer.weight *= scale


        elif layer.__class__.__name__ == "LlamaRMSNorm" \
                or layer.__class__.__name__ == "T5LayerNorm":  ##quite tricky
            layer.weight *= scale

        else:
            logger.info(f"found unsupported layer {type(layer)}, try to multiply scale to "
                f"weight and bias directly, this may introduce accuracy issue, please have a check ")
            if hasattr(layer, "weight") and layer.weight != None:
                layer.weight *= scale
            if hasattr(layer, "bias") and layer.bias != None:
                layer.bias *= scale

    @torch.no_grad()
    def _scale_layer_weight(self, layer, scale):  ##input channel
        """
        Scale the layer weights at input channel, depthwise conv output channel
        :param layer_name: The layer name
        :param scale: The scale to be multiplied
        :return:
        """
        if layer.__class__.__name__ == "TEQMulLinear":
            layer = layer.sq_linear

        if layer.__class__.__name__ == "TEQLinearFakeQuant":
            layer = layer.orig_layer

        scale = scale.view(1, scale.shape[0])
        layer.weight = torch.nn.Parameter(layer.weight * scale)
        return scale

    @torch.no_grad()
    def transform(self):
        """
        apply alpha/scale
        """
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

        # for insert_mul = False
        for n, m in self.model.named_modules():
            if isinstance(m, TEQLinearFakeQuant):
                set_module(self.model, n, m.orig_layer)

    def _trace(self, op_types):
        """
        Try the model to find the layers which can be smooth quantized.
        :param op_types: The op types to be smooth quantized
        :return:
        absorb_to_layer: A dict, absorb layer name:layers to be smooth quantized
        no_absorb_layers: A list saving the layers which could not find the absorb layer
        """
        tg = GraphTrace()
        self._get_example_input()
        absorb_to_layer, no_absorb_layers = tg.get_absorb_to_layer(self.traced_model, self.example_inputs, op_types)
        return absorb_to_layer, no_absorb_layers

    def train(self, train_steps=100, lr=1e-3, warmup_ratio=0.05, gradient_accumulation_steps=1, logging_steps=10,
            betas=[0.9, 0.9], weight_decay=0, optimizer=None, lr_scheduler=None, lr_scheduler_type="linear"):
        """
        train function
        """
        if optimizer is None:
            trained_alphas_list = []
            for item in self.trained_alphas.items():
                trained_alphas_list.append(item[1])
            optimizer = torch.optim.Adam(trained_alphas_list, lr=lr, weight_decay=weight_decay, betas=betas)


        if lr_scheduler is None:
            lr_scheduler = get_scheduler(
                    name=lr_scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=int(train_steps * warmup_ratio) // gradient_accumulation_steps,
                    num_training_steps=train_steps // gradient_accumulation_steps,
                    )

        logger.info("start training")
        self.model.train()
        global_steps = 0

        while True:
            for inputs in self.dataloader:
                if isinstance(inputs, dict):
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

                if global_steps == train_steps:
                    break
            if global_steps == train_steps:
                break

        logger.info("finish training")
        self.model.eval()

    @torch.no_grad()
    def quantize(self, scheme=None, quant_lm_head=False):
        """
        quantization
        """
        if scheme is None:
            scheme = self.scheme

        for n, m in self.model.named_modules():
            if quant_lm_head:
                if isinstance(m, torch.nn.Linear):
                    m.weight.data.copy_(
                            quant_weight(m.weight, num_bits=self.num_bits,
                                group_size=self.group_size, scheme=scheme))
            else:
                if isinstance(m, torch.nn.Linear) and "lm_head" not in n:
                    m.weight.data.copy_(
                            quant_weight(m.weight, num_bits=self.num_bits,
                                group_size=self.group_size, scheme=scheme))

    def save(save_scale_file="", save_state_dict_file=""):
        """
        save alpha/scale or model weight
        :param save_scale_file: save alpha/scale with torch.save
        :param save_state_dict_file: save model state_dict
        """
        if save_scale_file:
            torch.save(self.trained_alphas, save_scale_file)

        if save_state_dict_file:
            torch.save(self.model.state_dict(), save_state_dict_file)
