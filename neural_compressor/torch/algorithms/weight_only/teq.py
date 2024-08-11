#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
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
"""TEQ quantization."""

from typing import Any, List

import torch

from neural_compressor.torch.algorithms.base_algorithm import Quantizer
from neural_compressor.torch.utils import get_accelerator, get_model_device, is_transformers_imported, logger

from .modules import MulLinear, TEQLinearFakeQuant
from .utility import get_module, quant_tensor, set_module

if is_transformers_imported():
    import transformers

__all__ = ["TrainableEquivalentTransformation", "TEQuantizer"]


class TrainableEquivalentTransformation:
    """Weight-only quantization, Trainable Equivalent Transformation (TEQ)."""

    _PREPARE_ATTRS: List[str] = ["weight_config", "trained_alphas"]
    _PREPARE_ATTRS_PREFIX = "_prepare_"

    def __init__(self, model, weight_config={}, absorb_to_layer=None, folding=True, example_inputs=None):
        """Init the TrainableEquivalentTransformation object.

        Args:
            model (torch.nn.module): the model for quantization
            weight_config (dict, optional): contains all info required by RTN. Defaults to {}.
            absorb_to_layer (dict): The layer dict that scale can be absorbed. Default to None.
            folding(bool): Allow insert mul before linear when the scale cannot be absorbed by last layer.
                            Default to True.
            example_inputs: inputs for trace. Default to None.
        """
        self.model = model
        self.weight_config = weight_config
        self.folding = folding
        self.example_inputs = example_inputs
        self.device = self._get_device()
        self.trained_alphas = {}
        self.absorb_to_layer = absorb_to_layer
        self._post_initialized = False

    def _detect_absorb_to_layer(self, model, folding, example_inputs):
        # If user not provide the layers to absorb the quantization, detect layers automatically
        supported_layers = ["Linear"]
        detected_absorb_layers = {}
        # Detect the layers that can be absorbed automatically
        if folding:
            from neural_compressor.torch.algorithms.weight_only.utility import GraphTrace

            tg = GraphTrace()
            detected_absorb_layers, _ = tg.get_absorb_to_layer(model, example_inputs, supported_layers)
        else:  # pragma: no cover
            for name, module in model.named_modules():
                if module.__class__.__name__ in supported_layers:
                    detected_absorb_layers[name] = [name]
        logger.info("Detected **absorb layer**: **absorbed layers**")
        logger.info(detected_absorb_layers)
        return detected_absorb_layers

    def _post_init(self):
        self.dtype = self._get_dtype()
        self.model.to(self.device)
        self.model.eval()
        self._post_initialized = True

    def _get_device(self):
        """Get the model device.

        Returns:
            str: Model device.
        """
        device = get_accelerator().current_device_name()
        return device

    def _get_dtype(self):
        for _, p in self.model.named_parameters():
            return p.data.dtype

    def add_tuning_scale(self, sqrt_w_init=False):
        """Add tuning scales.

        Args:
            sqrt_w_init: use sqrt weight to init.
        """
        if not self.absorb_to_layer:
            self.absorb_to_layer = self._detect_absorb_to_layer(self.model, self.folding, self.example_inputs)
        if not self._post_initialized:
            self._post_init()
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
                if not self.weight_config.get(layer_name):  # pragma: no cover
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

        for layer_name, m in self.model.named_modules():
            if isinstance(m, torch.nn.Linear) and "orig_layer" not in n:
                if not self.weight_config.get(layer_name):  # pragma: no cover
                    logger.info(f"out of absorbed layer {layer_name} not in weight config, skip.")
                    continue
                num_bits = self.weight_config[layer_name]["bits"]
                group_size = self.weight_config[layer_name]["group_size"]
                scheme = self.weight_config[layer_name]["scheme"]

                alpha = torch.nn.Parameter(torch.ones(m.weight.shape[1], device=self.device))
                alpha.requires_grad_(False)
                wrapper_module = TEQLinearFakeQuant(
                    orig_layer=m, alpha=alpha, num_bits=num_bits, group_size=group_size, scheme=scheme
                )
                set_module(self.model, layer_name, wrapper_module)
        # Attach the weight config captured at prepare stage to the model
        self.model._weight_config = self.weight_config
        self.model._trained_alphas = self.trained_alphas

    @torch.no_grad()
    def _absorb_scales(self, layer, scale, layer_name=""):
        """Absorb the scale to the layer at output channel.

        Args:
            layer: the module.
            scale: the scale to be absorbed.
            layer_name: the layer name.
        """
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

        elif (
            layer.__class__.__name__ == "LlamaRMSNorm" or layer.__class__.__name__ == "T5LayerNorm"
        ):  # pragma: no cover
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
        """Scale the layer weights at input channel, depthwise conv output channel.

        Args:
            layer: the layer.
            scale: the scale to be multiplied.
        """
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
        if not self._post_initialized:  # pragma: no cover
            self._post_init()
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

    @torch.no_grad()
    def quantize(self, **kwargs):
        """quantization."""
        use_optimum_format = kwargs.get("use_optimum_format", True)
        device = get_accelerator().current_device_name()
        model_device = get_model_device(self.model)  # return model on the same device
        model = self.model
        for name, m in model.named_modules():
            if self.weight_config.get(name) is None:  # pragma: no cover
                logger.info(f"quantize layer {name} not in weight config, skip.")
                continue
            num_bits = self.weight_config[name]["bits"]
            group_size = self.weight_config[name]["group_size"]
            scheme = self.weight_config[name]["scheme"]
            group_dim = self.weight_config[name].get("group_dim", 1)
            # for only group_dim is 0 or only `transformers.Conv1D`, we need transpose weight.
            if is_transformers_imported():
                transpose = (group_dim == 0) ^ (isinstance(m, transformers.Conv1D))
            else:  # pragma: no cover
                transpose = group_dim == 0
            if transpose:  # pragma: no cover
                weight = m.weight.detach().T.contiguous()
            else:
                weight = m.weight.detach()
            if isinstance(m, torch.nn.Linear):  # pragma: no cover
                int_weight, scale, zp = quant_tensor(
                    weight.data,
                    num_bits=num_bits,
                    group_size=group_size,
                    scheme=scheme,
                    return_int=True,
                )
                int_weight = int_weight.t_().contiguous() if transpose else int_weight
                scale = scale.t_().contiguous() if transpose else scale
                zp = zp.t_().contiguous() if transpose and zp is not None else zp
                if isinstance(m, torch.nn.Linear):
                    in_features = m.in_features
                    out_features = m.out_features
                elif is_transformers_imported() and isinstance(m, transformers.Conv1D):
                    in_features = m.weight.shape[0]
                    out_features = m.weight.shape[1]
                    int_weight = int_weight.t_().contiguous()
                    scale = scale.t_().contiguous()
                    zp = zp.t_().contiguous() if zp is not None else zp
                from .modules import INCWeightOnlyLinear

                new_module = INCWeightOnlyLinear(
                    in_features,
                    out_features,
                    bits=num_bits,
                    group_size=group_size,
                    zp=zp is not None,
                    bias=m.bias is not None,
                    use_optimum_format=use_optimum_format,
                    device=device,
                )
                new_module.pack(int_weight, scale, zp, m.bias)
                if name == "":
                    return new_module
                else:
                    set_module(model, name, new_module)
                # Move modules back to the model device layer-by-layer
                m.to(model_device)
                new_module.to(model_device)
        self.model = model

    def save(self, save_scale_file="", save_state_dict_file=""):
        """Save alpha/scale or model weight.

        Args:
            save_scale_file: path to save alpha/scale with torch.save.
            save_state_dict_file: path to save model state_dict.
        """
        if save_scale_file:  # pragma: no cover
            torch.save(self.trained_alphas, save_scale_file)

        if save_state_dict_file:  # pragma: no cover
            torch.save(self.model.state_dict(), save_state_dict_file)


class TEQuantizer(Quantizer):
    """TEQ Quantizer."""

    def __init__(self, quant_config, folding, example_inputs, absorb_to_layer=None):
        """Init the TEQuantizer object."""
        super().__init__(quant_config=quant_config)
        self.folding = folding
        self.absorb_to_layer = absorb_to_layer
        self.example_inputs = example_inputs
        self._quantizer = TrainableEquivalentTransformation(
            model=None,
            weight_config=quant_config,
            absorb_to_layer=absorb_to_layer,
            folding=folding,
            example_inputs=example_inputs,
        )

    def prepare(self, model, *args, **kwargs):
        """Prepares a given model for quantization.

        Args:
            model: A float model to be quantized.

        Returns:
            A prepared model.
        """
        float_model = model
        assert isinstance(model, torch.nn.Module), "only support torch module"
        self._quantizer.model = float_model
        logger.info("TEQ quantizing start.")
        self._quantizer.add_tuning_scale()
        for attr in self._quantizer._PREPARE_ATTRS:
            setattr(float_model, self._quantizer._PREPARE_ATTRS_PREFIX + attr, getattr(self._quantizer, attr))
        return float_model

    def convert(self, model, *args: Any, **kwargs: Any):
        """Convert the prepared model to a quantized model.

        Args:
            model (torch.nn.Module): the prepared model

        Returns:
            The quantized model.
        """
        for attr in self._quantizer._PREPARE_ATTRS:
            setattr(self._quantizer, attr, getattr(model, self._quantizer._PREPARE_ATTRS_PREFIX + attr, None))
        self._quantizer.model = model
        self._quantizer.transform()
        self._quantizer.quantize(**kwargs)
        logger.info("TEQ quantizing done.")
        return self._quantizer.model
