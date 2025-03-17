#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
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
"""The quantizer using SmoothQuant path."""


import json
import os

import torch

try:
    import intel_extension_for_pytorch as ipex
except:  # pragma: no cover
    assert False, "Please install IPEX for smooth quantization."

from collections import OrderedDict
from types import MethodType

from packaging.version import Version

from neural_compressor.torch.algorithms import Quantizer

from .utility import (
    CpuInfo,
    TorchSmoothQuant,
    cfg_to_qconfig,
    dump_model_op_stats,
    get_ipex_version,
    get_quantizable_ops_recursively,
    ipex_config_path,
    logger,
    simple_inference,
    update_sq_scale,
)

ipex_ver = get_ipex_version()


class SmoothQuantQuantizer(Quantizer):
    """SmoothQuantQuantizer Class."""

    def __init__(self, quant_config: OrderedDict = {}):  # pragma: no cover
        """Init a SmoothQuantQuantizer object.

        Args:
            quant_config (OrderedDict, optional): quantization config for ops. Defaults to {}.
        """
        super().__init__(quant_config)

    def prepare(self, model, example_inputs, inplace=True, *args, **kwargs):
        """Prepares a given model for quantization.

        Args:
            model (torch.nn.Module): raw fp32 model or prepared model.
            example_inputs (tensor/tuple/dict): used to trace torch model.
            inplace (bool, optional): whether to carry out model transformations in-place. Defaults to True.

        Returns:
            A prepared model.
        """
        assert example_inputs is not None, "Please provide example_inputs for smooth quantization."
        assert not ipex_ver.release < Version("2.1").release, "IPEX version >= 2.1 is required for SmoothQuant."

        # Note: we should make sure smoothquant is only executed once with inplacing fp32 model.
        if hasattr(model, "_smoothquant_optimized") and model._smoothquant_optimized:  # pragma: no cover
            logger.info("The model is already optimized by SmoothQuant algorithm, skip it.")
            return model

        cfgs, op_infos_from_cfgs, output_tensor_id_op_name = (
            model.cfgs,
            model.op_infos_from_cfgs,
            model.output_tensor_id_op_name,
        )

        # check smoothquant alpha and act_algo value
        recipe_cfgs = self.quant_config.get("recipe_cfgs", None)
        alpha = recipe_cfgs["smooth_quant_args"]["alpha"]

        # Update json file in ipex_config_path
        cfg_to_qconfig(self.quant_config, cfgs, op_infos_from_cfgs, output_tensor_id_op_name, alpha, smooth_quant=True)
        model.eval()

        for op, _ in self.quant_config["op"].items():
            act_algo = self.quant_config["op"][op]["activation"]["algorithm"]

        # Check save_qconf_summary part is a workaround for IPEX bug.
        # Sometimes the prepared model from get_op_capablitiy loss this attribute.
        if not hasattr(model, "save_qconf_summary") or not hasattr(model, "load_qconf_summary"):  # pragma: no cover
            from torch.ao.quantization.observer import MinMaxObserver

            if ipex_ver.release >= Version("2.1.1").release:
                static_qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(
                    alpha=alpha, act_observer=MinMaxObserver
                )
            else:  # pragma: no cover
                if act_algo == "minmax":
                    static_qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(
                        alpha=alpha, act_observer=MinMaxObserver()
                    )
                    logger.warning(
                        "The int8 model accuracy will be close to 0 with MinMaxobserver, "
                        + "the suggested IPEX version is higher or equal than 2.1.100+cpu."
                    )
                else:
                    static_qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(alpha=alpha)

            if isinstance(example_inputs, dict):
                model = ipex.quantization.prepare(
                    model, static_qconfig, example_kwarg_inputs=example_inputs, inplace=inplace
                )
            else:
                model = ipex.quantization.prepare(model, static_qconfig, example_inputs=example_inputs, inplace=inplace)

        model.load_qconf_summary(qconf_summary=ipex_config_path)
        return model

    def convert(self, model, example_inputs, inplace=True, *args, **kwargs):
        """Converts a prepared model to a quantized model.

        Args:
            model (QuantizationInterceptionModule): the prepared model to be converted.
            example_inputs (tensor/tuple/dict): used to trace torch model.
            inplace (bool, optional): whether to carry out model transformations in-place. Defaults to True.

        Returns:
            A quantized model.
        """
        use_bf16 = self.quant_config.get("use_bf16", None)

        model.save_qconf_summary(qconf_summary=ipex_config_path)
        model = _ipex_post_quant_process(model, example_inputs, use_bf16, inplace=inplace)

        with open(ipex_config_path, "r") as f:
            model.tune_cfg = json.load(f)
        model.ipex_config_path = ipex_config_path
        dump_model_op_stats(self.quant_config["op"])

        from neural_compressor.torch.algorithms.smooth_quant import save

        logger.info("Smooth quantization done.")
        model.ori_save = model.save
        model.save = MethodType(save, model)
        return model

    def quantize(self, model, tune_cfg, run_fn, example_inputs, inplace=True, *args, **kwargs):
        """Executes the quantize process on the specified model.

        Args:
            model (torch.nn.Module): raw fp32 model or prepared model.
            tune_cfg (OrderedDict): quantization config for ops.
            run_fn (Callable): a calibration function for calibrating the model.
            example_inputs (tensor/tuple/dict): used to trace torch model.
            inplace (bool, optional): whether to carry out model transformations in-place. Defaults to True.

        Returns:
            A quantized model.
        """
        assert not ipex_ver.release < Version("2.1").release, "IPEX version >= 2.1 is required for SmoothQuant."

        cfgs, op_infos_from_cfgs, output_tensor_id_op_name = (
            model.cfgs,
            model.op_infos_from_cfgs,
            model.output_tensor_id_op_name,
        )

        use_bf16 = tune_cfg.get("use_bf16", None)

        # check smoothquant folding value
        recipe_cfgs = tune_cfg.get("recipe_cfgs", None)
        if "smooth_quant_args" in recipe_cfgs and "folding" in recipe_cfgs["smooth_quant_args"]:
            if recipe_cfgs["smooth_quant_args"]["folding"] is None:  # pragma: no cover
                if ipex_ver.release < Version("2.1").release:
                    folding = True
                else:
                    folding = False
            else:
                folding = recipe_cfgs["smooth_quant_args"]["folding"]

        # Note: we should make sure smoothquant is only executed once with inplacing fp32 model.
        if hasattr(model, "_smoothquant_optimized") and model._smoothquant_optimized:  # pragma: no cover
            logger.info("The model is already optimized by SmoothQuant algorithm, skip it.")
            return model

        sq_info = model.sq_info

        # Update model parameter when smoothquant folding = False
        if recipe_cfgs and recipe_cfgs.get("smooth_quant", False) and not folding:
            return qdq_quantize(
                model,
                tune_cfg,
                run_fn,
                example_inputs,
                inplace,
                cfgs,
                op_infos_from_cfgs,
                output_tensor_id_op_name,
                sq_info,
            )

        # Update model parameter when smoothquant folding = True
        if recipe_cfgs and recipe_cfgs.get("smooth_quant", False) and folding:
            _apply_pre_optimization(model, tune_cfg, sq_info)

        # Update json file in ipex_config_path
        cfg_to_qconfig(self.quant_config, cfgs, op_infos_from_cfgs, output_tensor_id_op_name)
        model.eval()

        # Check save_qconf_summary part is a workaround for IPEX bug.
        # Sometimes the prepared model from get_op_capablitiy loss this attribute
        if not hasattr(model, "save_qconf_summary") or not hasattr(model, "load_qconf_summary"):  # pragma: no cover
            static_qconfig = ipex.quantization.default_static_qconfig_mapping
            if isinstance(example_inputs, dict):
                model = ipex.quantization.prepare(
                    model, static_qconfig, example_kwarg_inputs=example_inputs, inplace=inplace
                )
            else:
                model = ipex.quantization.prepare(model, static_qconfig, example_inputs=example_inputs, inplace=inplace)

        run_fn(model)
        model.save_qconf_summary(qconf_summary=ipex_config_path)
        model = _ipex_post_quant_process(model, example_inputs, use_bf16, inplace=inplace)

        # Recover model parameter when smoothquant folding = True
        if (
            recipe_cfgs
            and recipe_cfgs.get("smooth_quant", False)
            and recipe_cfgs["smooth_quant_args"]["folding"]
            and not inplace
        ):  # pragma: no cover
            _apply_pre_optimization(model, tune_cfg, sq_info, recover=True)

        with open(ipex_config_path, "r") as f:
            model.tune_cfg = json.load(f)
        model.ipex_config_path = ipex_config_path
        dump_model_op_stats(tune_cfg["op"])

        from neural_compressor.torch.algorithms.smooth_quant import save

        logger.info("Smooth quantization done.")
        model.ori_save = model.save
        model.save = MethodType(save, model)
        return model


def qdq_quantize(
    model, tune_cfg, run_fn, example_inputs, inplace, cfgs, op_infos_from_cfgs, output_tensor_id_op_name, sq
):
    """Executes the smooth quantize process.

    Args:
        model (torch.nn.Module): raw fp32 model or prepared model.
        tune_cfg (OrderedDict): quantization config for ops.
        run_fn (Callable): a calibration function for calibrating the model.
        example_inputs (tensor/tuple/dict): used to trace torch model.
        inplace (bool): whether to carry out model transformations in-place. Defaults to True.
        cfgs (dict): configs loaded from ipex config path.
        op_infos_from_cfgs (dict): dict containing configs that have been parsed for each op.
        output_tensor_id_op_name (dict): dict containing op names corresponding to 'op_infos_from_cfgs'.
        sq (TorchSmoothQuant): TorchSmoothQuant class containing sq infos.

    Returns:
        A quantized model.
    """
    smoothquant_scale_info = sq.sq_scale_info
    sq_minmax_init = True if tune_cfg.get("act_algo", "kl") == "minmax" else False

    use_bf16 = tune_cfg.get("use_bf16", None)

    # Check save_qconf_summary part is a workaround for IPEX bug.
    # Sometimes the prepared model from get_op_capablitiy loss this attribute
    if not hasattr(model, "save_qconf_summary") or not hasattr(model, "load_qconf_summary"):  # pragma: no cover
        from torch.ao.quantization.observer import MinMaxObserver

        if ipex_ver.release >= Version("2.1.1").release:
            static_qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(alpha=0.5, act_observer=MinMaxObserver)
        else:  # pragma: no cover
            if sq_minmax_init:
                static_qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(
                    alpha=0.5, act_observer=MinMaxObserver()
                )
                logger.warning(
                    "The int8 model accuracy will be close to 0 with MinMaxobserver, "
                    + "the suggested IPEX version is higher or equal than 2.1.100+cpu."
                )
            else:
                static_qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(alpha=0.5)
        if isinstance(example_inputs, dict):
            model = ipex.quantization.prepare(
                model, static_qconfig, example_kwarg_inputs=example_inputs, inplace=inplace
            )
        else:
            model = ipex.quantization.prepare(model, static_qconfig, example_inputs=example_inputs, inplace=inplace)

    # The IPEX SmoothQuant observer can only use save/load_qconf_summary once.
    # The save_qconf_summary API will freeze the scale used in model and calibration won't work anymore.
    # The load_qconf_summary will overwrite the scales used in model but only work in the first call.
    # Here, we use INC collected scale for Linear and set normal observer instead of SQObserver \
    # to make sure calibration works for other ops, like add, bmm.
    cfg_to_qconfig(tune_cfg, cfgs, op_infos_from_cfgs, output_tensor_id_op_name, smooth_quant=True)
    update_sq_scale(ipex_config_path, smoothquant_scale_info)
    model.load_qconf_summary(qconf_summary=ipex_config_path)
    # real calibration for other operators
    try:
        # IPEX may raise an error on the second iteration.
        # OverflowError: cannot convert float infinity to integer
        run_fn(model)
    except:  # pragma: no cover
        logger.warning(
            "The calibration failed when calibrating with ipex, "
            + "using scale info from SmoothQuant for Linear and "
            + "one iter calibration for other ops."
        )

    model.save_qconf_summary(qconf_summary=ipex_config_path)
    if ipex_ver.release > Version("2.1.0").release:
        update_sq_scale(ipex_config_path, smoothquant_scale_info)
        model.load_qconf_summary(qconf_summary=ipex_config_path)
    model = _ipex_post_quant_process(model, example_inputs, use_bf16, inplace=inplace)

    with open(ipex_config_path, "r") as f:
        model.tune_cfg = json.load(f)
    model.ipex_config_path = ipex_config_path
    dump_model_op_stats(tune_cfg["op"])

    from neural_compressor.torch.algorithms.smooth_quant import save

    logger.info("Smooth quantization done.")
    model.ori_save = model.save
    model.save = MethodType(save, model)
    return model


def _apply_pre_optimization(model, tune_cfg, sq, recover=False):
    """Retrieves sq info to absorb the scale to the layer at output channel.

    Args:
        model (QuantizationInterceptionModule): a prepared model.
        tune_cfg (OrderedDict): quantization config for ops.
        sq (TorchSmoothQuant): TorchSmoothQuant class containing sq infos.
        recover (bool, optional): whether to recover the scale. Defaults to False.
    """
    sq_max_info = {}
    if sq.record_max_info:
        sq_max_info = sq.max_value_info
    if sq_max_info:
        tsq = TorchSmoothQuant(model, None)
        alpha = tune_cfg["recipe_cfgs"]["smooth_quant_args"]["alpha"]
        for op_name, info in sq_max_info.items():
            if alpha == "auto":  # pragma: no cover
                alpha = info["alpha"]
            absorb_layer = op_name
            absorbed_layer = info["absorbed_layer"]
            input_minmax = info["input_minmax"]
            weight_max = info["weight_max"]
            if sq.weight_clip:
                weight_max = weight_max.clamp(min=1e-5)
            abs_input_max = torch.max(torch.abs(input_minmax[0]), torch.abs(input_minmax[1]))
            input_power = torch.pow(abs_input_max, alpha)
            weight_power = torch.pow(weight_max, 1 - alpha)
            scale = torch.clip(input_power / weight_power, min=1e-5)
            with torch.no_grad():
                if recover:
                    scale = 1.0 / scale
                for layer in absorbed_layer:
                    tsq._scale_layer_weight(layer, scale)
                tsq._absorb_scales(absorb_layer, 1.0 / scale)
            logger.debug(f"Current smoothquant scale of {op_name} is {scale}, alpha is {alpha}")


def _ipex_post_quant_process(model, example_inputs, use_bf16, inplace=False):
    """Converts to a jit model.

    Args:
        model (QuantizationInterceptionModule): a prepared model.
        example_inputs (tensor/tuple/dict): used to trace torch model.
        use_bf16 (bool): whether to use bf16 for mixed precision.
        inplace (bool, optional): whether to carry out model transformations in-place. Defaults to True.

    Returns:
        A converted jit model.
    """
    if use_bf16 and (CpuInfo().bf16 or os.getenv("FORCE_BF16") == "1"):  # pragma: no cover
        with torch.no_grad():
            with torch.cpu.amp.autocast():
                model = ipex.quantization.convert(model, inplace=inplace)
                try:
                    if isinstance(example_inputs, dict):
                        model = torch.jit.trace(model, example_kwarg_inputs=example_inputs)
                    else:
                        model = torch.jit.trace(model, example_inputs)
                    model = torch.jit.freeze(model.eval())
                except:
                    if isinstance(example_inputs, dict):
                        model = torch.jit.trace(
                            model, example_kwarg_inputs=example_inputs, strict=False, check_trace=False
                        )
                    else:
                        model = torch.jit.trace(model, example_inputs, strict=False)
                    model = torch.jit.freeze(model.eval())
    else:
        model = ipex.quantization.convert(model, inplace=inplace)
        with torch.no_grad():
            try:
                if isinstance(example_inputs, dict):
                    model = torch.jit.trace(model, example_kwarg_inputs=example_inputs)
                else:
                    model = torch.jit.trace(model, example_inputs)
                model = torch.jit.freeze(model.eval())
            except:  # pragma: no cover
                if isinstance(example_inputs, dict):
                    model = torch.jit.trace(model, example_kwarg_inputs=example_inputs, strict=False, check_trace=False)
                else:
                    model = torch.jit.trace(model, example_inputs, strict=False)
                model = torch.jit.freeze(model.eval())

    # After freezing, run 1 time to warm up the profiling graph executor to insert prim::profile
    # At the 2nd run, the llga pass will be triggered and the model is turned into
    # an int8 model: prim::profile will be removed and will have LlgaFusionGroup in the graph
    simple_inference(model, example_inputs, iterations=2)
    return model
