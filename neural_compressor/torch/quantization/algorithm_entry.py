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

from copy import deepcopy
from types import MethodType
from typing import Any, Callable, Dict, Tuple

import torch

from neural_compressor.common.utils import AUTOROUND, AWQ, FP8_QUANT, GPTQ, HQQ, RTN, SMOOTH_QUANT, STATIC_QUANT, TEQ
from neural_compressor.torch.quantization import (
    AutoRoundConfig,
    AWQConfig,
    FP8Config,
    GPTQConfig,
    HQQConfig,
    RTNConfig,
    SmoothQuantConfig,
    StaticQuantConfig,
    TEQConfig,
)
from neural_compressor.torch.utils import logger, register_algo


###################### RTN Algo Entry ##################################
@register_algo(RTN)
@torch.no_grad()
def rtn_entry(
    model: torch.nn.Module, configs_mapping: Dict[Tuple[str, callable], RTNConfig], *args, **kwargs
) -> torch.nn.Module:
    """The main entry to apply rtn quantization."""
    from neural_compressor.torch.algorithms.weight_only import rtn_quantize

    # rebuild weight_config for rtn_quantize function
    weight_config = {}
    for (op_name, op_type), quant_config in configs_mapping.items():
        if quant_config.name != RTN:
            continue
        weight_config[op_name] = {
            "dtype": quant_config.dtype,
            "bits": quant_config.bits,
            "scheme": "sym" if quant_config.use_sym else "asym",
            "group_size": quant_config.group_size,
            "group_dim": quant_config.group_dim,
            "use_full_range": quant_config.use_full_range,
            "use_mse_search": quant_config.use_mse_search,
            "use_layer_wise": quant_config.use_layer_wise,
            "export_compressed_model": quant_config.export_compressed_model,
            "use_double_quant": quant_config.use_double_quant,
            "double_quant_dtype": quant_config.double_quant_dtype,
            "double_quant_bits": quant_config.double_quant_bits,
            "double_quant_scheme": "sym" if quant_config.double_quant_use_sym else "asym",
            "double_quant_group_size": quant_config.double_quant_group_size,
        }

    model = rtn_quantize(model, weight_config=weight_config)
    return model


###################### GPTQ Algo Entry ##################################
@register_algo(GPTQ)
@torch.no_grad()
def gptq_entry(
    model: torch.nn.Module, configs_mapping: Dict[Tuple[str, callable], GPTQConfig], *args, **kwargs
) -> torch.nn.Module:
    logger.info("Quantize model with the GPTQ algorithm.")
    from neural_compressor.torch.algorithms.weight_only import gptq_quantize

    # rebuild weight_config for gptq_quantize function
    weight_config = {}
    for (op_name, op_type), quant_config in configs_mapping.items():
        if quant_config.name != GPTQ:
            continue
        weight_config[op_name] = {
            "dtype": quant_config.dtype,
            "bits": quant_config.bits,
            "sym": quant_config.use_sym,
            "group_size": quant_config.group_size,
            "mse": quant_config.use_mse_search,
            "use_double_quant": quant_config.use_double_quant,
            "double_quant_dtype": quant_config.double_quant_dtype,
            "double_quant_bits": quant_config.double_quant_bits,
            "double_quant_sym": quant_config.double_quant_use_sym,
            "double_quant_group_size": quant_config.double_quant_group_size,
            "act_order": quant_config.act_order,
            "percdamp": quant_config.percdamp,
            "block_size": quant_config.block_size,
            "static_groups": quant_config.static_groups,
        }
    kwargs.update(
        {
            "export_compressed_model": quant_config.export_compressed_model,
            "use_layer_wise": quant_config.use_layer_wise,
            "model_path": quant_config.model_path,
        }
    )
    kwargs.pop("example_inputs")

    logger.warning("lm_head in transformer model is skipped by GPTQ")
    model, quantization_perm = gptq_quantize(model=model, weight_config=weight_config, *args, **kwargs)
    # Assign the gptq config as an attribute of model
    model._gptq_quantization_perm = quantization_perm
    return model


###################### Static Quant Algo Entry ##################################
@register_algo(name=STATIC_QUANT)
@torch.no_grad()
def static_quant_entry(
    model: torch.nn.Module, configs_mapping: Dict[Tuple[str, callable], StaticQuantConfig], *args, **kwargs
) -> torch.nn.Module:
    logger.info("Quantize model with the static quant algorithm.")
    from neural_compressor.torch.algorithms.static_quant import save, static_quantize

    # convert the user config into internal format
    quant_config_mapping = {}
    cfgs = deepcopy(configs_mapping)
    quant_config_mapping["op"] = cfgs
    for (op_name, op_type), cfg in cfgs.items():
        if cfg.name != STATIC_QUANT:
            continue
        quant_config_mapping["op"][(op_name, op_type)] = {
            "weight": {
                "dtype": cfg.w_dtype,
                "scheme": "sym",
                "granularity": cfg.w_granularity,
                "algorithm": cfg.w_algo,
            },
            "activation": {
                "dtype": cfg.act_dtype,
                "scheme": "sym" if cfg.act_sym else "asym",
                "granularity": cfg.act_granularity,
                "algorithm": cfg.act_algo,
            },
        }

    run_fn = kwargs.get("run_fn", None)
    example_inputs = kwargs.get("example_inputs", None)
    inplace = kwargs.get("inplace", True)
    assert example_inputs is not None, "Please provide example_inputs for static quantization."
    q_model = static_quantize(
        model=model,
        tune_cfg=quant_config_mapping,
        run_fn=run_fn,
        example_inputs=example_inputs,
        inplace=inplace,
    )
    logger.info("Static quantization done.")
    q_model.ori_save = q_model.save
    q_model.save = MethodType(save, q_model)
    return q_model


###################### Smooth Quant Algo Entry ##################################
@register_algo(name=SMOOTH_QUANT)
@torch.no_grad()
def smooth_quant_entry(
    model: torch.nn.Module, configs_mapping: Dict[Tuple[str, callable], SmoothQuantConfig], *args, **kwargs
) -> torch.nn.Module:
    logger.info("Quantize model with the smooth quant algorithm.")
    from neural_compressor.torch.algorithms.smooth_quant import save, smooth_quantize

    # convert the user config into internal format
    quant_config_mapping = {}
    cfgs = deepcopy(configs_mapping)
    quant_config_mapping["op"] = cfgs
    for (op_name, op_type), cfg in cfgs.items():
        quant_config_mapping["op"][(op_name, op_type)] = {
            "weight": {
                "dtype": cfg.w_dtype,
                "scheme": "sym",
                "granularity": cfg.w_granularity,
                "algorithm": cfg.w_algo,
            },
            "activation": {
                "dtype": cfg.act_dtype,
                "scheme": "sym" if cfg.act_sym else "asym",
                "granularity": cfg.act_granularity,
                "algorithm": cfg.act_algo,
            },
        }
        quant_config_mapping["recipe_cfgs"] = {
            "smooth_quant": True,
            "smooth_quant_args": {
                "alpha": cfg.alpha,
                "folding": cfg.folding,
                "scale_sharing": cfg.scale_sharing,
                "auto_alpha_args": cfg.auto_alpha_args if cfg.auto_alpha_args is not None else {},
            },
            "layer_wise_quant_args": {},
            "first_conv_or_matmul_quantization": True,
            "last_conv_or_matmul_quantization": True,
            "pre_post_process_quantization": True,
        }

    run_fn = kwargs.get("run_fn", None)
    example_inputs = kwargs.get("example_inputs", None)
    inplace = kwargs.get("inplace", True)
    assert example_inputs is not None, "Please provide example_inputs for smooth quantization."
    q_model = smooth_quantize(
        model=model,
        tune_cfg=quant_config_mapping,
        run_fn=run_fn,
        example_inputs=example_inputs,
        inplace=inplace,
    )
    logger.info("Smooth quantization done.")
    q_model.ori_save = q_model.save
    q_model.save = MethodType(save, q_model)
    return q_model


###################### AWQ Algo Entry ##################################
@register_algo(name=AWQ)
@torch.no_grad()
def awq_quantize_entry(
    model: torch.nn.Module, configs_mapping: Dict[Tuple[str, callable], AWQConfig], *args, **kwargs
) -> torch.nn.Module:
    logger.info("Quantize model with the AWQ algorithm.")
    from neural_compressor.torch.algorithms.weight_only import awq_quantize

    weight_config = {}
    for (op_name, op_type), op_config in configs_mapping.items():
        if op_config.name != AWQ:
            continue
        if op_config.dtype == "fp32":
            weight_config[op_name] = {
                "bits": -1,
                "dtype": "fp32",  # skip quantization
                "group_size": 128,
                "scheme": "asym",
            }
        else:
            weight_config[op_name] = {
                "dtype": op_config.dtype,
                "bits": op_config.bits,
                "group_size": op_config.group_size,
                "group_dim": op_config.group_dim,
                "scheme": "sym" if op_config.use_sym else "asym",
                "use_full_range": op_config.use_full_range,
                "use_mse_search": op_config.use_mse_search,
                "use_layer_wise": op_config.use_layer_wise,
                "export_compressed_model": op_config.export_compressed_model,
                "use_double_quant": op_config.use_double_quant,
                "double_quant_dtype": op_config.double_quant_dtype,
                "double_quant_bits": op_config.double_quant_bits,
                "double_quant_scheme": op_config.double_quant_use_sym,
                "double_quant_group_size": op_config.double_quant_group_size,
            }
            use_auto_scale = op_config.use_auto_scale
            use_mse_search = op_config.use_auto_clip  # for awq clip
            folding = op_config.folding
            return_int = op_config.export_compressed_model
            use_full_range = op_config.use_full_range

    calib_func = kwargs.get("run_fn", None)
    example_inputs = kwargs.get("example_inputs", None)
    assert example_inputs is not None, "Please provide example_inputs for AWQ quantization."
    model = awq_quantize(
        model,
        bits=-1,  # no quantize for op not in weight_config
        example_inputs=example_inputs,  # must be required
        calib_func=calib_func,
        weight_config=weight_config,
        use_auto_scale=use_auto_scale,
        use_mse_search=use_mse_search,
        folding=folding,
        return_int=return_int,
        use_full_range=use_full_range,
    )
    logger.info("AWQ quantization done.")
    return model


###################### TEQ Algo Entry ##################################
@register_algo(name=TEQ)
def teq_quantize_entry(
    model: torch.nn.Module, configs_mapping: Dict[Tuple[str, callable], TEQConfig], *args, **kwargs
) -> torch.nn.Module:
    from neural_compressor.torch.algorithms.weight_only import teq_quantize

    logger.info("Quantize model with the TEQ algorithm.")
    weight_config = {}
    absorb_to_layer = {}
    example_inputs = kwargs.get("example_inputs", None)
    assert example_inputs is not None, "Please provide example_inputs for TEQ quantization."
    calib_func = kwargs.get("run_fn", None)
    folding = True
    for (op_name, op_type), quant_config in configs_mapping.items():
        if quant_config.dtype == "fp32":
            continue
        else:
            weight_config[op_name] = {
                "dtype": quant_config.dtype,
                "bits": quant_config.bits,
                "scheme": "sym" if quant_config.use_sym else "asym",
                "group_size": quant_config.group_size,
                "group_dim": quant_config.group_dim,
                "use_full_range": quant_config.use_full_range,
                "use_mse_search": quant_config.use_mse_search,
                "use_layer_wise": quant_config.use_layer_wise,
                "export_compressed_model": quant_config.export_compressed_model,
                "use_double_quant": quant_config.use_double_quant,
                "double_quant_dtype": quant_config.double_quant_dtype,
                "double_quant_bits": quant_config.double_quant_bits,
                "double_quant_scheme": "sym" if quant_config.double_quant_use_sym else "asym",
                "double_quant_group_size": quant_config.double_quant_group_size,
            }
            absorb_to_layer = quant_config.absorb_to_layer
            folding = quant_config.folding
    assert isinstance(model, torch.nn.Module), "only support torch module"

    model = teq_quantize(
        model,
        example_inputs=example_inputs,
        folding=folding,
        absorb_to_layer=absorb_to_layer,
        calib_func=calib_func,
        weight_config=weight_config,
    )
    logger.info("TEQ quantization done.")
    return model


###################### AUTOROUND Algo Entry ##################################
@register_algo(name=AUTOROUND)
def autoround_quantize_entry(
    model: torch.nn.Module, configs_mapping: Dict[Tuple[str, callable], AutoRoundConfig], *args, **kwargs
) -> torch.nn.Module:
    from neural_compressor.torch.algorithms.weight_only import autoround_quantize

    logger.info("Quantize model with the AutoRound algorithm.")
    calib_func = kwargs.get("run_fn", None)
    weight_config = {}
    for (op_name, op_type), quant_config in configs_mapping.items():
        if quant_config.name != AUTOROUND or quant_config.dtype == "fp32":
            continue
        else:
            weight_config[op_name] = {
                "data_type": quant_config.dtype,
                "bits": quant_config.bits,
                "sym": quant_config.use_sym,
                "group_size": quant_config.group_size,
            }
            enable_full_range = quant_config.enable_full_range
            batch_size = quant_config.batch_size
            lr_scheduler = quant_config.lr_scheduler
            use_quant_input = quant_config.use_quant_input
            enable_minmax_tuning = quant_config.enable_minmax_tuning
            lr = quant_config.lr
            minmax_lr = quant_config.minmax_lr
            low_gpu_mem_usage = quant_config.low_gpu_mem_usage
            iters = quant_config.iters
            seqlen = quant_config.seqlen
            n_samples = quant_config.n_samples
            sampler = quant_config.sampler
            seed = quant_config.seed
            n_blocks = quant_config.n_blocks
            gradient_accumulate_steps = quant_config.gradient_accumulate_steps
            not_use_best_mse = quant_config.not_use_best_mse
            dynamic_max_gap = quant_config.dynamic_max_gap
            scale_dtype = quant_config.scale_dtype

    kwargs.pop("example_inputs")
    model, autoround_config = autoround_quantize(
        model=model,
        weight_config=weight_config,
        enable_full_range=enable_full_range,
        batch_size=batch_size,
        lr_scheduler=lr_scheduler,
        use_quant_input=use_quant_input,
        enable_minmax_tuning=enable_minmax_tuning,
        lr=lr,
        minmax_lr=minmax_lr,
        low_gpu_mem_usage=low_gpu_mem_usage,
        iters=iters,
        seqlen=seqlen,
        n_samples=n_samples,
        sampler=sampler,
        seed=seed,
        n_blocks=n_blocks,
        gradient_accumulate_steps=gradient_accumulate_steps,
        not_use_best_mse=not_use_best_mse,
        dynamic_max_gap=dynamic_max_gap,
        scale_dtype=scale_dtype,
        **kwargs
    )
    model.autoround_config = autoround_config
    logger.info("AutoRound quantization done.")
    return model


###################### HQQ Algo Entry ##################################
@register_algo(name=HQQ)
@torch.no_grad()
def hqq_entry(
    model: torch.nn.Module, configs_mapping: Dict[Tuple[str, Callable], HQQConfig], *args, **kwargs
) -> torch.nn.Module:
    from neural_compressor.torch.algorithms.weight_only import hqq_quantize

    logger.info("Quantize model with the HQQ algorithm.")
    q_model = hqq_quantize(model, configs_mapping)
    return q_model


###################### Habana FP8 Algo Entry ##################################
from neural_compressor.torch.utils import is_hpex_available

if is_hpex_available():
    from neural_compressor.torch.algorithms.habana_fp8 import quantize, save

    @register_algo(FP8_QUANT)
    def fp8_quant_entry(
        model: torch.nn.Module, configs_mapping: Dict[Tuple[str], FP8Config], *args, **kwargs
    ) -> torch.nn.Module:
        kwargs.pop("example_inputs")
        model = quantize(model, configs_mapping, *args, **kwargs)
        model.qconfig = configs_mapping
        model.save = MethodType(save, model)
        return model
