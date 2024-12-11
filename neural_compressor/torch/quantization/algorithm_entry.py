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
"""Intel Neural Compressor PyTorch supported algorithm entries."""

from copy import deepcopy
from types import MethodType
from typing import Callable, Dict, Tuple

import torch

from neural_compressor.common.utils import (
    AUTOROUND,
    AWQ,
    FP8_QUANT,
    GPTQ,
    HQQ,
    MIXED_PRECISION,
    MX_QUANT,
    RTN,
    SMOOTH_QUANT,
    STATIC_QUANT,
    TEQ,
    Mode,
)
from neural_compressor.torch.quantization import (
    AutoRoundConfig,
    AWQConfig,
    FP8Config,
    GPTQConfig,
    HQQConfig,
    MixedPrecisionConfig,
    MXQuantConfig,
    RTNConfig,
    SmoothQuantConfig,
    StaticQuantConfig,
    TEQConfig,
)
from neural_compressor.torch.utils import (
    dump_model_op_stats,
    get_quantizer,
    is_ipex_imported,
    logger,
    postprocess_model,
    register_algo,
)
from neural_compressor.torch.utils.constants import PT2E_DYNAMIC_QUANT, PT2E_STATIC_QUANT


###################### RTN Algo Entry ##################################
@register_algo(RTN)
@torch.no_grad()
def rtn_entry(
    model: torch.nn.Module,
    configs_mapping: Dict[Tuple[str, callable], RTNConfig],
    mode: Mode = Mode.QUANTIZE,
    *args,
    **kwargs,
) -> torch.nn.Module:
    """The main entry to apply rtn quantization.

    Args:
        model (torch.nn.Module): raw fp32 model or prepared model.
        configs_mapping (Dict[Tuple[str, callable], RTNConfig]): per-op configuration.
        mode (Mode, optional): select from [PREPARE, CONVERT and QUANTIZE]. Defaults to Mode.QUANTIZE.

    Returns:
        torch.nn.Module: prepared model or quantized model.
    """
    from neural_compressor.torch.algorithms.weight_only.rtn import RTNQuantizer
    from neural_compressor.torch.algorithms.weight_only.save_load import save

    # rebuild weight_config for RTNQuantizer class
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
            "use_double_quant": quant_config.use_double_quant,
            "double_quant_dtype": quant_config.double_quant_dtype,
            "double_quant_bits": quant_config.double_quant_bits,
            "double_quant_scheme": "sym" if quant_config.double_quant_use_sym else "asym",
            "double_quant_group_size": quant_config.double_quant_group_size,
        }
    kwargs.update(
        {
            "use_layer_wise": quant_config.use_layer_wise,
            "model_path": quant_config.model_path,
            "quant_lm_head": quant_config.quant_lm_head,
        }
    )
    quantizer = get_quantizer(model, quantizer_cls=RTNQuantizer, quant_config=weight_config)
    model = quantizer.execute(model, mode=mode, *args, **kwargs)
    model.qconfig = configs_mapping
    model.save = MethodType(save, model)
    postprocess_model(model, mode, quantizer)
    dump_model_op_stats(mode, configs_mapping)
    return model


###################### GPTQ Algo Entry ##################################
@register_algo(GPTQ)
@torch.no_grad()
def gptq_entry(
    model: torch.nn.Module,
    configs_mapping: Dict[Tuple[str, callable], GPTQConfig],
    mode: Mode = Mode.QUANTIZE,
    *args,
    **kwargs,
) -> torch.nn.Module:
    """The main entry to apply gptq quantization.

    Args:
        model (torch.nn.Module): raw fp32 model or prepared model.
        configs_mapping (Dict[Tuple[str, callable], GPTQConfig]): per-op configuration.
        mode (Mode, optional): select from [PREPARE, CONVERT and QUANTIZE]. Defaults to Mode.QUANTIZE.

    Returns:
        torch.nn.Module: prepared model or quantized model.
    """
    logger.info("Quantize model with the GPTQ algorithm.")
    from neural_compressor.torch.algorithms.weight_only.gptq import GPTQuantizer
    from neural_compressor.torch.algorithms.weight_only.save_load import save

    # rebuild weight_config for gptq_quantize function
    weight_config = {}
    for (op_name, op_type), quant_config in configs_mapping.items():
        if quant_config.name != GPTQ or quant_config.dtype == "fp32":
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
            "true_sequential": quant_config.true_sequential,
        }
    kwargs.update(
        {
            "use_layer_wise": quant_config.use_layer_wise,
            "model_path": quant_config.model_path,
            "quant_lm_head": quant_config.quant_lm_head,
        }
    )
    kwargs.pop("example_inputs")
    logger.warning("lm_head in transformer model is skipped by GPTQ")

    quantizer = get_quantizer(model, quantizer_cls=GPTQuantizer, quant_config=weight_config)
    model = quantizer.execute(model, mode=mode, *args, **kwargs)
    model.qconfig = configs_mapping
    model.save = MethodType(save, model)
    postprocess_model(model, mode, quantizer)
    dump_model_op_stats(mode, configs_mapping)

    return model


###################### Static Quant Algo Entry ##################################
@register_algo(name=STATIC_QUANT)
@torch.no_grad()
def static_quant_entry(
    model: torch.nn.Module,
    configs_mapping: Dict[Tuple[str, callable], StaticQuantConfig],
    mode: Mode = Mode.QUANTIZE,
    *args,
    **kwargs,
) -> torch.nn.Module:
    """The main entry to apply static quantization, includes pt2e quantization and ipex quantization.

    Args:
        model (torch.nn.Module): raw fp32 model or prepared model.
        configs_mapping (Dict[Tuple[str, callable], StaticQuantConfig]): per-op configuration.
        mode (Mode, optional): select from [PREPARE, CONVERT and QUANTIZE]. Defaults to Mode.QUANTIZE.

    Returns:
        torch.nn.Module: prepared model or quantized model.
    """
    if not is_ipex_imported():
        return pt2e_static_quant_entry(model, configs_mapping, mode, *args, **kwargs)
    logger.info("Quantize model with the static quant algorithm.")
    from neural_compressor.torch.algorithms.static_quant import StaticQuantQuantizer

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
        quant_config_mapping["use_bf16"] = "bf16" not in cfg.excluded_precisions

    run_fn = kwargs.get("run_fn", None)
    example_inputs = kwargs.get("example_inputs", None)
    inplace = kwargs.get("inplace", True)
    assert example_inputs is not None, "Please provide example_inputs for static quantization."

    quantizer = get_quantizer(model, quantizer_cls=StaticQuantQuantizer, quant_config=quant_config_mapping)
    model = quantizer.execute(model, mode=mode, run_fn=run_fn, example_inputs=example_inputs, inplace=inplace)
    postprocess_model(model, mode, quantizer)

    return model


###################### PT2E Dynamic Quant Algo Entry ##################################
@register_algo(name=PT2E_DYNAMIC_QUANT)
@torch.no_grad()
def pt2e_dynamic_quant_entry(
    model: torch.nn.Module,
    configs_mapping,
    mode: Mode,
    *args,
    **kwargs,
) -> torch.nn.Module:
    """The main entry to apply pt2e dynamic quantization.

    Args:
        model (torch.nn.Module): raw fp32 model or prepared model.
        configs_mapping: per-op configuration.
        mode (Mode, optional): select from [PREPARE, CONVERT and QUANTIZE]. Defaults to Mode.QUANTIZE.

    Returns:
        torch.nn.Module: prepared model or quantized model.
    """
    logger.info("Quantize model with the PT2E static quant algorithm.")
    from neural_compressor.torch.algorithms.pt2e_quant.core import W8A8PT2EQuantizer
    from neural_compressor.torch.algorithms.pt2e_quant.save_load import save

    run_fn = kwargs.get("run_fn", None)
    example_inputs = kwargs.get("example_inputs", None)
    inplace = kwargs.get("inplace", True)
    dynamic_shapes = model.dynamic_shapes
    W8A8PT2EQuantizer.is_dynamic = True
    for _, quant_config in configs_mapping.items():
        if quant_config.name == PT2E_DYNAMIC_QUANT:
            w8a8_quantizer = W8A8PT2EQuantizer(quant_config=quant_config)
            model = w8a8_quantizer.execute(
                model, mode=mode, run_fn=run_fn, example_inputs=example_inputs, inplace=inplace
            )
            model.dynamic_shapes = dynamic_shapes
            model.qconfig = configs_mapping
            model.save = MethodType(save, model)
            return model


###################### PT2E Static Quant Algo Entry ##################################
@register_algo(name=PT2E_STATIC_QUANT)
@torch.no_grad()
def pt2e_static_quant_entry(
    model: torch.nn.Module,
    configs_mapping,
    mode: Mode,
    *args,
    **kwargs,
) -> torch.nn.Module:
    """The main entry to apply pt2e static quantization.

    Args:
        model (torch.nn.Module): raw fp32 model or prepared model.
        configs_mapping: per-op configuration.
        mode (Mode, optional): select from [PREPARE, CONVERT and QUANTIZE]. Defaults to Mode.QUANTIZE.

    Returns:
        torch.nn.Module: prepared model or quantized model.
    """
    logger.info("Quantize model with the PT2E static quant algorithm.")
    from neural_compressor.torch.algorithms.pt2e_quant.core import W8A8PT2EQuantizer
    from neural_compressor.torch.algorithms.pt2e_quant.save_load import save

    run_fn = kwargs.get("run_fn", None)
    example_inputs = kwargs.get("example_inputs", None)
    inplace = kwargs.get("inplace", True)
    dynamic_shapes = model.dynamic_shapes
    for _, quant_config in configs_mapping.items():
        if quant_config.name == STATIC_QUANT:
            w8a8_quantizer = W8A8PT2EQuantizer(quant_config=quant_config)
            model = w8a8_quantizer.execute(
                model, mode=mode, run_fn=run_fn, example_inputs=example_inputs, inplace=inplace
            )
            model.dynamic_shapes = dynamic_shapes
            model.qconfig = configs_mapping
            model.save = MethodType(save, model)
            return model


###################### Smooth Quant Algo Entry ##################################
@register_algo(name=SMOOTH_QUANT)
@torch.no_grad()
def smooth_quant_entry(
    model: torch.nn.Module,
    configs_mapping: Dict[Tuple[str, callable], SmoothQuantConfig],
    mode: Mode = Mode.QUANTIZE,
    *args,
    **kwargs,
) -> torch.nn.Module:
    """The main entry to apply smooth quantization.

    Args:
        model (torch.nn.Module): raw fp32 model or prepared model.
        configs_mapping (Dict[Tuple[str, callable], SmoothQuantConfig]): per-op configuration.
        mode (Mode, optional): select from [PREPARE, CONVERT and QUANTIZE]. Defaults to Mode.QUANTIZE.

    Returns:
        torch.nn.Module: prepared model or quantized model.
    """
    logger.info("Quantize model with the smooth quant algorithm.")
    from neural_compressor.torch.algorithms.smooth_quant import SmoothQuantQuantizer, TorchSmoothQuant

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
        quant_config_mapping["use_bf16"] = "bf16" not in cfg.excluded_precisions

    run_fn = kwargs.get("run_fn", None)
    example_inputs = kwargs.get("example_inputs", None)
    inplace = kwargs.get("inplace", True)
    assert example_inputs is not None, "Please provide example_inputs for smooth quantization."

    quantizer = get_quantizer(model, quantizer_cls=SmoothQuantQuantizer, quant_config=quant_config_mapping)
    model = quantizer.execute(model, mode=mode, run_fn=run_fn, example_inputs=example_inputs, inplace=inplace)
    postprocess_model(model, mode, quantizer)

    return model


###################### AWQ Algo Entry ##################################
@register_algo(name=AWQ)
@torch.no_grad()
def awq_quantize_entry(
    model: torch.nn.Module,
    configs_mapping: Dict[Tuple[str, callable], AWQConfig],
    mode: Mode = Mode.QUANTIZE,
    *args,
    **kwargs,
) -> torch.nn.Module:
    """The main entry to apply AWQ quantization.

    Args:
        model (torch.nn.Module): raw fp32 model or prepared model.
        configs_mapping (Dict[Tuple[str, callable], AWQConfig]): per-op configuration.
        mode (Mode, optional): select from [PREPARE, CONVERT and QUANTIZE]. Defaults to Mode.QUANTIZE.

    Returns:
        torch.nn.Module: prepared model or quantized model.
    """
    logger.info("Quantize model with the AWQ algorithm.")
    from neural_compressor.torch.algorithms.weight_only.awq import AWQQuantizer
    from neural_compressor.torch.algorithms.weight_only.save_load import save

    weight_config = {}
    for (op_name, op_type), quant_config in configs_mapping.items():
        if quant_config.name != AWQ:
            continue
        if quant_config.dtype == "fp32":
            weight_config[op_name] = {
                "bits": -1,
                "dtype": "fp32",  # skip quantization
                "group_size": 128,
                "scheme": "asym",
            }
        else:
            weight_config[op_name] = {
                "dtype": quant_config.dtype,
                "bits": quant_config.bits,
                "group_size": quant_config.group_size,
                "group_dim": quant_config.group_dim,
                "scheme": "sym" if quant_config.use_sym else "asym",
                "use_full_range": quant_config.use_full_range,
                "use_mse_search": quant_config.use_mse_search,
                "use_layer_wise": quant_config.use_layer_wise,
                "use_double_quant": quant_config.use_double_quant,
                "double_quant_dtype": quant_config.double_quant_dtype,
                "double_quant_bits": quant_config.double_quant_bits,
                "double_quant_scheme": quant_config.double_quant_use_sym,
                "double_quant_group_size": quant_config.double_quant_group_size,
            }
            use_auto_scale = quant_config.use_auto_scale
            use_mse_search = quant_config.use_auto_clip  # for awq clip
            folding = quant_config.folding
            use_full_range = quant_config.use_full_range
            absorb_layer_dict = quant_config.absorb_layer_dict

    run_fn = kwargs.get("run_fn", None)
    run_args = kwargs.get("run_args", None)
    example_inputs = kwargs.get("example_inputs", None)
    assert example_inputs is not None, "Please provide example_inputs for AWQ quantization."

    quantizer = get_quantizer(
        model, quantizer_cls=AWQQuantizer, quant_config=weight_config, absorb_layer_dict=absorb_layer_dict
    )
    model = quantizer.execute(
        model,
        mode=mode,
        bits=-1,  # no quantize for op not in weight_config
        example_inputs=example_inputs,  # must be required
        run_fn=run_fn,
        run_args=run_args,
        use_auto_scale=use_auto_scale,
        use_mse_search=use_mse_search,
        folding=folding,
        use_full_range=use_full_range,
    )

    model.qconfig = configs_mapping
    model.save = MethodType(save, model)
    postprocess_model(model, mode, quantizer)
    dump_model_op_stats(mode, configs_mapping)
    return model


###################### TEQ Algo Entry ##################################
@register_algo(name=TEQ)
def teq_quantize_entry(
    model: torch.nn.Module,
    configs_mapping: Dict[Tuple[str, callable], TEQConfig],
    mode: Mode,
    *args,
    **kwargs,
) -> torch.nn.Module:
    """The main entry to apply TEQ quantization.

    Args:
        model (torch.nn.Module): raw fp32 model or prepared model.
        configs_mapping (Dict[Tuple[str, callable], TEQConfig]): per-op configuration.
        mode (Mode, optional): select from [PREPARE, CONVERT and QUANTIZE]. Defaults to Mode.QUANTIZE.

    Returns:
        torch.nn.Module: prepared model or quantized model.
    """
    from neural_compressor.torch.algorithms.weight_only.save_load import save
    from neural_compressor.torch.algorithms.weight_only.teq import TEQuantizer

    logger.info("Quantize model with the TEQ algorithm.")
    weight_config = {}
    absorb_to_layer = {}
    example_inputs = kwargs.get("example_inputs", None)
    assert example_inputs is not None, "Please provide example_inputs for TEQ quantization."
    run_fn = kwargs.get("run_fn", None)
    inplace = kwargs.get("inplace", True)
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
                "use_double_quant": quant_config.use_double_quant,
                "double_quant_dtype": quant_config.double_quant_dtype,
                "double_quant_bits": quant_config.double_quant_bits,
                "double_quant_scheme": "sym" if quant_config.double_quant_use_sym else "asym",
                "double_quant_group_size": quant_config.double_quant_group_size,
            }
            absorb_to_layer = quant_config.absorb_to_layer
            folding = quant_config.folding
    assert isinstance(model, torch.nn.Module), "only support torch module"

    quantizer = get_quantizer(
        model,
        quantizer_cls=TEQuantizer,
        quant_config=weight_config,
        folding=folding,
        absorb_to_layer=absorb_to_layer,
        example_inputs=example_inputs,
    )
    model = quantizer.execute(model, mode=mode, run_fn=run_fn, example_inputs=example_inputs, inplace=inplace)
    model.qconfig = configs_mapping
    model.save = MethodType(save, model)
    postprocess_model(model, mode, quantizer)
    dump_model_op_stats(mode, configs_mapping)

    return model


###################### AUTOROUND Algo Entry ##################################
@register_algo(name=AUTOROUND)
def autoround_quantize_entry(
    model: torch.nn.Module,
    configs_mapping: Dict[Tuple[str, callable], AutoRoundConfig],
    mode: Mode = Mode.QUANTIZE,
    *args,
    **kwargs,
) -> torch.nn.Module:
    """The main entry to apply AutoRound quantization.

    Args:
        model (torch.nn.Module): raw fp32 model or prepared model.
        configs_mapping (Dict[Tuple[str, callable], AutoRoundConfig]): per-op configuration.
        mode (Mode, optional): select from [PREPARE, CONVERT and QUANTIZE]. Defaults to Mode.QUANTIZE.

    Returns:
        torch.nn.Module: prepared model or quantized model.
    """
    from neural_compressor.torch.algorithms.weight_only.autoround import AutoRoundQuantizer
    from neural_compressor.torch.algorithms.weight_only.save_load import save

    logger.info("Quantize model with the AutoRound algorithm.")
    weight_config = {}
    for (op_name, op_type), quant_config in configs_mapping.items():
        if quant_config.name != AUTOROUND or quant_config.dtype == "fp32":
            continue
        else:
            dtype = quant_config.dtype
            bits = quant_config.bits
            if dtype != "int" and "int" in dtype:
                bits = int(dtype.lstrip("int"))
                dtype = "int"
            weight_config[op_name] = {
                "data_type": dtype,
                "bits": bits,
                "sym": quant_config.use_sym,
                "group_size": quant_config.group_size,
                "act_bits": quant_config.act_bits,
                "act_group_size": quant_config.act_group_size,
                "act_sym": quant_config.act_sym,
                "act_dynamic": quant_config.act_dynamic,
            }
            enable_full_range = quant_config.enable_full_range
            batch_size = quant_config.batch_size
            lr_scheduler = quant_config.lr_scheduler
            enable_quanted_input = quant_config.enable_quanted_input
            enable_minmax_tuning = quant_config.enable_minmax_tuning
            lr = quant_config.lr
            minmax_lr = quant_config.minmax_lr
            low_gpu_mem_usage = quant_config.low_gpu_mem_usage
            iters = quant_config.iters
            seqlen = quant_config.seqlen
            nsamples = quant_config.nsamples
            sampler = quant_config.sampler
            seed = quant_config.seed
            nblocks = quant_config.nblocks
            gradient_accumulate_steps = quant_config.gradient_accumulate_steps
            not_use_best_mse = quant_config.not_use_best_mse
            dynamic_max_gap = quant_config.dynamic_max_gap
            scale_dtype = quant_config.scale_dtype
            to_quant_block_names = quant_config.to_quant_block_names
            low_cpu_mem_usage = quant_config.use_layer_wise
            export_format = quant_config.export_format
            enable_norm_bias_tuning = quant_config.enable_norm_bias_tuning
            enable_torch_compile = quant_config.enable_torch_compile
            is_mllm = quant_config.is_mllm
            quant_nontext_module = quant_config.quant_nontext_module
            extra_data_dir = quant_config.extra_data_dir
            processor = quant_config.processor
            image_processor = quant_config.image_processor
            template = quant_config.template
            truncation = quant_config.truncation

    kwargs.pop("example_inputs")

    quantizer = get_quantizer(
        model,
        quantizer_cls=AutoRoundQuantizer,
        quant_config=weight_config,
        enable_full_range=enable_full_range,
        batch_size=batch_size,
        lr_scheduler=lr_scheduler,
        enable_quanted_input=enable_quanted_input,
        enable_minmax_tuning=enable_minmax_tuning,
        lr=lr,
        minmax_lr=minmax_lr,
        low_gpu_mem_usage=low_gpu_mem_usage,
        iters=iters,
        seqlen=seqlen,
        nsamples=nsamples,
        sampler=sampler,
        seed=seed,
        nblocks=nblocks,
        gradient_accumulate_steps=gradient_accumulate_steps,
        not_use_best_mse=not_use_best_mse,
        dynamic_max_gap=dynamic_max_gap,
        scale_dtype=scale_dtype,
        to_quant_block_names=to_quant_block_names,
        low_cpu_mem_usage=low_cpu_mem_usage,
        export_format=export_format,
        enable_norm_bias_tuning=enable_norm_bias_tuning,
        enable_torch_compile=enable_torch_compile,
        is_mllm=is_mllm,
        quant_nontext_module=quant_nontext_module,
        extra_data_dir=extra_data_dir,
        processor=processor,
        image_processor=image_processor,
        template=template,
        truncation=truncation,
    )
    model = quantizer.execute(model=model, mode=mode, *args, **kwargs)
    model.qconfig = configs_mapping
    model.save = MethodType(save, model)
    postprocess_model(model, mode, quantizer)
    dump_model_op_stats(mode, configs_mapping)
    return model


###################### HQQ Algo Entry ##################################
@register_algo(name=HQQ)
@torch.no_grad()
def hqq_entry(
    model: torch.nn.Module,
    configs_mapping: Dict[Tuple[str, Callable], HQQConfig],
    mode: Mode = Mode.QUANTIZE,
    *args,
    **kwargs,
) -> torch.nn.Module:
    """The main entry to apply AutoRound quantization.

    Args:
        model (torch.nn.Module): raw fp32 model or prepared model.
        configs_mapping (Dict[Tuple[str, callable], AutoRoundConfig]): per-op configuration.
        mode (Mode, optional): select from [PREPARE, CONVERT and QUANTIZE]. Defaults to Mode.QUANTIZE.

    Returns:
        torch.nn.Module: prepared model or quantized model.
    """
    from neural_compressor.torch.algorithms.weight_only.hqq import HQQuantizer
    from neural_compressor.torch.algorithms.weight_only.save_load import save

    logger.info("Quantize model with the HQQ algorithm.")

    quantizer = get_quantizer(model, quantizer_cls=HQQuantizer, quant_config=configs_mapping)
    model = quantizer.execute(model, mode=mode)
    model.qconfig = configs_mapping
    model.save = MethodType(save, model)
    postprocess_model(model, mode, quantizer)
    dump_model_op_stats(mode, configs_mapping)

    return model


###################### Habana FP8 Algo Entry ##################################
@register_algo(FP8_QUANT)
@torch.no_grad()
def fp8_entry(
    model: torch.nn.Module,
    configs_mapping: Dict[Tuple[str], FP8Config],
    mode: Mode = Mode.QUANTIZE,
    *args,
    **kwargs,
) -> torch.nn.Module:
    """The main entry to apply fp8 quantization."""
    from neural_compressor.torch.algorithms.fp8_quant import FP8Quantizer
    from neural_compressor.torch.algorithms.fp8_quant.save_load import save

    quantizer = get_quantizer(model, quantizer_cls=FP8Quantizer, quant_config=configs_mapping)
    model = quantizer.execute(model, mode=mode)
    model.qconfig = configs_mapping
    model.save = MethodType(save, model)
    postprocess_model(model, mode, quantizer)
    return model


###################### MX Quant Algo Entry ##################################
@register_algo(name=MX_QUANT)
@torch.no_grad()
def mx_quant_entry(
    model: torch.nn.Module,
    configs_mapping: Dict[Tuple[str, callable], MXQuantConfig],
    mode: Mode = Mode.QUANTIZE,
    *args,
    **kwargs,
) -> torch.nn.Module:
    """The main entry to apply AutoRound quantization.

    Args:
        model (torch.nn.Module): raw fp32 model or prepared model.
        configs_mapping (Dict[Tuple[str, callable], AutoRoundConfig]): per-op configuration.
        mode (Mode, optional): select from [PREPARE, CONVERT and QUANTIZE]. Defaults to Mode.QUANTIZE.

    Returns:
        torch.nn.Module: prepared model or quantized model.
    """
    logger.info("Quantize model with the mx quant algorithm.")
    from neural_compressor.torch.algorithms.mx_quant.mx import MXQuantizer

    quantizer = get_quantizer(model, quantizer_cls=MXQuantizer, quant_config=configs_mapping)
    model = quantizer.execute(model, mode=mode)
    model.qconfig = configs_mapping
    postprocess_model(model, mode, quantizer)

    return model


###################### Mixed Precision Algo Entry ##################################
@register_algo(MIXED_PRECISION)
def mixed_precision_entry(
    model: torch.nn.Module, configs_mapping: Dict[Tuple[str], MixedPrecisionConfig], *args, **kwargs
) -> torch.nn.Module:
    """The main entry to apply Mixed Precision.

    Args:
        model (torch.nn.Module): raw fp32 model or prepared model.
        configs_mapping (Dict[Tuple[str, callable], MixPrecisionConfig]): per-op configuration.
        mode (Mode, optional): select from [PREPARE, CONVERT and QUANTIZE]. Defaults to Mode.QUANTIZE.

    Returns:
        torch.nn.Module: prepared model or quantized model.
    """
    # only support fp16 and bf16 now, more types might be added later
    from neural_compressor.torch.algorithms.mixed_precision import HalfPrecisionConverter

    half_precision_converter = HalfPrecisionConverter(configs_mapping, *args, **kwargs)
    mixed_precision_model = half_precision_converter.convert(model)

    return mixed_precision_model
