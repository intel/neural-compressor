#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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


import gc
import logging
import math
import os
from neural_compressor.transformers.utils.utility import _ipex_version
from accelerate import init_empty_weights
from datasets import load_dataset
from neural_compressor.torch.algorithms.weight_only.modules import INCWeightOnlyLinear as WeightOnlyLinear
from neural_compressor.torch.quantization import (
    AutoRoundConfig,
    AWQConfig,
    GPTQConfig,
    RTNConfig,
    SmoothQuantConfig,
    TEQConfig,
    convert,
    prepare,
    quantize,
)
from neural_compressor.utils.utility import LazyImport
from transformers import AutoTokenizer

from neural_compressor.transformers.utils.utility import (
    is_autoround_available,
    is_ipex_available,
)

from ..utils import CpuInfo

if is_ipex_available():
    import intel_extension_for_pytorch as ipex

if is_autoround_available():
    from auto_round.export.export_to_itrex.model_wrapper import WeightOnlyLinear as auto_round_woqlinear # pylint: disable=E0401
    from neural_compressor.torch.algorithms.weight_only.autoround import get_dataloader as get_autoround_dataloader

torch = LazyImport("torch")


logger = logging.getLogger(__name__)


DTYPE_BITS_MAPPING = {
    "nf4": 4,
    "fp4": 4,  # fp4 == fp4_e2m1
    "fp4_e2m1": 4,
    "int4": 4,
    "int4_fullrange": 4,
    "int4_clip": 4,
    "fp8": 8,  # fp8 == fp8_e4m3
    "fp8_e5m2": 8,
    "fp8_e4m3": 8,
    "int8": 8,
}


def unpack_weight(qweight, scales, qzeros, q_config):
    sym = q_config.sym
    bits = q_config.bits
    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)
    if qzeros is not None:
        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // bits), wf.unsqueeze(0)
        ).to(torch.int16 if bits == 8 else torch.int8)
        torch.bitwise_and(zeros, (2**bits) - 1, out=zeros)
        if bits == 8:
            zeros = zeros.to(torch.int8 if sym else torch.uint8)
        # due to INC minus one
        zeros = zeros + 1
        try:
            zeros = zeros.reshape(scales.shape)
        except:
            # zeros and scales have different iteam numbers.
            # remove 1 (due to 0 + 1 in line 68)
            zeros = zeros[zeros != 1]
            zeros = zeros.reshape(scales.shape)

        # due to INC asym return torch.uint8 but backend request int8,
        # change it to int8 with offset 128
        if not sym and bits == 8:
            zeros = (zeros.to(torch.int32) - 128).to(torch.int8)
        zeros = zeros.contiguous()
    else:
        zeros = None

    weight = torch.bitwise_right_shift(
        torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1), wf.unsqueeze(-1)
    ).to(torch.int16 if bits == 8 else torch.int8)
    torch.bitwise_and(weight, (2**bits) - 1, out=weight)
    if bits == 8:
        # due to INC add shift bias for sym
        if sym:
            shift_bias = 2 ** (bits - 1)
            weight -= shift_bias
        weight = weight.to(torch.int8 if sym else torch.uint8)
        # due to INC asym return torch.uint8 but backend request int8,
        # change it to int8 with offset 128
        if not sym:
            weight = (weight.to(torch.int32) - 128).to(torch.int8)
    return weight.contiguous(), scales.contiguous(), zeros


def replace_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    device="cpu",
    empty_weights=False,
):
    if modules_to_not_convert is None:
        # output_layer is chatglm last layer name
        # embed_out is dolly_v2 last layer name
        modules_to_not_convert = []
    if quantization_config.llm_int8_skip_modules:
        modules_to_not_convert.extend(
            quantization_config.llm_int8_skip_modules
        )
        modules_to_not_convert = list(set(modules_to_not_convert))
    model, is_replaced = _replace_linear(
        model,
        modules_to_not_convert,
        current_key_name,
        quantization_config,
        device=device,
        empty_weights=empty_weights,
    )

    if not is_replaced:
        logger.warning(
            "You are loading your model in 8bit or 4bit but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model


def _replace_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    is_replaced=False,
    device="cpu",
    empty_weights=False,
):
    """Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successfully or not.
    """
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)
        is_removed = False
        use_optimum_format = getattr(module, "use_optimum_format", False) or \
            quantization_config.weight_dtype not in [
                "fp8_e5m2",
                "fp8_e4m3",
                "int4_fullrange",
            ]

        if (
            isinstance(module, torch.nn.Linear)
            or isinstance(module, WeightOnlyLinear)
            or (is_autoround_available() and isinstance(module, auto_round_woqlinear))
            or (
                is_ipex_available()
                and isinstance(module, ipex.nn.utils._weight_prepack._IPEXLinear)
            )
        ) and (name not in modules_to_not_convert):
            # Check if the current key is not in the `modules_to_not_convert`
            if not any(
                key in ".".join(current_key_name) for key in modules_to_not_convert
            ):
                with init_empty_weights():
                    in_features = module.in_features
                    out_features = module.out_features
                    if (
                        device == "cpu"
                        or device == torch.device("cpu")
                        or device == "auto"
                    ):
                        if is_ipex_available() and quantization_config.use_ipex:
                            from intel_extension_for_pytorch.nn.modules import (
                                WeightOnlyQuantizedLinear as ipex_linear,
                            )
                            from intel_extension_for_pytorch.utils.weight_only_quantization import (
                                _convert_optimum_format_to_desired,
                            )

                            qweight, scales, qzeros = (
                                _convert_optimum_format_to_desired(
                                    module.qweight, module.scales, module.qzeros
                                )
                            )

                            weight_dtype = {
                                4: ipex.quantization.WoqWeightDtype.INT4,
                                8: ipex.quantization.WoqWeightDtype.INT8,
                            }
                            compute_dtype = {
                                "fp32": ipex.quantization.WoqLowpMode.NONE,  # follow the activation datatype.
                                "bf16": ipex.quantization.WoqLowpMode.BF16,
                                "fp16": ipex.quantization.WoqLowpMode.FP16,
                                "int8": ipex.quantization.WoqLowpMode.INT8,
                            }

                            ipex_qconfig_mapping = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                                weight_dtype=weight_dtype[quantization_config.bits],
                                lowp_mode=compute_dtype[
                                    quantization_config.compute_dtype
                                ],
                                act_quant_mode=ipex.quantization.WoqActQuantMode.PER_IC_BLOCK,
                                group_size=quantization_config.group_size,
                            )
                            tmp_linear = torch.nn.Linear(
                                in_features,
                                out_features,
                                True if hasattr(module, "bias") else False,
                            )
                            tmp_linear.qconfig = ipex_qconfig_mapping.global_qconfig
                            model._modules[name] = (
                                ipex_linear.from_float_and_int4_weight(
                                    mod=tmp_linear,
                                    qweight=qweight,
                                    scales=scales,
                                    zero_points=qzeros,
                                    bias=(
                                        module.bias if hasattr(module, "bias") else None
                                    ),
                                    group_size=quantization_config.group_size,
                                    g_idx=(
                                        module.g_idx
                                        if hasattr(module, "g_idx")
                                        else None
                                    ),
                                )
                            )
                    elif device == "xpu" or device == torch.device("xpu"):
                        from intel_extension_for_pytorch.nn.utils._quantize_convert import \
                            WeightOnlyQuantizedLinear as ipex_linear # pylint: disable=E0401
                        model._modules[name] = ipex_linear(
                            in_features,
                            out_features,
                            module.bias is not None,
                            compute_dtype=quantization_config.compute_dtype,
                            compress_statistics=False,
                            weight_dtype=quantization_config.weight_dtype,
                            scale_dtype=quantization_config.scale_dtype,
                            blocksize=quantization_config.group_size,
                            scheme=quantization_config.scheme,
                            compression_dtype=getattr(module, "compression_dtype",
                                                      torch.int8 if _ipex_version < "2.3.10" else torch.int32),
                            compression_dim=getattr(module, "compression_dim", 0 if _ipex_version < "2.3.10" else 1),
                            device=device,
                            use_optimum_format=getattr(module, "use_optimum_format",
                                                       False if _ipex_version < "2.3.10" else True),
                        )
                        if quantization_config.quant_method.value == "gptq":
                            g_idx = getattr(
                                module,
                                "g_idx",
                                torch.zeros(in_features, dtype=torch.int32).to(device),
                            )
                        else:
                            g_idx = None
                        model._modules[name].set_scales_zps_gidx(
                            (
                                module.scales
                                if hasattr(module, "scales")
                                else torch.ones(
                                    (
                                        out_features,
                                        math.ceil(
                                            in_features / quantization_config.group_size
                                        ),
                                    ),
                                    dtype=convert_dtype_str2torch(
                                        quantization_config.compute_dtype
                                    ),
                                    device=torch.device(device),
                                ) if _ipex_version < "2.3.10" else torch.ones(
                                    (
                                        math.ceil(
                                            in_features / quantization_config.group_size
                                        ),
                                        out_features,
                                    ),
                                    dtype=convert_dtype_str2torch(
                                        quantization_config.compute_dtype
                                    ),
                                    device=torch.device(device),
                                )
                            ),
                            module.qzeros if hasattr(module, "qzeros") else None,
                            g_idx,
                        )
                    else:
                        raise Exception(
                            "{} device Unsupported weight only quantization!".format(
                                device
                            )
                        )

                    is_replaced = True
                    # Store the module class in case we need to transpose the weight later
                    model._modules[name].source_cls = type(module)
                    # Force requires grad to False to avoid unexpected errors
                    model._modules[name].requires_grad_(False)
                if quantization_config.use_ipex:
                    pass
                elif (
                    device == "cpu" or device == torch.device("cpu") or device == "auto"
                ):
                    if quantization_config.weight_dtype in [
                        "fp8_e5m2",
                        "fp8_e4m3",
                    ]:
                        model._modules[name].set_fp_weights_bias(
                            module.weight.data,
                            None if module.bias is None else module.bias.data,
                        )
                    else:
                        if quantization_config.weight_dtype in ["int4", "int4_clip", "int8"]:
                            int_weight, scales, zeros = unpack_weight(
                                module.qweight,
                                module.scales,
                                module.qzeros if hasattr(module, "qzeros") else None,
                                quantization_config,
                            )
                            int_weight = int_weight.view(-1, int_weight.shape[-1])
                        else:
                            int_weight = module.unpack_tensor_with_numpy(module.qweight)
                            scales = module.scales
                            zeros = module.qzeros if hasattr(module, "qzeros") else None

                        model._modules[name].set_weights_bias(
                            int_weight,
                            scales,
                            zeros,
                            module.g_idx if hasattr(module, "g_idx") else None,
                            quantization_config,
                            bias=None if module.bias is None else module.bias.data,
                        )
                else:
                    if not hasattr(module, "qweight"):
                        n_pack = (
                            (8 if _ipex_version < "2.3.10" else 32)
                            // DTYPE_BITS_MAPPING[quantization_config.weight_dtype]
                        )
                        weight = torch.zeros(
                            (math.ceil(out_features / n_pack), in_features) if _ipex_version < "2.3.10" else
                            (math.ceil(in_features / n_pack), out_features),
                            dtype=torch.int8 if _ipex_version < "2.3.10" else torch.int32,
                            device=torch.device(device),
                        )
                    model._modules[name].set_weights_bias(
                        module.qweight.data if hasattr(module, "qweight") else weight,
                        None if module.bias is None else module.bias.data,
                    )
                    del module
                    gc.collect()
                    is_removed = True

        if not is_removed and len(list(module.children())) > 0:  # pylint: disable=E1101
            _, is_replaced = _replace_linear(
                module,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                is_replaced=is_replaced,
                device=device,
                empty_weights=empty_weights,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, is_replaced


def default_run_fn(
    model, tokenizer, dataset, max_length=512, n_samples=100, batch_size=8, algo="rtn"
):
    from torch.utils.data import DataLoader

    if isinstance(dataset, (str, bytes, os.PathLike)):
        calib_dataset = load_dataset(dataset, split="train")
    calib_dataset = calib_dataset.shuffle(seed=42)
    if tokenizer is None:
        logger.error("Please provide the tokenizer in quantization_config.")
        exit(0)

    def tokenize_function(examples):
        if algo == "teq":
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        if "prompt" in examples:
            if algo == "teq":
                example = tokenizer(
                    examples["prompt"], padding="max_length", max_length=max_length
                )
            else:
                example = tokenizer(examples["prompt"])
        elif "code" in examples:
            if algo == "teq":
                example = tokenizer(
                    examples["code"], padding="max_length", max_length=max_length
                )
            else:
                example = tokenizer(examples["code"])
        elif "text" in examples:
            if algo == "teq":
                example = tokenizer(
                    examples["text"], padding="max_length", max_length=max_length
                )
            else:
                example = tokenizer(examples["text"])
        else:
            logger.error(
                "Please check dataset prompt identifier,"
                + " NeelNanda/pile-10k is default used calibration dataset."
            )
            exit(0)
        return example

    tokenized_dataset = calib_dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids"])
    tokenized_dataset = tokenized_dataset.filter(lambda x: x["input_ids"].shape[-1] >= max_length)

    def collate_batch(batch):
        input_ids_padded = []
        for text in batch:
            input_ids = text["input_ids"]
            if len(input_ids) >= max_length:
                input_ids = input_ids[:max_length]
                input_ids_padded.append(input_ids)
            else:
                continue
        assert input_ids_padded != [], \
            "The dataset does not have data that meets the required input length. Please reduce seq_len."
        return torch.vstack(input_ids_padded)


    calib_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    total_cnt = 0
    for i, (input_ids) in enumerate(calib_dataloader):
        if total_cnt + input_ids.shape[0] > n_samples:
            input_ids = input_ids[: n_samples - total_cnt, ...]
        total_cnt += input_ids.shape[0]
        if total_cnt >= n_samples:
            break

        try:
            model(
                input_ids=input_ids,
            )
        except ValueError:
            pass

@torch.no_grad()
def run_fn_for_autoround(model, dataloader):
    for data in dataloader:
        if isinstance(data, tuple) or isinstance(data, list):
            model(*data)
        elif isinstance(data, dict):
            model(**data)
        else:
            model(data)

def convert_to_quantized_model(model, config, device="cpu"):
    if device == "xpu" or device == torch.device("xpu"):
        import intel_extension_for_pytorch

        assert (
            hasattr(torch, "xpu") and torch.xpu.is_available()
        ), "There is no xpu device in this system!"

    orig_dtype = torch.float32
    for param in model.parameters():
        orig_dtype = param.dtype
        if orig_dtype != torch.float32:
            model.to(dtype=torch.float32)
        break
    if config.weight_dtype in ["fp8_e4m3", "fp8_e5m2"]:
        return replace_linear(model, None, None, config, device=device)
    else:
        if config.weight_dtype == "int8":
            dtype = "int8"
        elif "int4" in config.weight_dtype:
            dtype = "int4"
        else:
            dtype = config.weight_dtype
        # mapping to INC config
        if config.quant_method.value == "rtn":
            quant_config = RTNConfig(
                 dtype=dtype,
                 bits=config.bits,
                 use_sym=config.sym,
                 group_size=config.group_size,
                 use_layer_wise=config.layer_wise,
            )
            if config.llm_int8_skip_modules != []:
                for module in config.llm_int8_skip_modules:
                    module_name = ".*" + module
                    quant_config.set_local(module_name, RTNConfig(dtype="fp32"))
            logger.info(f"Do RTN algorithm with config {quant_config}")
            model = prepare(model, quant_config)
            model = convert(model)
        elif config.quant_method.value == "awq":
            quant_config = AWQConfig(
                dtype=dtype,
                bits=config.bits,
                use_sym=config.sym,
                group_size=config.group_size,
                use_layer_wise=config.layer_wise,
                use_auto_scale=config.auto_scale,
                use_auto_clip=config.auto_clip,
                folding=True,
            )
            if config.llm_int8_skip_modules != []:
                for module in config.llm_int8_skip_modules:
                    module_name = ".*" + module
                    quant_config.set_local(module_name, AWQConfig(dtype="fp32"))
            logger.info(f"Do AWQ algorithm with config {quant_config}")
            run_fn = default_run_fn
            run_args = (
                config.tokenizer,
                config.dataset,
                config.seq_len,  # max_length
                config.n_samples,  # n_samples
                config.batch_size,  # batch_size
                config.quant_method.value,  # algo
            )
            example_inputs = torch.ones([1, 512], dtype=torch.long).to(device)
            model = prepare(model=model, quant_config=quant_config, example_inputs=example_inputs)
            run_fn(model, *run_args)
            model = convert(model)
        elif config.quant_method.value == "teq":
            quant_config = TEQConfig(
                dtype=dtype,
                bits=config.bits,
                use_sym=config.sym,
                group_size=config.group_size,
                use_layer_wise=config.layer_wise,
                absorb_to_layer=config.absorb_to_layer
            )
            if config.llm_int8_skip_modules != []:
                for module in config.llm_int8_skip_modules:
                    module_name = ".*" + module
                    quant_config.set_local(module_name, TEQConfig(dtype="fp32"))
            logger.info(f"Do TEQ algorithm with config {quant_config}")
            run_fn = default_run_fn
            run_args = (
                config.tokenizer,
                config.dataset,
                config.seq_len,  # max_length
                config.n_samples,  # n_samples
                config.batch_size,  # batch_size
                config.quant_method.value,  # algo
            )
            example_inputs = torch.ones([1, 512], dtype=torch.long).to(device)
            model = prepare(model=model, quant_config=quant_config, example_inputs=example_inputs)
            run_fn(model, *run_args)
            model = convert(model)

        elif config.quant_method.value == "gptq":
            model.seqlen = config.seq_len
            quant_config = GPTQConfig(
                dtype=dtype,
                bits=config.bits,
                use_sym=config.sym,
                group_size=config.group_size,
                use_layer_wise=config.layer_wise,
                act_order=config.desc_act,
                percdamp=config.damp_percent,
                block_size=config.blocksize,
                static_groups=config.static_groups,
                use_mse_search=config.use_mse_search,
            )
            if config.llm_int8_skip_modules != []:
                for module in config.llm_int8_skip_modules:
                    module_name = ".*" + module
                    quant_config.set_local(module_name, GPTQConfig(dtype="fp32"))
            logger.info(f"Do GPTQ algorithm with config {quant_config}")
            run_fn = default_run_fn
            run_args = (
                config.tokenizer,
                config.dataset,
                config.seq_len,  # max_length
                config.n_samples,  # n_samples
                config.batch_size,  # batch_size
                config.quant_method.value,  # algo
            )
            model = prepare(model=model, quant_config=quant_config)
            run_fn(model, *run_args)
            model = convert(model)
        elif config.quant_method.value == "autoround":
            quant_config = AutoRoundConfig(
                dtype=dtype,
                bits=config.bits,
                use_sym=config.sym,
                group_size=config.group_size,
                enable_quanted_input=not config.disable_quanted_input,
                lr=config.lr,
                minmax_lr=config.minmax_lr,
                seqlen=config.seq_len,
                nsamples=config.n_samples,
                iters=config.iters,
                scale_dtype=config.scale_dtype,
            )
            if config.llm_int8_skip_modules != []:
                for module in config.llm_int8_skip_modules:
                    module_name = ".*" + module
                    quant_config.set_local(module_name, AutoRoundConfig(dtype="fp32"))
            logger.info(f"Do AutoRound algorithm with config {quant_config}")
            dataloader = get_autoround_dataloader(tokenizer=config.tokenizer,
                                                  seqlen=config.seq_len,
                                                  dataset_name="NeelNanda/pile-10k",
                                                  seed=42,
                                                  bs=config.batch_size,
                                                  nsamples=config.n_samples)
            run_fn = run_fn_for_autoround
            run_args = (dataloader,)
            model = prepare(model=model, quant_config=quant_config)
            run_fn(model, *run_args)
            model = convert(model)
        else:
            assert False, "The Supported algorithm are RTN, AWQ, TEQ, GPTQ, AUTOROUND"

        if device == "xpu" or device == torch.device("xpu"):
            logger.warning("The recommended ipex version is higher than 2.3.10 for xpu device.")

        model.eval()
        # INC attribute conflicted with transformers when use nf4/int8 training.
        del model.is_quantized
        # TODO replace_linear
        # q_model = replace_linear(model, None, None, config, device=device)
        q_model = model
        
        if orig_dtype != torch.float32:
            q_model.to(dtype=orig_dtype)

        return q_model.to(device)


def convert_dtype_str2torch(str_dtype):
    if str_dtype == "int8":
        return torch.int8
    elif str_dtype == "fp32" or str_dtype == "auto":
        return torch.float
    elif str_dtype == "fp16":
        return torch.float16
    elif str_dtype == "bf16":
        return torch.bfloat16
    else:
        assert False, "Unsupported str dtype {} to torch dtype".format(str_dtype)


def convert_dtype_torch2str(dtype):
    if dtype == torch.int8:
        return "int8"
    elif dtype == torch.float:
        return "fp32"
    elif dtype == torch.float16:
        return "fp16"
    elif dtype == torch.bfloat16:
        return "bf16"
    elif isinstance(dtype, str) and dtype in ["int8", "fp32", "fp16", "bf16"]:
        return dtype
    else:
        assert False, "Unsupported pytorch dtype {} to str dtype".format(dtype)


def get_bits(config):
    if config.weight_dtype == "int8":
        bits = 8
    elif "int4" in config.weight_dtype:
        bits = 4
    else:
        assert False, "Unsupported {} for quantize weight only by IPEX backend".format(
            config.weight_dtype
        )
    return bits


def convert_to_smoothquant_model(model, quantization_config):
    model_type = model.config.model_type.replace("_", "-")
    # ipex.optimize_transformers
    if quantization_config.ipex_opt_llm is None:
        if model_type in IPEX_OPT_LLM_SUPPORTED:
            quantization_config.ipex_opt_llm = True
            logger.info(
                "quantization_config.ipex_opt_llm set to True and ipex.llm.optimize is used."
            )
        else:
            quantization_config.ipex_opt_llm = False
    if quantization_config.ipex_opt_llm:
        qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(alpha=0.5)
        model = ipex.llm.optimize(
            model.eval(),
            quantization_config=qconfig,
            dtype=torch.float32,
            inplace=True,
            deployment_mode=False,
        )
        model.eval()
    # past_key_values
    num_beams = quantization_config.num_beams
    if quantization_config.ipex_opt_llm:
        past_key_values = generate_dummy_past_key_values_for_opt_llm(
            config=model.config, input_bs=1, num_beams=num_beams
        )
    else:
        past_key_values = generate_dummy_past_key_values(
            config=model.config, input_bs=1
        )
    # get calibration dataloader
    calib_dataloader = get_dataloader(
        model_type, quantization_config, past_key_values=past_key_values
    )

    def calib_func(model):
        with torch.no_grad():
            for i, (inputs, last_ind) in enumerate(calib_dataloader):
                if i >= quantization_config.n_samples:
                    break
                if model_type in MODEL_TYPES_REQUIRING_POSITION_IDS:
                    model(
                        input_ids=inputs["input_ids"],
                        past_key_values=inputs["past_key_values"],
                        position_ids=inputs["position_ids"],
                        attention_mask=inputs["attention_mask"],
                    )
                else:
                    model(
                        input_ids=inputs["input_ids"],
                        past_key_values=inputs["past_key_values"],
                        attention_mask=inputs["attention_mask"],
                    )

    # example_inputs
    for i, (inputs, last_ind) in enumerate(calib_dataloader):
        if model_type in MODEL_TYPES_REQUIRING_POSITION_IDS:
            example_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "position_ids": inputs["position_ids"],
                "past_key_values": inputs["past_key_values"],
            }
        else:
            example_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "past_key_values": inputs["past_key_values"],
            }
        break
    quant_config = SmoothQuantConfig(
        alpha=quantization_config.alpha,
        init_alpha=quantization_config.init_alpha,
        alpha_min=quantization_config.alpha_min,
        alpha_max=quantization_config.alpha_max,
        alpha_step=quantization_config.alpha_step,
        shared_criterion=quantization_config.shared_criterion,
        do_blockwise=quantization_config.do_blockwise,
        excluded_precisions=quantization_config.excluded_precisions,
    )
    # fallback
    if model_type in ["gptj", "gpt_neox", "mpt"]:
        quant_config = quant_config.set_local(
            torch.add, SmoothQuantConfig(w_dtype="fp32", act_dtype="fp32")
        )
    model = quantize(
        model,
        quant_config=quant_config,
        run_fn=calib_func,
        example_inputs=example_inputs,
    )

    return model
