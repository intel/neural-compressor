# -*- coding: utf-8 -*-
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
"""Intel Neural Compressor model convert."""

import json
import math
import os
import types

from datasets import load_dataset

from neural_compressor.common.utils import LazyImport, logger
from neural_compressor.torch.algorithms.weight_only.modules import INCWeightOnlyLinear
from neural_compressor.torch.algorithms.weight_only.utility import repack_awq_to_optimum_format
from neural_compressor.torch.quantization import (
    AutoRoundConfig,
    AWQConfig,
    GPTQConfig,
    RTNConfig,
    TEQConfig,
    convert,
    prepare,
)
from neural_compressor.torch.utils import is_ipex_available

if is_ipex_available():
    import intel_extension_for_pytorch as ipex

from typing import Union

torch = LazyImport("torch")


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
    if quantization_config.modules_to_not_convert:
        modules_to_not_convert.extend(quantization_config.modules_to_not_convert)
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
        if (
            isinstance(module, torch.nn.Linear)
            or isinstance(module, INCWeightOnlyLinear)
            or (is_ipex_available() and isinstance(module, ipex.nn.utils._weight_prepack._IPEXLinear))
        ) and (name not in modules_to_not_convert):
            # Check if the current key is not in the `modules_to_not_convert`
            if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
                in_features = module.in_features
                out_features = module.out_features
                if device == "cpu" or device == torch.device("cpu") or device == "auto":
                    from intel_extension_for_pytorch.nn.modules import WeightOnlyQuantizedLinear as ipex_linear
                    from intel_extension_for_pytorch.utils.weight_only_quantization import (
                        _convert_optimum_format_to_desired,
                    )

                    qweight = module.qweight
                    scales = module.scales
                    qzeros = module.qzeros

                    qweight, scales, qzeros = _convert_optimum_format_to_desired(qweight, scales, qzeros)
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
                        lowp_mode=compute_dtype[quantization_config.compute_dtype],
                        act_quant_mode=ipex.quantization.WoqActQuantMode.PER_IC_BLOCK,
                        group_size=quantization_config.group_size,
                    )
                    tmp_linear = torch.nn.Linear(
                        in_features,
                        out_features,
                        True if hasattr(module, "bias") and module.bias is not None else False,
                    )
                    if tmp_linear.bias is not None and module.bias is not None:
                        tmp_linear.bias = torch.nn.Parameter(module.bias.float())

                    tmp_linear.qconfig = ipex_qconfig_mapping.global_qconfig
                    model._modules[name] = ipex_linear.from_float_and_int4_weight(
                        mod=tmp_linear,
                        qweight=qweight,
                        scales=scales,
                        zero_points=qzeros,
                        bias=(module.bias.float() if hasattr(module, "bias") and module.bias is not None else None),
                        group_size=quantization_config.group_size,
                        g_idx=(module.g_idx if hasattr(module, "g_idx") else None),
                    )

                elif device == "xpu" or device == torch.device("xpu"):
                    from intel_extension_for_pytorch.nn.utils._quantize_convert import (
                        WeightOnlyQuantizedLinear as ipex_linear,  # pylint: disable=E0401
                    )

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
                        compression_dtype=getattr(module, "compression_dtype", torch.int32),
                        compression_dim=getattr(module, "compression_dim", 1),
                        device=device,
                        use_optimum_format=getattr(module, "use_optimum_format", True),
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
                                    math.ceil(in_features / quantization_config.group_size),
                                    out_features,
                                ),
                                dtype=convert_dtype_str2torch(quantization_config.compute_dtype),
                                device=torch.device(device),
                            )
                        ),
                        module.qzeros if hasattr(module, "qzeros") else None,
                        g_idx,
                    )
                    if not hasattr(module, "qweight"):
                        n_pack = 32 // quantization_config.bits

                        weight = torch.zeros(
                            (math.ceil(in_features / n_pack), out_features),
                            dtype=torch.int32,
                            device=torch.device(device),
                        )
                    model._modules[name].set_weights_bias(
                        module.qweight.data if hasattr(module, "qweight") else weight,
                        None if module.bias is None else module.bias.data,
                    )
                else:
                    raise Exception("{} device Unsupported weight only quantization!".format(device))

                is_replaced = True
                is_removed = True
                # Store the module class in case we need to transpose the weight later
                model._modules[name].source_cls = type(module)
                # Force requires grad to False to avoid unexpected errors
                model._modules[name].requires_grad_(False)

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


def default_run_fn(model, tokenizer, dataset, max_length=512, n_samples=100, batch_size=8, algo="rtn"):
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
                example = tokenizer(examples["prompt"], padding="max_length", max_length=max_length)
            else:
                example = tokenizer(examples["prompt"])
        elif "code" in examples:
            if algo == "teq":
                example = tokenizer(examples["code"], padding="max_length", max_length=max_length)
            else:
                example = tokenizer(examples["code"])
        elif "text" in examples:
            if algo == "teq":
                example = tokenizer(examples["text"], padding="max_length", max_length=max_length)
            else:
                example = tokenizer(examples["text"])
        else:
            logger.error(
                "Please check dataset prompt identifier," + " NeelNanda/pile-10k is default used calibration dataset."
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
        assert (
            input_ids_padded != []
        ), "The dataset does not have data that meets the required input length. Please reduce seq_len."
        return torch.vstack(input_ids_padded)

    calib_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    total_cnt = 0
    from neural_compressor.torch.utils import get_accelerator

    device = get_accelerator().current_device_name()
    for i, (input_ids) in enumerate(calib_dataloader):
        input_ids = input_ids.to(torch.device(device))
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

        assert hasattr(torch, "xpu") and torch.xpu.is_available(), "There is no xpu device in this system!"
        if "INC_TARGET_DEVICE" not in os.environ:
            os.environ["INC_TARGET_DEVICE"] = "cpu"
            logger.info(
                "Set the environment variable INC_TARGET_DEVICE='cpu'"
                " to ensure the quantization process occurs on the CPU."
            )

    # mapping to INC config
    dtype = "int4" if config.weight_dtype == "int4_fullrange" else config.weight_dtype
    if config.quant_method.value == "rtn":
        quant_config = RTNConfig(
            dtype=dtype,
            bits=config.bits,
            use_sym=config.sym,
            group_size=config.group_size,
            use_layer_wise=config.use_layer_wise,
            model_path=config.model_path,
            quant_lm_head=config.quant_lm_head,
        )
        if config.modules_to_not_convert != []:
            for module in config.modules_to_not_convert:
                module_name = ".*" + module
                quant_config.set_local(module_name, RTNConfig(dtype="fp32"))
        logger.info(f"Do RTN algorithm with config {quant_config}")
        model = prepare(model, quant_config)
        model = convert(model)
    elif config.quant_method.value == "gptq":
        model.seqlen = config.seq_len
        quant_config = GPTQConfig(
            dtype=dtype,
            bits=config.bits,
            use_sym=config.sym,
            group_size=config.group_size,
            use_layer_wise=config.use_layer_wise,
            model_path=config.model_path,
            act_order=config.desc_act,
            percdamp=config.damp_percent,
            block_size=config.blocksize,
            static_groups=config.static_groups,
            use_mse_search=config.use_mse_search,
            true_sequential=config.true_sequential,
            quant_lm_head=config.quant_lm_head,
        )
        if config.modules_to_not_convert != []:
            for module in config.modules_to_not_convert:
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
    elif config.quant_method.value == "awq":
        quant_config = AWQConfig(
            dtype=dtype,
            bits=config.bits,
            use_sym=config.sym,
            group_size=config.group_size,
            use_layer_wise=config.use_layer_wise,
            use_auto_scale=config.auto_scale,
            use_auto_clip=config.auto_clip,
            folding=True,
            absorb_layer_dict=config.absorb_layer_dict,
            quant_lm_head=config.quant_lm_head,
        )
        if config.modules_to_not_convert != []:
            for module in config.modules_to_not_convert:
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
            use_layer_wise=config.use_layer_wise,
            quant_lm_head=config.quant_lm_head,
            absorb_to_layer=config.absorb_layer_dict,
        )
        if config.modules_to_not_convert != []:
            for module in config.modules_to_not_convert:
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
            use_layer_wise=config.use_layer_wise,
        )
        if config.modules_to_not_convert != []:
            for module in config.modules_to_not_convert:
                module_name = ".*" + module
                quant_config.set_local(module_name, AutoRoundConfig(dtype="fp32"))
        logger.info(f"Do AutoRound algorithm with config {quant_config}")
        from neural_compressor.torch.algorithms.weight_only.autoround import get_dataloader as get_autoround_dataloader

        dataloader = get_autoround_dataloader(
            tokenizer=config.tokenizer,
            seqlen=config.seq_len,
            dataset_name=config.dataset,
            seed=42,
            bs=config.batch_size,
            nsamples=config.n_samples,
        )
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

    q_model = replace_linear(model, None, None, config, device=device)

    if config.use_layer_wise and not (q_model.device == device or q_model.device.type == device):
        logger.warning(
            "Do not convert device to avoid out of memory. Recommend using saved quantized model to inference."
        )
        return q_model

    return q_model.to(device)


def convert_to_GPTQ_checkpoints(model, quantization_config):
    from intel_extension_for_pytorch.nn.modules import WeightOnlyQuantizedLinear as ipex_cpu_linear

    from neural_compressor.adaptor.torch_utils.util import set_module
    from neural_compressor.torch.algorithms.weight_only.modules import INCWeightOnlyLinear

    dtype = "int4" if quantization_config.bits == 4 else "int8"
    bits = quantization_config.bits
    group_size = quantization_config.group_size
    zp = False if quantization_config.sym else True
    scale_dtype = quantization_config.scale_dtype
    desc_act = (True if hasattr(quantization_config, "desc_act") else False,)

    for name, module in model.named_modules():
        if isinstance(module, ipex_cpu_linear):
            in_features = module.in_features
            out_features = module.out_features
            new_module = INCWeightOnlyLinear(
                in_features,
                out_features,
                dtype=dtype,
                bits=bits,
                group_size=group_size,
                zp=zp,
                bias=True if hasattr(module, "bias") else False,
                scale_dtype=scale_dtype,
                g_idx=desc_act,
                use_optimum_format=True,
            )

            new_module.bits = 8
            new_module.n_pack = 32 // 8
            qweight = (
                new_module.pack_tensor_with_numpy(module._op_context.to_public(module._op_context.get_weight()))
                .t()
                .contiguous()
            )
            new_module.bits = bits
            new_module.n_pack = 32 // bits
            scales = module._op_context.get_scales().t().contiguous()
            bias = module._op_context.get_bias()
            qzeros = new_module.pack_tensor_with_numpy(
                module._op_context.get_zero_points().t().to(torch.uint8) - 1
            ).contiguous()
            g_idx = module._op_context.get_g_idx()

            new_module.qweight = qweight
            new_module.scales = scales
            new_module.qzeros = qzeros
            if g_idx is not None:
                new_module.g_idx = g_idx.contiguous()
            if bias is not None:
                new_module.bias = bias.contiguous()

            set_module(model, name, new_module)
    return model


def make_contiguous(model):
    for param in model.parameters():
        if param.data.ndimension() > 1:
            param.data = param.data.contiguous()


def save_low_bit(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):

    assert hasattr(self, "quantization_config"), "Detected this model is not a low-bit model."

    if os.path.isfile(save_directory):
        logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
        return

    os.makedirs(save_directory, exist_ok=True)
    # use transformers original `save_pretrained` function
    del self.save_pretrained
    make_contiguous(self)

    if self.device == "cpu" or self.device == torch.device("cpu"):
        convert_to_GPTQ_checkpoints(self, self.quantization_config)
    if self.device == "xpu" or (isinstance(self.device, torch.device) and self.device.type == "xpu"):
        from intel_extension_for_pytorch.nn.utils._quantize_convert import WeightOnlyQuantizedLinear

        for name, module in self.named_modules():
            if isinstance(module, WeightOnlyQuantizedLinear):
                if module.weight_transposed:
                    module.qweight.data = module.qweight.t_().contiguous()
                    module.scales.data = module.scales.t_().contiguous()
                    module.weight_transposed = False

    self.save_pretrained(save_directory=save_directory, push_to_hub=push_to_hub, **kwargs)
    self.save_pretrained = types.MethodType(save_low_bit, self)
    # We conveniently save all the keys of the model to have them on hand,
    # so that when using 'low_cpumem load',
    # it's not necessary to load the entire model to extract its keys
    # and we can avoid gc not triggered potentially.
    all_checkpoint_keys = {"all_checkpoint_keys": list(self.state_dict().keys())}
    json_file_path = os.path.join(save_directory, "all_checkpoint_keys.json")
    with open(json_file_path, "w") as json_file:
        json.dump(all_checkpoint_keys, json_file)
    if push_to_hub:
        use_auth_token = kwargs.pop("use_auth_token", None)

        if use_auth_token is not None:
            logger.warning.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers.",
                FutureWarning,
            )

            token = use_auth_token
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )

            if token is not None:
                kwargs["token"] = token
        commit_message = kwargs.pop("commit_message", None)
        repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
        repo_id = self._create_repo(repo_id, **kwargs)
        files_timestamps = self._get_files_timestamps(save_directory)
        self._upload_modified_files(
            save_directory,
            repo_id,
            files_timestamps,
            commit_message=commit_message,
            token=kwargs.get("token"),
        )
    self.quantization_config.save_pretrained(save_directory, **kwargs)


def repack_awq_and_load_state_dict(
    model, resolved_archive_file, loaded_state_dict_keys, quantization_config, is_sharded
):
    from transformers.modeling_utils import load_state_dict

    bits = quantization_config.bits
    group_size = quantization_config.group_size

    state_dict = {}
    if isinstance(resolved_archive_file, str):
        resolved_archive_file = [resolved_archive_file]
    assert isinstance(resolved_archive_file, list), "Please check if the loading weight is shared."
    for shard_file in resolved_archive_file:
        assert shard_file.endswith("safetensors"), "Please check the loading weight saved format."
        state_dict.update(load_state_dict(shard_file))
        assert len(state_dict.keys()) > 0, "Please check the state_dict loading."
    for name, module in model.named_modules():
        if isinstance(module, INCWeightOnlyLinear):
            assert name + ".qweight" in loaded_state_dict_keys, f"Please check the state_dict key { name + '.qweight'}"
            assert name + ".qzeros" in loaded_state_dict_keys, f"Please check the state_dict key {name + '.qzeros'}"
            assert name + ".scales" in loaded_state_dict_keys, f"Please check the state_dict key { name + '.scales'}"
            if name + ".scales" in loaded_state_dict_keys:
                awq_qweight = state_dict[name + ".qweight"]
                awq_qzeros = state_dict[name + ".qzeros"]
                awq_scales = state_dict[name + ".scales"]
                qweight, qzeros, awq_scales = repack_awq_to_optimum_format(
                    awq_qweight, awq_qzeros, awq_scales, bits, group_size
                )
                state_dict[name + ".qweight"] = qweight
                state_dict[name + ".qzeros"] = qzeros
                state_dict[name + ".scales"] = awq_scales

    model.load_state_dict(state_dict, strict=False, assign=True)

    return model
