#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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
import math
import os
import torch
import torch.nn.functional as F
import transformers
import peft
from peft import __version__ as PEFT_VERSION
from pathlib import Path
from typing import List, Mapping, NewType, Optional, Tuple, Union
from tqdm import tqdm
from packaging.version import Version

from transformers import BatchEncoding

from lm_eval import utils
from lm_eval.base import BaseLM
import re

TokenSequence = Union[List[int], torch.LongTensor, torch.Tensor, BatchEncoding]

_DeviceMapping = NewType("DeviceMapping", Mapping[str, Union[int, str, torch.device]])


def _get_accelerate_args(
    device_map_option: Optional[str] = "auto",
    max_memory_per_gpu: Optional[Union[int, str]] = None,
    max_cpu_memory: Optional[Union[int, str]] = None,
    offload_folder: Optional[str] = "./offload",
) -> dict:
    """Returns the kwargs needed to apply `accelerate` in `AutoModel.from_pretrained`."""
    max_memory = {}
    if max_memory_per_gpu is not None:
        max_memory_per_gpu_map = {
            device_idx: max_memory_per_gpu
            for device_idx in range(torch.cuda.device_count())
        }
        max_memory.update(max_memory_per_gpu_map)
    if max_cpu_memory is not None:
        max_memory["cpu"] = max_cpu_memory

    args = {}
    if max_memory:
        args["max_memory"] = max_memory
    args["device_map"] = device_map_option
    args["offload_folder"] = offload_folder
    return args


def _get_dtype(
    dtype: Union[str, torch.dtype], config: Optional[transformers.AutoConfig] = None
) -> torch.dtype:
    """Converts `dtype` from `str` to torch.dtype when possible."""
    if dtype is None and config is not None:
        _torch_dtype = config.torch_dtype
    elif isinstance(dtype, str) and dtype != "auto":
        # Convert `str` args torch dtype: `float16` -> `torch.float16`
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype


class HuggingFaceAutoLM(BaseLM):
    AUTO_CONFIG_CLASS: transformers.AutoConfig = transformers.AutoConfig
    AUTO_TOKENIZER_CLASS: transformers.AutoTokenizer = transformers.AutoTokenizer
    AUTO_MODEL_CLASS: transformers.AutoModel = None
    AUTO_PEFT_CLASS: peft.PeftModel = None

    # Default max sequence length setting for when no `max_length` is provided
    # or no max length config setting is found in the model or tokenizer.
    _DEFAULT_MAX_LENGTH: int = 2048

    def __init__(
        self,
        pretrained: str,
        quantized: Optional[Union[bool, str]] = False,
        tokenizer: Optional[str] = None,
        subfolder: Optional[str] = None,
        revision: Optional[str] = "main",
        batch_size: Optional[Union[int, str]] = 1,
        max_batch_size: Optional[int] = 512,
        max_gen_toks: Optional[int] = 256,
        max_length: Optional[int] = None,
        add_special_tokens: Optional[bool] = None,
        use_accelerate: Optional[bool] = False,
        low_cpu_mem_usage: Optional[bool] = True,
        device_map_option: Optional[str] = "auto",
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        dtype: Optional[Union[str, torch.dtype]] = None,
        device: Optional[Union[int, str]] = "cuda",
        peft: str = None,
        load_in_8bit: Optional[bool] = False,
        load_in_4bit: Optional[bool] = False,
        trust_remote_code: Optional[bool] = False,
        gptq_use_triton: Optional[bool] = False,
        inject_fused_attention: Optional[bool] = True,
        bnb_4bit_quant_type: Optional[str] = None,
        bnb_4bit_compute_dtype: Optional[Union[str, torch.dtype]] = None,
        bnb_4bit_use_double_quant: Optional[bool] = False,
        init_empty_weights: Optional[bool] = False,
        model_format: Optional[str] = "torch",
        backend: Optional[str] = "default",
        _commit_hash: Optional[str] = None
    ):
        """Initializes a HuggingFace `AutoModel` and `AutoTokenizer` for evaluation.
        Args:
            pretrained (str):
                The HuggingFace Hub model ID name or the path to a pre-trained
                model to load. This is effectively the `pretrained_model_name_or_path`
                argument of `from_pretrained` in the HuggingFace `transformers` API.
            quantized (str or bool, optional, defaults to False):
                File name of a GPTQ quantized model to load. Set to `True` to use the
                default name of the quantized model.
            add_special_tokens (bool, optional, defaults to True):
                Whether to add special tokens to the input sequences. If `None`, the
                default value will be set to `True` for seq2seq models (e.g. T5) and
                `False` for causal models.
                WARNING: Evaluating causal models with `add_special_tokens=True` is
                currently __not__ supported.
            > Large model loading `accelerate` arguments
            use_accelerate (bool, optional, defaults to False):
                If True, uses the `accelerate` library to load a large model across
                multiple devices.
            low_cpu_mem_usage (bool, optional, defaults to True):
                It True, uses the `accelerate` library to accelerate loading the model.
            device_map_option (str, optional, defaults to "auto"):
                The device map option to use when loading the model with
                `accelerate`.
                Options:
                    "auto", "balanced", "balanced_low_0", "sequential"
                See the `accelerate` docs for more details on these options:
                https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained.device_map
            max_memory_per_gpu (Union[int, str], optional, defaults to None):
                The maximum memory available for each GPU in bytes as `int` or in
                the format f"{significand}{unit_symbol}" where {unit_symbol} is
                any of ["GB", "MB", "GIB", "MIB"]. Refer to the `max_memory` arg in
                the "Parameters for big model inference" section of the following
                docs:
                https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained.max_memory
            max_cpu_memory (Union[int, str], optional, defaults to None):
                The maximum available CPU RAM in bytes as `int` or in the format
                f"{significand}{unit_symbol}" where {unit_symbol} is any of
                ["GB", "MB", "GIB", "MIB"]. Refer to the `max_memory` arg in the
                "Parameters for big model inference" section of the following docs:
                https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained.max_memory
            offload_folder (str, optional, defaults to "./offload"):
                The folder to offload weights into if `device_map` contains any
                "disk" value.
            dtype (Union[str, torch.dtype], optional, defaults to None):):
                Converts the model weights to `dtype`, if specified. Strings get
                converted to `torch.dtype` objects (e.g. `float16` -> `torch.float16`).
                Use `dtype="auto"` to derive the type from the model’s weights.
            peft (str, optional, defaults to None):
                Path of the adapter weights to load from Huggingface. This will usually
                include a directory that includes the files `adapter_config.json` and
                `adapter_model.bin`. Compatible with [PEFT](https://github.com/huggingface/peft)
            load_in_8bit (bool, optional, defaults to False):
                If True, will convert the loaded model into mixed-8bit quantized model. See:
                https://huggingface.co/docs/transformers/main/en/main_classes/quantization#load-a-large-model-in-8bit
            load_in_4bit (bool, optional, defaults to False):
                If True, will convert the loaded model into mixed-4bit quantized model. See:
                https://huggingface.co/docs/transformers/main/en/main_classes/quantization#load-a-large-model-in-4bit
            trust_remote_code (bool, optional, defaults to False):
                If True, will trust the remote code when loading the model.
            gptq_use_triton (bool, optional, defaults to False):
                Use Triton for GPTQ inference.
            inject_fused_attention (bool, optional, defaults to True):
                Inject fused attention into GPTQ model.
            bnb_4bit_quant_type (str, optional, defaults to None):
                The quantization type to use for BnB 4bit quantization. See:
                https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L77
            bnb_4bit_compute_dtype (Union[str, torch.dtype], optional, defaults to None):
                The compute dtype to use for BnB 4bit quantization. See:
                https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L74
            bnb_4bit_use_double_quant (bool, optional, defaults to False):
                Whether or not to use double quant to quantize the absmax.
                https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L80
            init_empty_weights (bool, optional, defaults to False):):
                Initialize model with empty weights if model is not used for inference.
            model_format (str, optional, defaults to torch):
                The format of target model, support 'torch' and 'onnx'
        """
        super().__init__()

        assert isinstance(pretrained, str)
        assert isinstance(device, str)
        assert isinstance(batch_size, (int, str))
        if (
            add_special_tokens is not None
            and self.AUTO_MODEL_CLASS is transformers.AutoModelForCausalLM
        ):
            # TODO: Support evaluating causal models with special tokens. Currently,
            # this is not possible because the `_loglikelihood_tokens()` method for
            # causal LMs makes a no-special-tokens assumption given that contexts
            # and labels/continuations are tokenized separately without special
            # tokens, concatenated, and then processed as inputs.
            assert (
                not add_special_tokens
            ), "Evaluating causal models with `add_special_tokens=True` is currently not supported."

        # setup for automatic batch size detection
        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self._batch_size = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self._batch_size = int(batch_size)
        self.init_empty_weights = init_empty_weights
        self.max_batch_size = max_batch_size

        self._max_gen_toks = max_gen_toks
        self._max_length = max_length
        self._config = self.AUTO_CONFIG_CLASS.from_pretrained(
            pretrained,
            trust_remote_code=trust_remote_code,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
        )

        self._add_special_tokens = add_special_tokens
        self.tokenizer = self._create_auto_tokenizer(
            pretrained=pretrained,
            revision=revision,
            subfolder=subfolder,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer.model_max_length = self.max_length

        model_kwargs = {}
        if use_accelerate:
            model_kwargs = _get_accelerate_args(
                device_map_option,
                max_memory_per_gpu,
                max_cpu_memory,
                offload_folder,
            )
        self._device = device
        self.model_format = model_format
        if model_format == "torch":
            self.model = self._create_auto_model(
                pretrained=pretrained,
                quantized=quantized,
                trust_remote_code=trust_remote_code,
                revision=revision,
                subfolder=subfolder,
                torch_dtype=_get_dtype(dtype, self._config),
                gptq_use_triton=gptq_use_triton,
                inject_fused_attention=inject_fused_attention,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
                bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                low_cpu_mem_usage=low_cpu_mem_usage,
                **model_kwargs,
            )
            # note: peft_path can be different than pretrained model path
            if peft is not None:
                self.model = self._create_auto_model_peft(
                    model=self.model,
                    peft=peft,
                    revision=revision,
                    subfolder=subfolder,
                    load_in_4bit=load_in_4bit,
                )
            self.model.eval()
            torch.set_grad_enabled(False)

            if use_accelerate and "lm_head" in self.model.hf_device_map:
                # `accelerate` can place `lm_head` weights on a different device than
                # the user specified one so we force `self._device` to be the same as
                # `lm_head`'s.
                self._device = self.model.hf_device_map["lm_head"]
            if not use_accelerate and not (load_in_4bit or load_in_8bit):
                try:
                    self.model.to(self._device)
                except:
                    print(
                        "Failed to place model onto specified device." + \
                        "This may be because the model is quantized via `bitsandbytes`." + \
                        "If the desired GPU is being used, this message is safe to ignore."
                    )

    def _create_auto_model(
        self,
        *,
        pretrained: str,
        quantized: Optional[Union[bool, str]] = False,
        revision: str,
        subfolder: str,
        low_cpu_mem_usage: Optional[bool] = True,
        device_map: Optional[Union[str, _DeviceMapping]] = None,
        max_memory: Optional[dict] = None,
        offload_folder: Optional[str] = None,
        load_in_8bit: Optional[bool] = False,
        load_in_4bit: Optional[bool] = False,
        trust_remote_code: Optional[bool] = False,
        torch_dtype: Optional[Union[str, torch.dtype]] = None,
        gptq_use_triton: Optional[bool] = False,
        inject_fused_attention: Optional[bool] = True,
        bnb_4bit_quant_type: Optional[str] = None,
        bnb_4bit_compute_dtype: Optional[Union[str, torch.dtype]] = None,
        bnb_4bit_use_double_quant: Optional[bool] = False,
    ) -> transformers.AutoModel:
        """Returns a pre-trained pytorch model from a pre-trained model configuration."""
        if not quantized:
            if self.init_empty_weights:
                from accelerate import init_empty_weights
                with init_empty_weights():
                    if self._config.model_type =="chatglm":
                        self.AUTO_MODEL_CLASS = transformers.AutoModel
                    if re.search("qwen-72b", self._config._name_or_path.lower()):
                        model = self.AUTO_MODEL_CLASS.from_pretrained(
                            pretrained,
                            revision=revision + ("/" + subfolder if subfolder is not None else ""),
                            low_cpu_mem_usage=low_cpu_mem_usage,
                            device_map=device_map,
                            max_memory=max_memory,
                            offload_folder=offload_folder,
                            load_in_8bit=load_in_8bit,
                            trust_remote_code=trust_remote_code,
                            torch_dtype=torch_dtype,
                            fp32=(bool(torch_dtype==torch.float32)),
                            fp16=(bool(torch_dtype==torch.float16)),
                        )
                    else:
                        model = self.AUTO_MODEL_CLASS.from_pretrained(
                            pretrained,
                            revision=revision + ("/" + subfolder if subfolder is not None else ""),
                            low_cpu_mem_usage=low_cpu_mem_usage,
                            device_map=device_map,
                            max_memory=max_memory,
                            offload_folder=offload_folder,
                            load_in_8bit=load_in_8bit,
                            trust_remote_code=trust_remote_code,
                            torch_dtype=torch_dtype
                        )
            else:
                if load_in_4bit:
                    assert (
                        transformers.__version__ >= "4.30.0"
                    ), "load_in_4bit requires transformers >= 4.30.0"
                model_kwargs = {}
                if transformers.__version__ >= "4.30.0":
                    model_kwargs["load_in_4bit"] = load_in_4bit
                    if load_in_4bit:
                        if bnb_4bit_quant_type:
                            model_kwargs["bnb_4bit_quant_type"] = bnb_4bit_quant_type
                        if bnb_4bit_compute_dtype:
                            model_kwargs["bnb_4bit_compute_dtype"] = _get_dtype(
                                bnb_4bit_compute_dtype
                            )
                        if bnb_4bit_use_double_quant:
                            model_kwargs[
                                "bnb_4bit_use_double_quant"
                            ] = bnb_4bit_use_double_quant
                if re.search("qwen-72b", self._config._name_or_path.lower()):
                    model = self.AUTO_MODEL_CLASS.from_pretrained(
                        pretrained,
                        revision=revision + ("/" + subfolder if subfolder is not None else ""),
                        low_cpu_mem_usage=low_cpu_mem_usage,
                        device_map=device_map,
                        max_memory=max_memory,
                        offload_folder=offload_folder,
                        load_in_8bit=load_in_8bit,
                        trust_remote_code=trust_remote_code,
                        torch_dtype=torch_dtype,
                        **model_kwargs,
                        fp32=(bool(torch_dtype==torch.float32)),
                        fp16=(bool(torch_dtype==torch.float16)),
                    )
                else:
                    model = self.AUTO_MODEL_CLASS.from_pretrained(
                        pretrained,
                        revision=revision + ("/" + subfolder if subfolder is not None else ""),
                        low_cpu_mem_usage=low_cpu_mem_usage,
                        device_map=device_map,
                        max_memory=max_memory,
                        offload_folder=offload_folder,
                        load_in_8bit=load_in_8bit,
                        trust_remote_code=trust_remote_code,
                        torch_dtype=torch_dtype,
                        **model_kwargs
                    )
        else:
            from auto_gptq import AutoGPTQForCausalLM    # pylint: disable=E0401

            model = AutoGPTQForCausalLM.from_quantized(
                pretrained,
                model_basename=None if quantized == True else Path(quantized).stem,
                device_map=device_map,
                max_memory=max_memory,
                trust_remote_code=trust_remote_code,
                use_safetensors=True
                if quantized == True
                else quantized.endswith(".safetensors"),
                use_triton=gptq_use_triton,
                warmup_triton=gptq_use_triton,
                inject_fused_attention=inject_fused_attention,
            )
        return model

    def _create_auto_model_peft(
        self,
        *,
        model: transformers.PreTrainedModel,
        peft: str,
        revision: str,
        subfolder: str,
        load_in_4bit: Optional[bool] = False,
    ):
        if load_in_4bit:
            assert PEFT_VERSION >= "0.4.0", "load_in_4bit requires peft >= 0.4.0"
        model = self.AUTO_PEFT_CLASS.from_pretrained(
            model,
            peft,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
        )
        return model

    def _create_auto_tokenizer(
        self,
        *,
        pretrained: str,
        revision: str,
        subfolder: str,
        tokenizer: Optional[str] = None,
        trust_remote_code: Optional[bool] = False,
    ) -> transformers.PreTrainedTokenizer:
        """Returns a pre-trained tokenizer from a pre-trained tokenizer configuration."""
        tokenizer = self.AUTO_TOKENIZER_CLASS.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
            trust_remote_code=trust_remote_code,
        )
        try:
            tokenizer.pad_token = tokenizer.eos_token
        except:
            print("token.pad_token setting failed.")
        return tokenizer

    @property
    def add_special_tokens(self) -> bool:
        """Whether to include special tokens in encoded text. This should be
        determined by whether or not the model was trained with special tokens.
        TODO: Remove these conditionals once HuggingFace supports a way to
        check whether or not an arbitrary model was trained with special tokens.
        """
        if self._add_special_tokens is not None:
            return self._add_special_tokens
        elif self.model_format == "runtime":
            return True
        elif self.AUTO_MODEL_CLASS is transformers.AutoModelForCausalLM:
            return False
        elif self.AUTO_MODEL_CLASS is transformers.AutoModel:
            return False
        elif self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM:
            return True
        else:
            raise ValueError(
                "Could not determine `add_special_tokens` value from the model "
                "class. Set to `True` or `False` depending on whether the model "
                "was pre-trained with special tokens."
            )

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model.
        NOTE: Different model configurations have different max sequence length
        attribute names.
            - n_positions: (CTRLConfig, T5Config)
            - max_position_embeddings: (BartConfig, RoFormerConfig)
            - n_ctx: (GPT2Config)
        NOTE: For relative position encoded models you should specify the max
        sequence length of the model in the constructor via `max_length`.
        """
        if self._max_length is not None:
            return self._max_length
        # Try to get the sequence length from the model config.
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self._config, attr):
                return getattr(self._config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def batch_size(self) -> int:
        # TODO: Add adaptive batch size.
        return self._batch_size  # * gpus

    @property
    def device(self) -> Union[int, str, torch.device]:
        return self._device

    def tok_encode(self, string: str) -> TokenSequence:
        # TODO: Merge `tok_encode_batch` here.
        return self.tokenizer.encode(string, add_special_tokens=self.add_special_tokens)

    def tok_encode_batch(self, strings: List[str]) -> TokenSequence:
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )

    def tok_decode(self, tokens: torch.LongTensor) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def greedy_until(
        self, requests: List[Tuple[str, Union[List[str], str]]]
    ) -> List[str]:
        def _collate(x):
            tokens = self.tok_encode(x[0])
            return len(tokens), x[0]

        results = []
        reorder = utils.Reorderer(requests, _collate)

        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size

        for chunk in utils.chunks(
            tqdm(reorder.get_reordered(), disable=False),
            self.batch_size if self.batch_size != "auto" else adaptive_batch_size,
        ):
            context = [c[0] for c in chunk]
            request_args = chunk[0][1]
            stop = request_args.get("until", None)
            stop_sequences = stop if isinstance(stop, list) else [stop]
            max_generation_length = request_args.get("max_length", None)

            assert (
                isinstance(max_generation_length, int) or max_generation_length is None
            )
            assert isinstance(stop_sequences, list) or stop_sequences is None

            # TODO: Find a better way to handle stop sequences for 0-shot.
            if stop_sequences is None:
                until = [self.eot_token]
            else:
                until = stop_sequences + [self.eot_token]

            if max_generation_length is None:
                max_tokens = self.max_gen_toks
            else:
                max_tokens = max_generation_length

            token_context = self.tok_encode_batch(context)

            responses = self._model_generate(     # pylint: disable=E1123, E1120
                inputs=token_context,
                max_tokens=max_tokens,
                stop=until,
            )
            responses = self.tok_decode(responses.tolist())

            for response in responses:
                # Ensure the generated responses do not contain the stop sequences.
                for term in until:
                    response = response.split(term)[0]
                # partial caching
                self.cache_hook.add_partial("greedy_until", (context, until), response)
                results.append(response)
        return reorder.get_original(results)


class AutoCausalLM(HuggingFaceAutoLM):
    """Causal language modeling.
    You can find a set of supported models in the HF documentation:
    https://huggingface.co/docs/transformers/main/model_doc/auto#transformers.AutoModelForCausalLM
    """

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
    AUTO_PEFT_CLASS = peft.PeftModel

    def __init__(self, *args, pretrained, model_format, **kwargs):
        self.model_format = model_format
        if self.model_format == "runtime":
            from intel_extension_for_transformers.transformers import WeightOnlyQuantConfig
            use_gptq = kwargs.pop("use_gptq", False)
            self.woq_config = WeightOnlyQuantConfig(compute_dtype="int8", weight_dtype="int4", use_gptq=use_gptq)
        super().__init__(*args, pretrained=pretrained, model_format=model_format, **kwargs)

        if self.model_format == "runtime":
            from transformers import AutoTokenizer, TextStreamer
            from intel_extension_for_transformers.transformers import AutoModelForCausalLM
            self.runtime_model = AutoModelForCausalLM.from_pretrained(pretrained, quantization_config=self.woq_config)

        if self.model_format == "onnx":
            if not os.path.exists(os.path.join(pretrained, "decoder_model.onnx")) and \
               not os.path.exists(os.path.join(pretrained, "decoder_with_past_model.onnx")) and \
               not os.path.exists(os.path.join(pretrained, "decoder_model_merged.onnx")) and \
               not os.path.exists(os.path.join(pretrained, "model.onnx")):
                raise ValueError(
                "Couldn't find any ONNX model name in ['decoder_model.onnx', 'decoder_with_past_model.onnx', "\
                "'decoder_model_merged.onnx', 'model.onnx'] in {}.".format(
                    pretrained)
                )

            import optimum.version
            import onnxruntime as ort
            from transformers import PretrainedConfig
            from optimum.onnxruntime import ORTModelForCausalLM

            model_config = PretrainedConfig.from_pretrained(pretrained)
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            backend = kwargs.pop("backend", "default")
            backend = "CPUExecutionProvider" if backend == "default" else backend

            if Version(optimum.version.__version__) >= Version("1.14.0"):
                if os.path.exists(os.path.join(pretrained, "model.onnx")):
                    session = ORTModelForCausalLM.load_model(  # pylint: disable=E1123
                        os.path.join(pretrained, "model.onnx"),
                        session_options=sess_options,
                        provider=backend
                        )
                    inputs_names = [input.name for input in session.get_inputs()] # pylint: disable=E1101
                    key_value_input_names = [key for key in inputs_names if (".key" in key) or (".value" in key)]
                    use_cache = len(key_value_input_names) > 0

                    self.model = ORTModelForCausalLM(session,  # pylint: disable=E1120
                                                     model_config,
                                                     use_cache=True if use_cache else False,
                                                     use_io_binding=True if use_cache else False)
                else:
                    if os.path.exists(os.path.join(pretrained, "decoder_model_merged.onnx")):
                        session = ORTModelForCausalLM.load_model(  # pylint: disable=E1123
                        os.path.join(pretrained, "decoder_model_merged.onnx"),
                        session_options=sess_options,
                        provider=backend
                        )
                        self.model = ORTModelForCausalLM(session,  # pylint: disable=E1120
                                                         model_config,
                                                         use_cache=True)
                    elif os.path.exists(os.path.join(pretrained, "decoder_with_past_model.onnx")):
                        session = ORTModelForCausalLM.load_model(  # pylint: disable=E1123
                        os.path.join(pretrained, "decoder_with_past_model.onnx"),
                        session_options=sess_options,
                        provider=backend
                        )
                        self.model = ORTModelForCausalLM(session,  # pylint: disable=E1120
                                                         model_config,
                                                         use_cache=True)
                    elif os.path.exists(os.path.join(pretrained, "decoder_model.onnx")):
                        session = ORTModelForCausalLM.load_model(  # pylint: disable=E1123
                        os.path.join(pretrained, "decoder_model.onnx"),
                        session_options=sess_options,
                        provider=backend
                        )
                        self.model = ORTModelForCausalLM(session,  # pylint: disable=E1120
                                                        model_config,
                                                        use_cache=False,
                                                        use_io_binding=False)
            else:
                if os.path.exists(os.path.join(pretrained, "model.onnx")):
                    session = ORTModelForCausalLM.load_model(  # pylint: disable=E1123
                        os.path.join(pretrained, "model.onnx"),
                        session_options=sess_options,
                        provider=backend
                        )
                    inputs_names = session.get_inputs()
                    key_value_input_names = [key for key in inputs_names if (".key" in key) or (".value" in key)]
                    use_cache = len(key_value_input_names) > 0

                    self.model = ORTModelForCausalLM(session[0],  # pylint: disable=E1121
                                                     model_config,
                                                     pretrained,
                                                     use_cache=True if use_cache else False,
                                                     use_io_binding=True if use_cache else False,)
                else:
                    if os.path.exists(os.path.join(pretrained, "decoder_model_merged.onnx")):
                        sessions = ORTModelForCausalLM.load_model(  # pylint: disable=E1123
                            os.path.join(pretrained, "decoder_model_merged.onnx"),
                            session_options=sess_options,
                            provider=backend
                            )
                        self.model = ORTModelForCausalLM(sessions[0],  # pylint: disable=E1121
                                                        model_config,
                                                        pretrained,
                                                        use_cache=True)
                    elif os.path.exists(os.path.join(pretrained, "decoder_with_past_model.onnx")):
                        sessions = ORTModelForCausalLM.load_model(  # pylint: disable=E1123
                            os.path.join(pretrained, "decoder_model.onnx"),
                            os.path.join(pretrained, "decoder_with_past_model.onnx"),
                            session_options=sess_options,
                            provider=backend
                            )
                        self.model = ORTModelForCausalLM(sessions[0],  # pylint: disable=E1121
                                                        model_config,
                                                        pretrained,
                                                        sessions[1],
                                                        use_cache=True)
                    else:
                        sessions = ORTModelForCausalLM.load_model(  # pylint: disable=E1123
                            os.path.join(pretrained, "decoder_model.onnx"),
                            session_options=sess_options,
                            provider=backend
                            )
                        self.model = ORTModelForCausalLM(sessions[0],  # pylint: disable=E1121
                                                        model_config,
                                                        pretrained,
                                                        use_cache=False,
                                                        use_io_binding=False)

    def _create_auto_tokenizer(
        self,
        *,
        pretrained: str,
        revision: str,
        subfolder: str,
        tokenizer: Optional[str] = None,
        trust_remote_code: Optional[bool] = False,
    ) -> transformers.PreTrainedTokenizer:
        tokenizer = super()._create_auto_tokenizer(
            pretrained=pretrained,
            revision=revision,
            subfolder=subfolder,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
        )
        tokenizer.padding_side = "left"
        return tokenizer

    def _model_call(
        self, inputs: TokenSequence, labels: Optional[TokenSequence] = None
    ) -> TokenSequence:
        if hasattr(self._config, "_name_or_path") and self._config._name_or_path == "THUDM/chatglm-6b":
            input_bs, input_len = inputs.shape
            eos = torch.tensor([130001, 130004]).repeat(input_bs, 1)
            inputs = torch.cat((inputs, eos), 1)
        if hasattr(self._config, "_name_or_path") and self._config._name_or_path == "THUDM/chatglm2-6b":
            input_bs, input_len = inputs.shape
            bos = torch.tensor([64790, 64792]).repeat(input_bs, 1)
            inputs = torch.cat((bos, inputs), 1)
        if self.model_format == "runtime":
            out = self.runtime_model(inputs, reinit=True, logits_all=True)
            output = {"logits": torch.tensor(out).unsqueeze(0)}
        elif self.model_format != "onnx":
            output = self.model(inputs)
        else:
            inputs_names = [input.name for input in self.model.model.get_inputs()]
            if "position_ids" in inputs_names:
                # model is exported with optimum >= 1.14.0 with new input 'position_ids'
                input_shape = inputs.shape
                position_ids = torch.arange(0, input_shape[-1], dtype=torch.long).unsqueeze(0).view(-1, input_shape[-1])
                output = self.model(inputs, torch.ones(inputs.shape, dtype=torch.int64), position_ids)
            else:
                output = self.model(inputs, torch.ones(inputs.shape, dtype=torch.int64))
        if isinstance(output, tuple):
            return output[0]
        return output["logits"]

    def _model_generate(
        self,
        inputs: transformers.BatchEncoding,
        max_tokens: int,
        stop: Optional[List[str]] = None,
    ) -> TokenSequence:
        # Ensure that the context does not encroach into the `space`
        # for the generation.
        input_ids = inputs["input_ids"][:, self.max_gen_toks - self.max_length :]
        attention_mask = inputs["attention_mask"][
            :, self.max_gen_toks - self.max_length :
        ]
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, input_ids.shape[1], input_ids.shape[0]
        )

        generations = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # GPT style models require the `generate` `max_length` arg to include the
            # context length, so we instead set `max_new_tokens` which is the number
            # of new tokens to generate, excluding the current number of tokens.
            max_new_tokens=max_tokens,
            stopping_criteria=stopping_criteria,
            do_sample=False,
        )
        return utils.select_continuation_from_batch_left_padding(
            generations, max_context_size=inputs["input_ids"].size(1)
        )


class AutoSeq2SeqLM(HuggingFaceAutoLM):
    """Seq2Seq language modeling.
    You can find a set of supported models in the following documentation:
    https://huggingface.co/docs/transformers/main/model_doc/auto#transformers.AutoModelForSeq2SeqLM
    """

    AUTO_MODEL_CLASS = transformers.AutoModelForSeq2SeqLM
    AUTO_PEFT_CLASS = peft.PeftModel
    def __init__(self, *args, pretrained, model_format, **kwargs):
        super().__init__(*args, pretrained=pretrained, model_format=model_format, **kwargs)

        self.model_format = model_format
        if self.model_format == "onnx":
            if not os.path.exists(os.path.join(pretrained, "encoder_model.onnx")) or \
               (not os.path.exists(os.path.join(pretrained, "decoder_model.onnx")) and \
                not os.path.exists(os.path.join(pretrained, "decoder_model_merged.onnx"))):
                raise ValueError(
                    "Please ensure encoder_model.onnx and " \
                    "decoder_model(_merged).onnx are under {}.".format(pretrained)
                )

            import onnxruntime as ort
            from transformers import PretrainedConfig
            from optimum.onnxruntime import ORTModelForSeq2SeqLM

            model_config = PretrainedConfig.from_pretrained(pretrained)
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            backend = kwargs.pop("backend", "default")
            backend = "CPUExecutionProvider" if backend == "default" else backend
            if os.path.exists(os.path.join(pretrained, "decoder_model_merged.onnx")):
                sessions = ORTModelForSeq2SeqLM.load_model(
                                os.path.join(pretrained, 'encoder_model.onnx'),
                                os.path.join(pretrained, 'decoder_model_merged.onnx'),
                                provider=backend
                                )

                self.model = ORTModelForSeq2SeqLM(sessions[0],
                                                  sessions[1],
                                                  model_config,
                                                  pretrained,
                                                  use_cache=True)

            elif os.path.exists(os.path.join(pretrained, "decoder_with_past_model.onnx")):
                sessions = ORTModelForSeq2SeqLM.load_model(
                                os.path.join(pretrained, 'encoder_model.onnx'),
                                os.path.join(pretrained, 'decoder_model.onnx'),
                                os.path.join(pretrained, 'decoder_with_past_model.onnx'),
                                provider=backend
                                )

                self.model = ORTModelForSeq2SeqLM(sessions[0],
                                                  sessions[1],
                                                  model_config,
                                                  pretrained,
                                                  sessions[2],
                                                  use_cache=True)
            else:
                sessions = ORTModelForSeq2SeqLM.load_model(  # pylint: disable=E1120
                                os.path.join(pretrained, 'encoder_model.onnx'),
                                os.path.join(pretrained, 'decoder_model.onnx'),
                                provider=backend
                                )

                self.model = ORTModelForSeq2SeqLM(sessions[0],
                                                  sessions[1],
                                                  model_config,
                                                  pretrained,
                                                  use_cache=False,
                                                  use_io_binding=False)
    def loglikelihood(
        self, requests: List[Tuple[str, str]]
    ) -> List[Tuple[float, bool]]:
        new_requests = []
        for chunk in utils.chunks(requests, self.batch_size):
            context, continuation = zip(*chunk)

            # Fill empty contexts with the EOT token.
            context = [
                f"{self.eot_token}" if len(text) == 0 else text for text in context
            ]
            context_enc = self.tok_encode_batch(context)
            for key in context_enc:
                context_enc[key] = context_enc[key][:, -self.max_length :]

            # Remove leading whitespace introduced by the default
            # `text_target_separator` since the context and continuation
            # will not be concatenated as a single (decoder) input.
            continuation = [text.lstrip() for text in continuation]
            continuation_enc = self.tok_encode_batch(list(continuation))
            for key in continuation_enc:
                continuation_enc[key] = continuation_enc[key][:, -self.max_length :]

            new_requests.append(
                ((context, continuation), context_enc, continuation_enc)
            )
        return self._loglikelihood_tokens(new_requests)

    def loglikelihood_rolling(self, requests: List[Tuple[str, str]]) -> List[float]:
        loglikelihoods = []
        for (string,) in tqdm(requests):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )
            contexts, conts = utils.split_and_pad_windows(  # pylint: disable=E1101
                rolling_token_windows,
                pad_token_id=self.eot_token_id,
                max_seq_len=self.max_length,
            )
            # Manually create BatchEncoding tensors with attention masks as
            # expected by `self._model_call` in `self._loglikelihood_tokens`.
            contexts_enc = torch.Tensor(contexts).long()
            contexts_enc = transformers.tokenization_utils_base.BatchEncoding(
                {
                    "input_ids": contexts_enc,
                    "attention_mask": (contexts_enc != self.eot_token_id).long(),
                }
            )
            conts_enc = torch.Tensor(conts).long()
            conts_enc = transformers.tokenization_utils_base.BatchEncoding(
                {
                    "input_ids": conts_enc,
                    "attention_mask": (conts_enc != self.eot_token_id).long(),
                }
            )
            # TODO: Extract out this call so it only gets called once and also
            # somehow figure out partial caching for.
            rolling_token_windows_request = [
                ((contexts, conts), contexts_enc, conts_enc)
            ]
            string_nll = self._loglikelihood_tokens(
                rolling_token_windows_request, disable_tqdm=True
            )
            string_nll = [x[0] for x in string_nll]  # discard is_greedy
            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)
        return loglikelihoods

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], TokenSequence, TokenSequence]],
        disable_tqdm: Optional[bool] = False,
    ) -> List[Tuple[float, bool]]:
        results = []
        for chunk in tqdm(
            requests, total=math.ceil(len(requests)), disable=disable_tqdm
        ):
            cache_keys, inputs_tokens, targets_tokens = chunk
            inputs_tokens = inputs_tokens.to(self.device)
            targets_tokens = targets_tokens.to(self.device)
            outputs = self._model_call(inputs=inputs_tokens, labels=targets_tokens)
            log_softmaxes = F.log_softmax(outputs.logits, dim=-1)

            output_iterator = zip(
                zip(cache_keys[0], cache_keys[1]),
                log_softmaxes,
                targets_tokens["input_ids"],
                targets_tokens["attention_mask"],
            )
            for cache_key, log_softmax, target_tokens, target_mask in output_iterator:
                length = target_mask.sum()
                log_softmax = log_softmax[:length]
                target_tokens = target_tokens[:length]
                greedy_tokens = log_softmax.argmax(dim=-1)
                max_equal = (greedy_tokens == target_tokens).all()
                target_logits = torch.gather(
                    log_softmax, 1, target_tokens.unsqueeze(-1)
                ).squeeze(-1)
                answer = (float(target_logits.sum()), bool(max_equal))
                results.append(answer)
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
        return results

    def _model_call(
        self, inputs: TokenSequence, labels: Optional[TokenSequence] = None
    ) -> TokenSequence:
        if self.model_format == "onnx":
            decoder_start_token_id = self._config.decoder_start_token_id
            pad_token_id = self._config.pad_token_id
            shifted_input_ids = labels["input_ids"].new_zeros(labels["input_ids"].shape)
            shifted_input_ids[..., 1:] = labels["input_ids"][..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id
            shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
            import pdb;pdb.set_trace()
            return self.model(**inputs, decoder_input_ids=shifted_input_ids, labels=labels["input_ids"])
        else:
            return self.model(**inputs, labels=labels["input_ids"])

    def _model_generate(
        self,
        inputs: transformers.BatchEncoding,
        max_tokens: int,
        stop: Optional[List[str]] = None,
    ) -> TokenSequence:
        input_ids = inputs["input_ids"][:, -self.max_length :].to(self.device)
        attention_mask = inputs["attention_mask"][:, -self.max_length :].to(self.device)

        # Generate one token to calculate the number of start tokens prepended to decoder_input_ids
        # (leaving this here in case the below assumption is violated in the future)
        # one_tok_gen = self.model.generate(
        #    input_ids=torch.zeros((1, 1), dtype=torch.int),
        #    min_length=2,
        #    max_new_tokens=1,
        # ).squeeze()
        # initial_decoder_input_length = len(one_tok_gen) - 1

        # Assume that there will always only be one token in the decoder inputs, assumption holds for existing HF models
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, 1, input_ids.shape[0]
        )

        generations = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            stopping_criteria=stopping_criteria,
            do_sample=False,
        )
        return generations


class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ):
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        self.sequence_id_len = len(self.sequence_ids)
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :][
            :, -self.sequence_id_len :
        ]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    stop_sequences: List[str],
    initial_decoder_input_length: int,
    batch_size: int,
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )
