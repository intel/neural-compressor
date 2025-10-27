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
"""AutoRound quantization."""
import copy
import json
import time
from functools import lru_cache
from typing import Optional, Union

import torch


@lru_cache(None)
def _is_auto_round_available():
    try:
        import auto_round  # pylint: disable=E0401
    except ImportError:
        logger.error("AutoRound is not installed. Please install it using 'pip install auto_round'.")
        return False
    return True


_is_auto_round_available()

from auto_round import AutoRound, AutoRoundMLLM  # pylint: disable=E0401
from auto_round.compressors.mllm.eval import lmms_eval, mllm_eval
from auto_round.compressors.mllm.template import Template, get_template
from auto_round.export.export_to_itrex.export import pack_model  # pylint: disable=E0401
from auto_round.schemes import QuantizationScheme

from neural_compressor.torch.algorithms import Quantizer
from neural_compressor.torch.utils import get_accelerator, logger

from .utility import CapturedDataloader, InputCaptureModule


class AutoRoundQuantizer(Quantizer):
    """AutoRound Quantizer."""

    def __init__(
        self,
        bits: int = None,
        group_size: int = None,
        sym: bool = None,
        data_type: str = None,
        act_bits: int = None,
        act_group_size: int = None,
        act_sym: bool = None,
        act_data_type: str = None,
        act_dynamic: bool = None,
        super_bits: int = None,
        super_group_size: int = None,
        quant_config: dict = {},  # for INC
        layer_config: dict[str, Union[str, dict, QuantizationScheme]] = None,
        enable_full_range: bool = False,  ##for symmetric, TODO support later
        batch_size: int = 8,
        amp: bool = True,
        device_map: str = None,
        quant_lm_head: bool = False,
        lr_scheduler=None,
        dataset: Union[str, list, tuple, torch.utils.data.DataLoader] = "NeelNanda/pile-10k",
        enable_quanted_input: bool = True,
        enable_minmax_tuning: bool = True,
        lr: float = None,
        minmax_lr: float = None,
        low_gpu_mem_usage: bool = False,
        iters: int = 200,
        seqlen: int = 2048,
        nsamples: int = 128,
        sampler: str = "rand",
        seed: int = 42,
        nblocks: int = 1,
        gradient_accumulate_steps: int = 1,
        not_use_best_mse: bool = False,
        dynamic_max_gap: int = -1,
        scale_dtype: str = "fp16",
        to_quant_block_names: list = None,
        low_cpu_mem_usage: bool = False,
        export_format: str = "itrex",
        # v0.4
        enable_norm_bias_tuning: bool = False,
        enable_torch_compile: bool = None,
        # mllm
        quant_nontext_module: bool = False,
        extra_data_dir: str = None,
        image_processor=None,
        processor=None,
        template: Union[str, Template] = None,
        truncation: bool = False,
        # 0.7
        scheme: Union[str, dict, QuantizationScheme] = "W4A16",
        # diffusion
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        generator_seed: int = None,
        **kwargs,
    ):
        """Init a AutQRoundQuantizer object.

        Args:
            quant_config (dict): Configuration for weight quantization (default is None).
            quant_config={
                        'layer1':##layer_name
                        {
                            'data_type': 'int',
                            'bits': 4,
                            'group_size': 32,
                            'sym': False,
                            'act_data_type': None,
                            'act_bits': 32,
                            'act_sym': None,
                            'act_dynamic': True,
                        }
                        ...,
                    }
                keys:
                    data_type (str): The data type to be used (default is "int").
                    bits (int): Number of bits for quantization (default is 4).
                    group_size (int): Size of the quantization group (default is 128).
                    sym (bool): Whether to use symmetric quantization. (default is None).
            layer_config (dict, optional): Layer-wise quantization config. Defaults to None.
            bits (int): Number of bits for quantization (default is 4).
            group_size (int): Size of the quantization group (default is 128).
            sym (bool): Whether symmetric quantization is to be used (default is False).
            enable_full_range (bool): Whether to enable full range quantization (default is False).
            batch_size (int): Batch size for training (default is 8).
            amp (bool): Whether to use automatic mixed precision (default is True).
            device_map: The device to be used for tuning (default is None).
            quant_lm_head (bool): Indicates whether quantize the lm_head layer in transformers. (default is False).
            lr_scheduler: The learning rate scheduler to be used.
            dataset (str): The default dataset name (default is "NeelNanda/pile-10k").
            enable_quanted_input (bool): Whether to use the output of the previous quantized block as
                                            the input for the current block (default is True).
            enable_minmax_tuning (bool): Whether to enable weight min-max tuning (default is True).
            lr (float): The learning rate (default is None, will be set to 1.0/iters).
            minmax_lr (float): The learning rate for min-max tuning
                                    (default is None, it will be set to lr automatically).
            low_gpu_mem_usage (bool): Whether to use low GPU memory (default is True).
            iters (int): Number of iterations (default is 200).
            seqlen (int): Data length of the sequence for tuning (default is 2048).
            nsamples (int): Number of samples (default is 128).
            sampler (str): The sampling method (default is "rand").
            seed (int): The random seed (default is 42).
            nblocks (int): Number of blocks (default is 1).
            gradient_accumulate_steps (int): Number of gradient accumulation steps (default is 1).
            not_use_best_mse (bool): Whether to use mean squared error (default is False).
            dynamic_max_gap (int): The dynamic maximum gap (default is -1).
            data_type (str): The data type to be used (default is "int").
            scale_dtype (str): The data type of quantization scale to be used (default is "float16"), different kernels
                have different choices.
            to_quant_block_names (list): A list whose elements are list of block's layer names to be quantized.
            act_bits (int): Number of bits for activation quantization. Default is 32.
            act_group_size (int): Group size for activation quantization. Default is None.
            act_sym (bool): Whether to use symmetric activation quantization. Default is None.
            act_dynamic (bool): Whether to use dynamic activation quantization. Default is True.
            enable_norm_bias_tuning (bool): Whether to enable fast norm/layer_bias tuning.
            enable_torch_compile (bool): Whether to enable torch compile to optimize quant_block/layer, torch>=2.6 True.
            quant_nontext_module (bool): Whether to quantize nontext module.
            extra_data_dir (str): The path for extra data such as images, audio or videos.
            processor (transformers.AutoProcessor): Any multi-modal model will require an object to encode or
                decode the data that groups several modalities (among text, vision and audio).
                This is handled by objects called processors, which group together two or more processing objects such
                as tokenizers (for the text modality), image processors (for vision) and feature extractors (for audio).
            image_processor (Processor): Image processor for special model like llava.
            template (Template): The template to specify process for different mllms.
            truncation (bool): Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            scheme (str| dict | QuantizationScheme ): A preset scheme that defines the quantization configurations.
            guidance_scale (float): Control how much the image generation process follows the text prompt.
                                    The more it is, the more closely it follows the prompt (default is 7.5).
            num_inference_steps (int): The reference number of denoising steps (default is 50).
            generator_seed (int): A seed that controls the initial noise for image generation (default is None).

        Returns:
            The quantized model.
        """
        super().__init__(quant_config)
        self.layer_config = layer_config
        self.output_dir = kwargs.pop("output_dir", "temp_auto_round")
        self.tokenizer = kwargs.pop("tokenizer", "Placeholder")  # for AutoRound initialization
        self.enable_full_range = enable_full_range
        self.bits = bits
        self.group_size = group_size
        self.sym = sym
        self.data_type = data_type
        self.act_bits = act_bits
        self.act_group_size = act_group_size
        self.act_sym = act_sym
        self.act_data_type = act_data_type
        self.act_dynamic = act_dynamic
        self.super_bits = super_bits
        self.super_group_size = super_group_size
        self.batch_size = batch_size
        self.amp = amp
        self.device = get_accelerator(kwargs.pop("device", "auto")).name()
        self.lr_scheduler = lr_scheduler
        self.dataset = dataset
        self.enable_quanted_input = enable_quanted_input
        self.enable_minmax_tuning = enable_minmax_tuning
        self.lr = lr
        self.minmax_lr = minmax_lr
        self.low_gpu_mem_usage = low_gpu_mem_usage
        self.iters = iters
        self.seqlen = seqlen
        self.nsamples = nsamples
        self.sampler = sampler
        self.seed = seed
        self.nblocks = nblocks
        self.gradient_accumulate_steps = gradient_accumulate_steps
        self.not_use_best_mse = not_use_best_mse
        self.dynamic_max_gap = dynamic_max_gap
        self.scale_dtype = scale_dtype
        self.to_quant_block_names = to_quant_block_names
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.export_format = export_format
        self.enable_norm_bias_tuning = enable_norm_bias_tuning
        self.enable_torch_compile = enable_torch_compile
        self.quant_nontext_module = quant_nontext_module
        self.extra_data_dir = extra_data_dir
        self.processor = processor
        self.image_processor = image_processor
        self.template = template
        self.truncation = truncation
        self.scheme = scheme
        self.device_map = device_map
        self.quant_lm_head = quant_lm_head
        self.enable_w4afp8 = self._is_w4afp8()
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.generator_seed = generator_seed

    def _is_w4afp8(self) -> bool:
        return any([v.get("data_type", None) == "fp8_to_int_sym" for v in self.quant_config.values()])

    def prepare(self, model: torch.nn.Module, *args, **kwargs):
        """Prepares a given model for quantization.

        Args:
            model (torch.nn.Module): The model to be prepared.

        Returns:
            A prepared model.
        """
        prepare_model = InputCaptureModule(model)
        return prepare_model

    def convert(self, model: torch.nn.Module, *args, **kwargs):
        """Convert the prepared model to a quantized model.

        Args:
            model (torch.nn.Module): the prepared model

        Returns:
            The quantized model.
        """
        pipe = kwargs.pop("pipeline", None)
        tokenizer = getattr(model.orig_model, "tokenizer", None)
        if tokenizer is not None:
            delattr(model.orig_model, "tokenizer")
        elif pipe is None:
            tokenizer = "Placeholder"
            self.dataset = CapturedDataloader(model.args_list, model.kwargs_list)
        model = model.orig_model
        if pipe is not None:
            model = pipe
        rounder = AutoRound(
            model,
            layer_config=self.layer_config,
            bits=self.bits,
            data_type=self.data_type,
            group_size=self.group_size,
            sym=self.sym,
            act_bits=self.act_bits,
            act_group_size=self.act_group_size,
            act_sym=self.act_sym,
            act_data_type=self.act_data_type,
            act_dynamic=self.act_dynamic,
            super_bits=self.super_bits,
            super_group_size=self.super_group_size,
            tokenizer=tokenizer,
            scheme=self.scheme,
            processor=self.processor,
            image_processor=self.image_processor,
            enable_full_range=self.enable_full_range,
            batch_size=self.batch_size,
            amp=self.amp,
            device_map=self.device_map,
            lr_scheduler=self.lr_scheduler,
            dataset=self.dataset,
            extra_data_dir=self.extra_data_dir,
            template=self.template,
            quant_nontext_module=self.quant_nontext_module,
            enable_quanted_input=self.enable_quanted_input,
            enable_minmax_tuning=self.enable_minmax_tuning,
            lr=self.lr,
            minmax_lr=self.minmax_lr,
            low_gpu_mem_usage=self.low_gpu_mem_usage,
            low_cpu_mem_usage=self.low_gpu_mem_usage,
            iters=self.iters,
            seqlen=self.seqlen,
            nsamples=self.nsamples,
            sampler=self.sampler,
            seed=self.seed,
            nblocks=self.nblocks,
            gradient_accumulate_steps=self.gradient_accumulate_steps,
            not_use_best_mse=self.not_use_best_mse,
            dynamic_max_gap=self.dynamic_max_gap,
            scale_dtype=self.scale_dtype,
            to_quant_block_names=self.to_quant_block_names,
            enable_norm_bias_tuning=self.enable_norm_bias_tuning,
            truncation=self.truncation,
            enable_torch_compile=self.enable_torch_compile,
            quant_lm_head=self.quant_lm_head,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            generator_seed=self.generator_seed,
        )

        if self.enable_w4afp8:
            model, weight_config = rounder.quantize()
            model.autoround_config = weight_config
            return rounder.save_quantized(output_dir=self.output_dir, inplace=True)
        elif "itrex" in self.export_format:
            model, weight_config = rounder.quantize()
            model.autoround_config = weight_config
            model = pack_model(model, weight_config, device=self.device, inplace=True)
        else:  # pragma: no cover
            rounder.quantize_and_save(output_dir=self.output_dir, format=self.export_format, inplace=True)
            model = rounder.model
            model.autoround_config = rounder.layer_config
        return model


def get_dataloader(tokenizer, seqlen, dataset_name="NeelNanda/pile-10k", seed=42, bs=8, nsamples=128):
    """Generate a DataLoader for calibration using specified parameters.

    Args:
        tokenizer (Tokenizer): The tokenizer to use for tokenization.
        seqlen (int): The exact sequence length. samples < seqlen will be dropped,
                      samples longer than seqlen will be truncated
        dataset_name (str, optional): The name of the dataset or datasets separated by commas.
                                     Defaults to "NeelNanda/pile-10k".
        split (str, optional): The data split to use. Defaults to None.
        seed (int, optional): The random seed for reproducibility. Defaults to 42.
        bs (int, optional): The batch size. Defaults to 4.
        nsamples (int, optional): The total number of samples to include. Defaults to 128.

    Returns:
        DataLoader: The DataLoader for the calibrated dataset.
    """
    from auto_round.calib_dataset import get_dataloader  # pylint: disable=E0401

    dataloader = get_dataloader(tokenizer, seqlen, dataset_name=dataset_name, seed=seed, bs=bs, nsamples=nsamples)
    return dataloader


def get_mllm_dataloader(
    model,
    tokenizer,
    template=None,
    processor=None,
    image_processor=None,
    dataset=None,
    extra_data_dir=None,
    seqlen=None,
    batch_size=8,
    split=None,
    apply_template=None,
    truncation=None,
    seed=42,
    nsamples=128,
    gradient_accumulate_steps=1,
    quant_nontext_module=False,
):
    """Generate a DataLoader for calibration using specified parameters.

    Args:
        template (Template): The template to specify process for different mllms.
        model (Model): The model to quantized.
        tokenizer (Tokenizer): The tokenizer to use for tokenization.
        Dataset_name (str): The name or path of the dataset.
        extra_data_dir (str): The path for extra data such as images, audio or videos.
        seqlen (int): The exact sequence length. samples < seqlen will be dropped,
                      samples longer than seqlen will be truncated
        bs (int, optional): The batch size. Defaults to 4.
        split (str, optional): The data split to use. Defaults to None.
        apply_template: Whether to apply chat template in tokenization.

    Returns:
        DataLoader: The DataLoader for the calibrated datasets.
    """
    from auto_round.calib_dataset import CALIB_DATASETS
    from auto_round.compressors.mllm.compressor import _only_text_test
    from auto_round.compressors.mllm.dataset import get_mllm_dataloader  # pylint: disable=E0401

    template = template if template is not None else model.config.model_type
    template = get_template(
        template, model=model, tokenizer=tokenizer, processor=processor, image_processor=image_processor
    )
    dataset = template.default_dataset if dataset is None else dataset
    if quant_nontext_module or (
        dataset in CALIB_DATASETS.keys() and not _only_text_test(model, tokenizer, "cpu", template.model_type)
    ):
        if quant_nontext_module:
            logger.warning(
                "Quantitative nontext module is not supported for plain text datasets,"
                "will use liuhaotian/llava_conv_58k with default config as an alternative."
            )
        else:
            logger.warning(
                f"{model.config.model_type} not support for {dataset},"
                " will use liuhaotian/llava_conv_58k with default config as an alternative."
            )
        dataset = "liuhaotian/llava_conv_58k"
        seqlen = 512 if seqlen is None else seqlen
        truncation = False
        gradient_accumulate_steps = batch_size * gradient_accumulate_steps
        batch_size = 1
        seed = 42  # The seed is fixed to 42 in transformers
    seqlen = 2048 if seqlen is None else seqlen  # set text only calibration default args
    truncation = True if truncation is None else truncation
    dataset = dataset.replace(" ", "")

    if nsamples % batch_size != 0:
        nsamples = (nsamples // batch_size + 1) * batch_size
        logger.warning(f"'nsamples' is not divisible by 'batch_size', will adjusted to {nsamples}")

    dataloader, batch_size, seqlen, gradient_accumulate_steps = get_mllm_dataloader(
        template=template,
        processor=processor,
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        dataset=dataset,
        extra_data_dir=extra_data_dir,
        seqlen=seqlen,
        bs=batch_size,
        seed=seed,
        truncation=truncation,
        nsamples=nsamples,
        gradient_accumulate_steps=gradient_accumulate_steps,
        quant_nontext_module=quant_nontext_module,
    )
    return dataloader, template, truncation, batch_size, gradient_accumulate_steps, seqlen, nsamples
