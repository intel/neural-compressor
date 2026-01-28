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
from typing import Iterable, Optional, Union

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
from auto_round.schemes import QuantizationScheme

from neural_compressor.common.utils import Statistics
from neural_compressor.torch.algorithms import Quantizer
from neural_compressor.torch.algorithms.weight_only.utility import CapturedDataloader, InputCaptureModule
from neural_compressor.torch.utils import get_accelerator, logger


class AutoRoundQuantizer(Quantizer):
    """AutoRound Quantizer."""

    def __init__(
        self,
        quant_config: Optional[dict] = None,
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
        super().__init__(quant_config=quant_config)
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.accelerator = get_accelerator(kwargs.pop("device", "auto"))
        self.device = self.accelerator.name()

    def _is_w4afp8(self) -> bool:
        return self.data_type == "fp8_to_int_sym"

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
        # Remove AutoRound specific args before passing to AutoRound constructor
        keys_to_pop = ["quant_config", "device", "export_format", "output_dir", "accelerator", "reloading"]
        if hasattr(self, "target_bits") and self.target_bits is not None:
            from auto_round import AutoScheme

            self.scheme = AutoScheme(
                avg_bits=self.target_bits,
                options=self.options,
                shared_layers=self.shared_layers,
                ignore_scale_zp_bits=self.ignore_scale_zp_bits,
                method=self.auto_scheme_method,
                batch_size=self.auto_scheme_batch_size,
                device_map=self.auto_scheme_device_map,
                low_gpu_mem_usage=self.low_gpu_mem_usage,
            )
        # Remove AutoRound specific AutoScheme args before passing to AutoRound constructor
        keys_to_pop += [
            "target_bits",
            "options",
            "shared_layers",
            "ignore_scale_zp_bits",
            "auto_scheme_method",
            "auto_scheme_batch_size",
            "auto_scheme_device_map",
        ]

        rounder = AutoRound(
            model,
            tokenizer=tokenizer,
            **{k: v for k, v in self.__dict__.items() if k not in keys_to_pop},
        )

        if self._is_w4afp8():
            model, weight_config = rounder.quantize()
            model.autoround_config = weight_config
            return rounder.save_quantized(output_dir=self.output_dir, inplace=True)
        else:  # pragma: no cover
            rounder.quantize_and_save(output_dir=self.output_dir, format=self.export_format, inplace=True)
            model = rounder.model
            model.autoround_config = rounder.layer_config

        self.accelerator.empty_cache()
        dump_model_op_stats(rounder.layer_config)

        reloading = self.__dict__.get("reloading", True)
        if self.export_format in ["auto_round", "llm_compressor"] and reloading:
            # the directly returned model is QuantLinear, which is used for packing.
            try:
                logger.info(
                    f"Quantization is done, reloading model from saved directory({self.output_dir})...\n"
                    "Set reloading=False to skip."
                )
                import transformers  # pylint: disable=E0401

                model = transformers.AutoModelForCausalLM.from_pretrained(self.output_dir)
            except:
                pass

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


def dump_model_op_stats(layer_config):
    """Dump quantizable ops stats of model to user."""
    # TODO: collect more ops besides Linear
    res = {}
    res["Linear"] = {}
    for name, info in layer_config.items():
        if "data_type" in info:
            data_type_str = info["data_type"].upper()
            if "bits" in info and str(info["bits"]) not in info["data_type"]:
                data_type_str += str(info["bits"])
            res["Linear"][data_type_str] = res.get("Linear", {}).get(data_type_str, 0) + 1

    # update stats format for dump.
    field_names = ["Op Type", "Total"]
    dtype_list = list(res["Linear"].keys())
    field_names.extend(dtype_list)
    output_data = []
    for op_type in res.keys():
        field_results = [op_type, sum(res[op_type].values())]
        field_results.extend([res[op_type][dtype] for dtype in dtype_list])
        output_data.append(field_results)

    Statistics(output_data, header="Mixed Precision Statistics", field_names=field_names).print_stat()
