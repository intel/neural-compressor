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

import json
import time
from typing import Union

import torch
from auto_round import AutoRound  # pylint: disable=E0401
from auto_round.calib_dataset import get_dataloader  # pylint: disable=E0401
from auto_round.export.export_to_itrex.model_wrapper import WeightOnlyLinear  # pylint: disable=E0401
from auto_round.utils import get_block_names, get_module, quant_weight_w_scale, set_module  # pylint: disable=E0401

from neural_compressor.torch.algorithms import Quantizer
from neural_compressor.torch.utils import is_transformers_imported, logger

if is_transformers_imported():
    import transformers


def pack_model(
    model,
    weight_config: Union[str, dict],
    enable_full_range=False,
    compression_dtype=torch.int32,
    compression_dim=1,
    device="cpu",
    use_optimum_format=True,
    inplace=False,
    **kwargs,
):
    """Convert Linear to WeightOnlyLinear for low memory inference.

    Args:
        weight_config (str|dict): qconfig dict or Path of qconfig.json.
        enable_full_range (bool, optional): Whether to leverage the full compression range
                                            under symmetric quantization. Defaults to False.
        compression_dtype (torch.Tensor, optional): The target dtype after comoression.
                                                    Defaults to torch.int32.
        compression_dim (int, optional): Select from [0, 1], 0 is output channel,
                                            1 is input channel. Defaults to 1.
        device (str, optional): choose device for compression. Defaults to cpu.
        use_optimum_format (bool, optional): use the popular huggingface compression format.
            1: compression_dim: weight = 1, zeros = 0 and both are transposed.
            2: zeros -= 1 before compression. Why we need it?
            3: g_idx: use same number for one group instead of recording the channel order.
            4. parameter name changed, such as 'packed_weight' -> 'qweight'.
            5. zeros is always needed even for sym.
        inplace (bool, optional): Compress the model in place, or copy the model and compress it.

    xpu args:
        compression_dtype=torch.int8,
        compression_dim=0,
        use_optimum_format=False,
        scale_dtype=convert_dtype_str2torch(config.scale_dtype),
        device="xpu",
    """
    if inplace:
        compressed_model = model
    else:
        compressed_model = copy.deepcopy(model)
    if isinstance(weight_config, str):
        with open(weight_config, "r") as f:
            q_config = json.load(f)
    else:
        q_config = weight_config
    for k, v in q_config.items():
        logger.info(f"Packing {k}")
        if "float" in v["data_type"]:
            continue
        dtype = v["data_type"]
        num_bits = v["bits"]
        group_size = v["group_size"]
        sym = v["sym"]
        scale_dtype = v["scale_dtype"]
        m = get_module(compressed_model, k)
        fp_weight = m.weight.data
        scale, zp = v["scale"], v["zp"]
        convert_dtype = scale_dtype
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=convert_dtype)
            zp = torch.tensor(zp, dtype=torch.int32)
            if device == "xpu":
                scale = torch.tensor(v["scale"], dtype=torch.float32)
                zp = None if sym else torch.tensor(v["zp"], dtype=torch.int32)
        else:
            if not inplace:
                scale = scale.clone()
                zp = zp.clone()
            if device == "xpu":
                # Please note that for XPU, the scale data type is forcibly set to fp32
                scale = scale.to(dtype=torch.float32)
                zp = None if sym else zp.to(dtype=torch.int32)
            else:
                scale = scale.to(dtype=convert_dtype)
                zp = zp.to(dtype=torch.int32)
        if is_transformers_imported() and isinstance(m, transformers.Conv1D):
            fp_weight = fp_weight.t_().contiguous()
        int_weight = quant_weight_w_scale(fp_weight, scale, zp, group_size, fp_weight.device)
        int_weight = int_weight.type(torch.int32)
        if isinstance(m, torch.nn.Linear):
            in_features = m.in_features
            out_features = m.out_features
        elif is_transformers_imported() and isinstance(m, transformers.Conv1D):
            in_features = m.weight.shape[0]
            out_features = m.weight.shape[1]

        new_module = WeightOnlyLinear(
            in_features,
            out_features,
            num_bits,
            group_size,
            dtype=dtype,
            scale_dtype=scale_dtype,
            zp=zp is not None,
            bias=m.bias is not None,
            device="cuda" if device == "xpu" else device,
            compression_dtype=compression_dtype,
            compression_dim=compression_dim,
            use_optimum_format=use_optimum_format,  # xpu is False
        )
        new_module.pack(int_weight, scale, zp, m.bias)
        set_module(compressed_model, k, new_module)
    return compressed_model


class AutoRoundQuantizer(Quantizer):
    def __init__(
        self,
        quant_config: dict = {},
        enable_full_range: bool = False,
        batch_size: int = 8,
        amp: bool = True,
        device=None,
        lr_scheduler=None,
        enable_quanted_input: bool = True,
        enable_minmax_tuning: bool = True,
        lr: float = None,
        minmax_lr: float = None,
        low_gpu_mem_usage: bool = True,
        iters: int = 200,
        seqlen: int = 2048,
        n_samples: int = 512,
        sampler: str = "rand",
        seed: int = 42,
        n_blocks: int = 1,
        gradient_accumulate_steps: int = 1,
        not_use_best_mse: bool = False,
        dynamic_max_gap: int = -1,
        data_type: str = "int",
        scale_dtype: str = "fp16",
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
                    }
                    ...
                }
            keys:
                data_type (str): The data type to be used (default is "int").
                bits (int): Number of bits for quantization (default is 4).
                group_size (int): Size of the quantization group (default is 128).
                sym (bool): Whether to use symmetric quantization. (default is None).
        enable_full_range (bool): Whether to enable full range quantization (default is False).
        batch_size (int): Batch size for training (default is 8).
        amp (bool): Whether to use automatic mixed precision (default is True). Automatically detect and set.
        device: The device to be used for tuning (default is None). Automatically detect and set.
        lr_scheduler: The learning rate scheduler to be used.
        use_quant_input (bool): Whether to use quantized input data (default is True).
        enable_minmax_tuning (bool): Whether to enable min-max tuning (default is True).
        lr (float): The learning rate (default is 0.005).
        minmax_lr (float): The learning rate for min-max tuning (default is None).
        low_gpu_mem_usage (bool): Whether to use low GPU memory (default is True).
        iters (int): Number of iterations (default is 200).
        seqlen (int): Length of the sequence.
        n_samples (int): Number of samples (default is 512).
        sampler (str): The sampling method (default is "rand").
        seed (int): The random seed (default is 42).
        n_blocks (int): Number of blocks (default is 1).
        gradient_accumulate_steps (int): Number of gradient accumulation steps (default is 1).
        not_use_best_mse (bool): Whether to use mean squared error (default is False).
        dynamic_max_gap (int): The dynamic maximum gap (default is -1).
        scale_dtype (str): The data type of quantization scale to be used (default is "float16"), different kernels
                            have different choices.
        """
        super().__init__(quant_config)
        self.tokenizer = None
        self.enable_full_range = enable_full_range
        self.batch_size = batch_size
        self.amp = amp
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.enable_quanted_input = enable_quanted_input
        self.enable_minmax_tuning = enable_minmax_tuning
        self.lr = lr
        self.minmax_lr = minmax_lr
        self.low_gpu_mem_usage = low_gpu_mem_usage
        self.iters = iters
        self.seqlen = seqlen
        self.n_samples = n_samples
        self.sampler = sampler
        self.seed = seed
        self.n_blocks = n_blocks
        self.gradient_accumulate_steps = gradient_accumulate_steps
        self.not_use_best_mse = not_use_best_mse
        self.dynamic_max_gap = dynamic_max_gap
        self.data_type = data_type
        self.scale_dtype = scale_dtype

    def prepare(self, model: torch.nn.Module, *args, **kwargs):
        """Prepares a given model for quantization.
        Args:
            model (torch.nn.Module): The model to be prepared.

        Returns:
            A prepared model.
        """
        self.rounder = AutoRoundProcessor(
            model=model,
            tokenizer=None,
            weight_config=self.quant_config or {},
            enable_full_range=self.enable_full_range,
            batch_size=self.batch_size,
            amp=self.amp,
            device=self.device,
            lr_scheduler=self.lr_scheduler,
            enable_quanted_input=self.enable_quanted_input,
            enable_minmax_tuning=self.enable_minmax_tuning,
            lr=self.lr,
            minmax_lr=self.minmax_lr,
            low_gpu_mem_usage=self.low_gpu_mem_usage,
            iters=self.iters,
            seqlen=self.seqlen,
            n_samples=self.n_samples,
            sampler=self.sampler,
            seed=self.seed,
            n_blocks=self.n_blocks,
            gradient_accumulate_steps=self.gradient_accumulate_steps,
            not_use_best_mse=self.not_use_best_mse,
            dynamic_max_gap=self.dynamic_max_gap,
            data_type=self.data_type,
            scale_dtype=self.scale_dtype,
        )
        self.rounder.prepare()
        return model

    def convert(self, model: torch.nn.Module, *args, **kwargs):
        model, weight_config = self.rounder.convert()
        model.autoround_config = weight_config
        model = pack_model(model, weight_config, inplace=True)
        return model


@torch.no_grad()
def get_autoround_default_run_fn(
    model,
    tokenizer,
    dataset="NeelNanda/pile-10k",
    n_samples=512,
    seqlen=2048,
    seed=42,
    bs=8,
):
    """Perform calibration for quantization.

    This method calibrates the model for quantization by processing a specified
    number of samples from the calibration dataset. It ensures that the data is
    properly formatted and feeds it to the model. If the number of samples processed
    is less than the specified number, it logs a warning. If no samples are processed,
    it logs an error and exits.

    Args:
        n_samples (int): The number of samples to use for calibration.
    """

    if isinstance(dataset, str):
        dataset = dataset.replace(" ", "")  ##remove all whitespaces
        dataloader = get_dataloader(
            tokenizer,
            seqlen,
            dataset,
            seed,
            bs,
            n_samples,
        )
    else:
        dataloader = dataset
    total_cnt = 0
    for data in dataloader:
        if data is None:
            continue
        if isinstance(data, torch.Tensor):
            input_ids = data.to(model.device)
            data_new = input_ids

        elif isinstance(data, str):
            if tokenizer is None:
                logger.error("please provide tokenizer for string input")
                exit()
            data = tokenizer(data, truncation=True, max_length=seqlen, return_tensors="pt").data
            data_new = {}
            for key in data.keys():
                data_new[key] = data[key].to(model.device)
            input_ids = data_new["input_ids"]
        else:
            data_new = {}
            for key in data.keys():
                data_new[key] = data[key].to(model.device)
            input_ids = data_new["input_ids"]
        if input_ids.shape[-1] < seqlen:
            continue

        try:
            if isinstance(data_new, torch.Tensor):
                model(data_new)
            else:
                model(**data_new)
        except NotImplementedError:
            pass
        except Exception as error:
            logger.error(error)
        total_cnt += input_ids.shape[0]
        if total_cnt >= n_samples:
            break
    if total_cnt == 0:
        logger.error(
            f"no data has been cached, please provide more data with sequence length >={self.seqlen} in the "
            f"dataset or decease the sequence length"
        )
        exit()
    elif total_cnt < n_samples:
        logger.warning(
            f"Insufficient number of samples collected may affect the quantification. "
            f"Valid samples size:{total_cnt}, Target sample size:{n_samples}"
        )


class AutoRoundProcessor(AutoRound):
    @torch.no_grad()
    def prepare(self):
        """Prepares a given model for quantization."""
        self.block_names = get_block_names(self.model)
        if len(self.block_names) == 0:
            logger.warning("could not find blocks, exit with original model")
            return self.model, self.weight_config

        if self.amp:
            self.model = self.model.to(self.amp_dtype)

        self.layer_names = self.get_quantized_layer_names_outside_blocks()
        self.start_time = time.time()
        # all_inputs = self.try_cache_inter_data_gpucpu([block_names[0]], self.n_samples, layer_names=layer_names)

        # try_cache_inter_data_gpucpu
        # ([block_names[0]], self.n_samples, layer_names=layer_names)
        # self, block_names, n_samples, layer_names=[], last_cache_name=None
        last_cache_name = None
        cache_block_names = [self.block_names[0]]
        try:
            self.model = self.model.to(self.device)
            # all_inputs = self.cache_inter_data(
            #     block_names[0], self.n_samples, layer_names=layer_names, last_cache_name=last_cache_name
            # )
            # cache_inter_data cache_inter_data(self, block_names, n_samples, layer_names=[], last_cache_name=None):
            self.inputs = {}
            self.to_cached_layers = cache_block_names + self.layer_names
            self.tmp_dtype = None
            ## have bug if block name is not the first block
            if (len(cache_block_names) > 1 or len(self.layer_names) > 0) and self.low_gpu_mem_usage:
                tmp_dtype = self.model.dtype
                self.model = self.model.to(torch.bfloat16) if self.amp else self.model.to(torch.float32)

            self.last_cache_name = last_cache_name
            if last_cache_name is None and len(cache_block_names) + len(self.layer_names) == 1:
                self.last_cache_name = cache_block_names[0] if len(cache_block_names) == 1 else self.layer_names[0]
            # calib_bs = self.train_bs
            self.hook_handles = []
            self._replace_forward()
            self.prepared_gpu = True
            # self.calib(self.n_samples, calib_bs)

        except:
            logger.info("switch to cpu to cache inputs")
            self.model = self.model.to("cpu")
            torch.cuda.empty_cache()
            # all_inputs = self.cache_inter_data(
            #     self.block_names[0], self.n_samples, layer_names=self.layer_names, last_cache_name=last_cache_name
            # )
            self.inputs = {}
            self.to_cached_layers = cache_block_names + self.layer_names
            self.tmp_dtype = None
            ## have bug if block name is not the first block
            if (len(cache_block_names) > 1 or len(self.layer_names) > 0) and self.low_gpu_mem_usage:
                tmp_dtype = self.model.dtype
                self.model = self.model.to(torch.bfloat16) if self.amp else self.model.to(torch.float32)

            self.last_cache_name = last_cache_name
            if last_cache_name is None and len(cache_block_names) + len(self.layer_names) == 1:
                self.last_cache_name = cache_block_names[0] if len(cache_block_names) == 1 else self.layer_names[0]
            # calib_bs = self.train_bs
            self.hook_handles = []
            self._replace_forward()
            cache_block_names
            # self.calib(n_samples, calib_bs)

    def convert(self):
        """Converts a prepared model to a quantized model."""
        self._recover_forward()
        res = self.inputs
        del self.last_cache_name
        del self.to_cached_layers
        if self.tmp_dtype is not None:
            self.model = self.model.to(self.tmp_dtype)
        if self.prepared_gpu is True:
            self.model = self.model.to("cpu")

        all_inputs = res

        del self.inputs
        inputs = all_inputs[self.block_names[0]]

        all_inputs.pop(self.block_names[0])
        self.inputs = None
        del self.inputs
        if "input_ids" in inputs.keys():
            total_samples = len(inputs["input_ids"])
            self.n_samples = total_samples
            if total_samples < self.train_bs:
                self.train_bs = total_samples
                logger.warning(f"force the train batch size to {total_samples} ")

        self.model = self.model.to("cpu")
        torch.cuda.empty_cache()
        self.quant_blocks(
            self.model,
            inputs,
            self.block_names,
            n_blocks=self.n_blocks,
            device=self.device,
        )

        self.quant_layers(self.layer_names, all_inputs)

        self.dump_data_to_weight_config()

        end_time = time.time()
        cost_time = end_time - self.start_time
        logger.info(f"quantization tuning time {cost_time}")

        ## dump a summary
        quantized_layers = []
        unquantized_layers = []
        for n, m in self.model.named_modules():
            if isinstance(m, tuple(self.supported_types)):
                if self.weight_config[n]["bits"] == 16:
                    unquantized_layers.append(n)
                else:
                    quantized_layers.append(n)
        summary_info = (
            f"Summary: quantized {len(quantized_layers)}/{len(quantized_layers) + len(unquantized_layers)} in the model"
        )
        if len(unquantized_layers) > 0:
            summary_info += f",  {unquantized_layers} have not been quantized"
        logger.info(summary_info)

        self.quantized = True
        ##self.model = self.model.to(self.model_orig_dtype)##keep it as amp dtype
        return self.model, self.weight_config
