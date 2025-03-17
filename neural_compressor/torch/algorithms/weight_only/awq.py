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
"""AWQ quantization."""
# Copied from neural_compressor/adaptor/torch_utils/awq.py

import copy
from collections import OrderedDict

import torch

from neural_compressor.torch.algorithms import Quantizer
from neural_compressor.torch.utils import get_accelerator, logger

from .modules import MulLinear
from .utility import (
    fetch_module,
    get_absorb_layers,
    get_block_prefix,
    get_module_input_output,
    recover_forward,
    replace_forward,
    set_module,
)

__all__ = ["AWQQuantizer"]


def _get_absorb_per_block(model, example_inputs, folding=False, weight_config={}):
    """Get absorbed layer per block.

    Args:
        model (torch.nn.Module): input model.
        example_inputs (tensor/tuple/dict, optional): used to trace torch model.
        folding (bool, optional): whether only allow update scale when it can be fold
                                    to upper layer. Defaults to False.
        weight_config (dict, optional): the quantization configuration. Defaults to {}.

    Returns:
        block_absorb_dict: The dict of absorbed layer per block. eg. {0, [[absorbed_1, xx], [xx]], ...}
        absorb_layer_dict: The layer dict that scale can be absorbed. The dict is the inverse of
                                block_absorb_dict for all blocks.
    """
    block_absorb_dict = {}  # record absorbed layer per block
    absorb_layer_dict = {}  # record absorb layers for absorbed layers
    absorb_to_layer, no_absorb_layers = get_absorb_layers(
        model, example_inputs, supported_layers=["Linear"], folding=False
    )
    logger.debug(f"The no absorb layers: {no_absorb_layers}")
    # skip ops when algorithm is not AWQ
    skip_op_set = set()
    for k, v in absorb_to_layer.items():
        for vv in v:
            if vv in weight_config and weight_config[vv]["dtype"] == "fp32":
                skip_op_set.add(k)
    for k in no_absorb_layers:
        if k in weight_config and weight_config[k]["dtype"] == "fp32":
            skip_op_set.add(k)
    for k in skip_op_set:
        if k in absorb_to_layer:
            absorb_to_layer.pop(k)
        if k in no_absorb_layers:
            no_absorb_layers.remove(k)
    if len(skip_op_set) > 0:
        logger.info(f"{skip_op_set} are skipped when running AWQ optimization")

    block_prefix, block_num = get_block_prefix(model)
    for i in range(block_num):
        block_absorb_dict[i] = []
        block_name = block_prefix + "." + str(i) + "."
        for k, v in absorb_to_layer.items():
            name_list = tuple(vv for vv in v if block_name in vv)
            if len(name_list) > 0:
                block_absorb_dict[i].append(name_list)
                absorb_layer_dict[name_list] = k
        if not folding:
            for k in no_absorb_layers:
                if block_name in k:
                    name_list = tuple([k])
                    block_absorb_dict[i].append(name_list)
                    absorb_layer_dict[name_list] = k
    logger.debug(f"The absorbed layers per block: {block_absorb_dict}")
    logger.debug(f"The absorb_layer_dict: {absorb_layer_dict}")
    return block_absorb_dict, absorb_layer_dict


def _get_absorb_dict(model, absorb_layer_dict):
    """Get absorbed layer per block from absorbed layer dict.

    Args:
        model (torch.nn.Module): input model
        absorb_layer_dict (dict): The layer type dict that scale can be absorbed, default is {}.

    Returns:
        block_absorb_dict: dict of absorbed layer per block. eg. {0, [[absorbed_1, xx], [xx]], ...}
        new_absorb_layer_dict: The layer dict that scale can be absorbed. The dict is the inverse of
                                block_absorb_dict for all blocks.
    """
    block_absorb_dict = {}
    block_prefix, block_num = get_block_prefix(model)
    new_absorb_layer_dict = {}
    for i in range(block_num):
        block_absorb_dict[i] = []
        block_name = block_prefix + "." + str(i) + "."

        for k, v in absorb_layer_dict.items():

            if isinstance(v, str):
                name_list = (block_name + v,)
            else:
                name_list = tuple(block_name + vv for vv in v)
            block_absorb_dict[i].append(name_list)
            new_absorb_layer_dict[name_list] = block_name + k
    logger.debug(f"The absorbed layers per block: {block_absorb_dict}")
    logger.debug(f"The absorb_layer_dict: {absorb_layer_dict}")
    return block_absorb_dict, new_absorb_layer_dict


@torch.no_grad()
def _get_weight_scale(weight, q_group_size=-1):
    """Get scale for weight.

    Args:
        weight (tensor): input weight
        q_group_size (int, optional): how many elements share one scale/zp. Defaults to -1.

    Returns:
        scale: the scale of input weight.
    """
    org_shape = weight.shape
    if q_group_size > 0:
        weight = weight.view(-1, q_group_size)
    scale = weight.abs() / weight.abs().amax(dim=1, keepdim=True)
    scale = scale.view(org_shape)
    scale = scale.mean(0)
    return scale


@torch.no_grad()
def _get_act_scale(input_val):
    tmp = [x.abs().view(-1, x.shape[-1]) for x in input_val]
    tmp = torch.cat(tmp, dim=0)
    return tmp.mean(0)


class ActAwareWeightQuant:
    """Implementation of Activation-aware Weight quantization (AWQ) algo."""

    def __init__(
        self,
        model,
        example_inputs=None,
        data_type="int",
        bits=4,
        group_size=32,
        scheme="asym",
        use_full_range=False,
        weight_config={},
        total_block_args=[],
        total_block_kwargs=[],
        device="auto",
        absorb_layer_dict={},
    ):

        self.example_inputs = example_inputs
        self.model = model
        self.device = device
        self._move_model_and_data_to_device()
        self.total_block_args = total_block_args
        self.total_block_kwargs = total_block_kwargs
        # get block list and block prefix, number
        self.block_prefix, self.block_num = get_block_prefix(model)
        self.block_list = fetch_module(model, self.block_prefix)
        self.data_type = data_type
        self.bits = bits
        self.group_size = group_size
        self.scheme = scheme
        self.use_full_range = use_full_range
        self.weight_config = weight_config
        self.absorb_layer_dict = absorb_layer_dict

    def _move_model_and_data_to_device(self):
        # Put the model and example_inputs into target device
        device = get_accelerator(self.device).current_device_name()
        self.model.to(device)
        self.example_inputs = self.example_inputs.to(device)

    def quantize(self, use_auto_scale=True, use_mse_search=True, folding=False, return_int=False):
        """Execute AWQ quantization.

        Args:
            use_auto_scale (bool, optional): whether search scale. Defaults to True.
            use_mse_search (bool, optional): whether search clip range. Defaults to True.
            folding (bool, optional): whether only allow update scale when it can be fold
                                      to upper layer. Defaults to False.
            return_int (bool, optional): whether return int dtype with INCWeightOnlyLinear.
                                         Defaults to False.

        Returns:
            model: quantized model
        """
        # Step 1: get absorbed module list per block, includes self-absorption
        # block_absorb_dict is split per block, includes all absorb relationship.
        # absorb_layer_dict is the inverse of block_absorb_dict for all blocks
        if not self.absorb_layer_dict:
            self.block_absorb_dict, self.absorb_layer_dict = _get_absorb_per_block(
                self.model,
                self.example_inputs,
                # for only use_mse_search, folding is useless.
                folding=folding if use_auto_scale else False,
                weight_config=self.weight_config,
            )
        else:
            self.block_absorb_dict, self.absorb_layer_dict = _get_absorb_dict(self.model, self.absorb_layer_dict)
        # process per block
        for i, module_list in self.block_absorb_dict.items():
            logger.info(f"Processing block: {i+1}/{self.block_num}")
            if len(module_list) == 0:
                logger.info("No need to process this block.")
                continue
            # Step 1: fetch all input values of each linear for scale calculation
            # use the first linear for QKV tuple
            block_name = self.block_prefix + "." + str(i)
            block = fetch_module(self.model, block_name)
            module_hook_config = {v[0].split(block_name + ".")[1]: ["input"] for v in module_list}

            def block_calibration(model):
                for args, kwargs in zip(self.total_block_args, self.total_block_kwargs):
                    model(*args, **kwargs)

            input_values = get_module_input_output(
                block,
                module_hook_config,
                calib_func=block_calibration,
            )
            # Step 3: search best scale for linears in one block and apply it
            if use_auto_scale:
                scale_info = self.search_scale(block, block_name, module_list, input_values)
            # Step 2: update self.total_block_args, self.total_block_kwargs for next block
            out_list = self.block_inference(block)
            self.update_block_input(out_list)
            # Step 4: get input of next block before update scale
            # weights of linear is updated by scale
            if use_auto_scale:
                self.apply_scale(scale_info)
            # Step 5: search best clip range for linears in one block and save to weight_config
            if use_mse_search:
                self.search_clip(block_name, module_list, input_values)
        # Step 6: apply clip range in weight_config when quantizing model weights
        self.apply_quantize_with_clip(return_int)
        return self.model

    def search_scale(self, block, block_name, module_list, input_values):
        """Search scales per block.

        Args:
            block (torch.nn.Module): a block of model
            block_name (str): the block name in model.
            module_list (dict): contains all linear tuple in current block,
                                linears in the same tuple shares scale.
            input_values (dict): contains all input values of linears in current block

        Returns:
            scale_info: a dict that contains input scales of linears in current block
        """
        from .utility import quant_tensor

        scale_info = {}
        logger.info("Searching best scales with AWQ algorithm")
        for module_tuple in module_list:
            # Step 1: Initialize quantization configuration.
            if module_tuple[0] in self.weight_config:
                cur_dtype = self.weight_config[module_tuple[0]]["dtype"]
                cur_bits = self.weight_config[module_tuple[0]]["bits"]
                cur_group_size = self.weight_config[module_tuple[0]]["group_size"]
                cur_scheme = self.weight_config[module_tuple[0]]["scheme"]
            else:
                cur_dtype, cur_bits, cur_group_size, cur_scheme = (
                    self.data_type,
                    self.bits,
                    self.group_size,
                    self.scheme,
                )
            if cur_bits < 0:
                continue
            logger.info(f"[SCALE] Processing module: {module_tuple}")
            # Step 2: update module name in block
            module_name_list = [i.split(block_name + ".")[1] for i in module_tuple]
            # Step 3: collect w_max and x_max for scale calculation.
            weight = torch.cat([fetch_module(block, _m).weight for _m in module_name_list], dim=0)
            w_max = _get_weight_scale(weight, q_group_size=cur_group_size)
            del weight
            input_val = input_values[module_name_list[0]]["input"]
            x_max = _get_act_scale(input_val)
            absorbed_modules = {_m: fetch_module(block, _m) for _m in module_name_list}
            # Step 4: collect origin output for MSE and state_dict for recover.
            org_stat = {_m: copy.deepcopy(module.state_dict()) for _m, module in absorbed_modules.items()}
            if len(module_tuple) > 1:
                # use block inference for multi-modules
                org_out = self.block_inference(block)
            else:
                module = absorbed_modules[module_name_list[0]]
                org_out = self.module_inference(module, input_val)
            # Step 5: collect origin output for MSE and state_dict for recover.
            best_error = float("inf")
            best_scales = None
            best_scale_alpha = None
            n_grid = 20
            history = []
            # Step 6: set different alpha for scale and compare the MSE loss.
            for ratio in range(n_grid):
                ratio = ratio * 1 / n_grid
                scales = (x_max.pow(ratio) / w_max.pow(1 - ratio)).clamp(min=1e-4).view(-1)
                scales = scales / (scales.max() * scales.min()).sqrt()
                for name, module in absorbed_modules.items():
                    module.weight.data = module.weight.data.mul(scales.view(1, -1))
                    module.weight.data = quant_tensor(
                        module.weight.data,
                        data_type=cur_dtype,
                        num_bits=cur_bits,
                        group_size=cur_group_size,
                        scheme=cur_scheme,
                        full_range=self.use_full_range,
                    ) / scales.view(1, -1)
                loss = 0
                if len(module_tuple) > 1:
                    # use block inference for multi-modules
                    cur_out = self.block_inference(block)
                else:
                    module = absorbed_modules[module_name_list[0]]
                    cur_out = self.module_inference(module, input_val)
                for out1, out2 in zip(org_out, cur_out):
                    loss += (out1 - out2).float().pow(2).mean().item()
                history.append(loss)
                is_best = loss < best_error
                if is_best:
                    best_error = loss
                    best_scales = scales
                    best_scale_alpha = ratio
                for name, module in absorbed_modules.items():
                    module.load_state_dict(org_stat[name])
            # Step 7: record the best scale alpha of each module_tuple
            assert best_scales is not None, "Loss is infinity! Cannot find the correct scale."
            best_scales = best_scales.view(-1)
            assert torch.isnan(best_scales).sum() == 0, best_scales
            scales = best_scales.detach()
            scale_info[module_tuple] = scales
            logger.debug("The loss history of different scale:{}".format(history))
            logger.info("The best scale alpha of {}: {}".format(module_tuple, best_scale_alpha))
        return scale_info

    @torch.no_grad()
    def apply_scale(self, scale_info):
        """Apply scales to model.

        Args:
            scale_info (dict): a dict that contains input scales of linears in current block
        """
        for module_tuple, scale in scale_info.items():
            logger.debug(f"apply scale for module: {module_tuple}")
            assert module_tuple in self.absorb_layer_dict, "cannot find the absorb module."
            absorb_module_name = self.absorb_layer_dict[module_tuple]
            absorb_module = fetch_module(self.model, absorb_module_name)
            if absorb_module_name == module_tuple[0]:
                # Case 1: module is self-absorption
                new_module = MulLinear(absorb_module, 1.0 / scale)
                new_module._update_linear()
                set_module(self.model, absorb_module_name, new_module)
            else:
                # Case 2: scale is absorbed by other layer
                if len(absorb_module.weight.shape) == 1:
                    absorb_module.weight.div_(scale)  # for LayerNorm
                else:
                    absorb_module.weight.div_(scale.view(-1, 1))
                # hasattr is for LlamaRMSNorm
                if hasattr(absorb_module, "bias") and absorb_module.bias is not None:
                    absorb_module.bias.div_(scale.view(-1))
                for name in module_tuple:
                    absorbed_module = fetch_module(self.model, name)
                    absorbed_module.weight.mul_(scale.view(1, -1))

    def search_clip(self, block_name, module_list, input_values):
        """Search best clip range of each linears in current block.

        Args:
            block_name (str): block name in model.
            module_list (dict): contains all linear tuple in current block,
                                linears in the same tuple shares scale.
            input_values (dict): contains all input values of linears in current block
        """
        from .utility import quant_tensor

        logger.info("Searching the best clip range with AWQ algorithm")
        for module_tuple in module_list:
            input_val = input_values[module_tuple[0].split(block_name + ".")[1]]["input"]
            # process linear modules one by one
            for module_name in module_tuple:
                # Step 1: Initialize quantization configuration.
                if module_name in self.weight_config:
                    cur_dtype = self.weight_config[module_name]["dtype"]
                    cur_bits = self.weight_config[module_name]["bits"]
                    cur_group_size = self.weight_config[module_name]["group_size"]
                    cur_scheme = self.weight_config[module_name]["scheme"]
                else:
                    cur_dtype, cur_bits, cur_group_size, cur_scheme = (
                        self.data_type,
                        self.bits,
                        self.group_size,
                        self.scheme,
                    )
                if cur_bits < 0:
                    continue
                logger.info(f"[CLIP] Processing module: {module_name}")
                # Step 2: update module name
                module = fetch_module(self.model, module_name)
                # Step 3: collect origin output for MSE and state_dict for recover.
                org_stat = copy.deepcopy(module.state_dict())
                org_out = self.module_inference(module, input_val)
                # Step 4:  set different clip range for weight and compare the MSE loss.
                logger.info("Searching the best clip range with AWQ algorithm")
                best_error = float("inf")
                best_clip_ratio = None
                n_grid = 100
                max_shrink = 0.1
                history = []
                for i_s in range(int(max_shrink * n_grid)):
                    ratio = 1 - i_s / n_grid  # 1, 0.91-1.0
                    # MulLinear can also work with @weight.setter
                    module.weight.data = quant_tensor(
                        module.weight.data,
                        data_type=cur_dtype,
                        num_bits=cur_bits,
                        group_size=cur_group_size,
                        scheme=cur_scheme,
                        full_range=self.use_full_range,
                        quantile=ratio,
                    )
                    loss = 0
                    cur_out = self.module_inference(module, input_val)
                    for out1, out2 in zip(org_out, cur_out):
                        loss += (out1 - out2).float().pow(2).mean().item()
                    history.append(loss)
                    is_best = loss < best_error
                    if is_best:
                        best_error = loss
                        best_clip_ratio = ratio
                    module.load_state_dict(org_stat)
                logger.debug("The loss history of different clip range:{}".format(history))
                if module_name not in self.weight_config:
                    self.weight_config[module_name] = {
                        "bits": cur_bits,
                        "group_size": cur_group_size,
                        "scheme": cur_scheme,
                    }
                self.weight_config[module_name]["quantile"] = best_clip_ratio
                if isinstance(module, MulLinear):
                    self.weight_config[module_name + ".linear"] = self.weight_config[module_name]
                    self.weight_config.pop(module_name)
                logger.debug("The best clip ratio for {}:{}".format(module_name, best_clip_ratio))

    def apply_quantize_with_clip(self, return_int=False):
        """Quantize model with clip range.

        Args:
            return_int (bool, optional): whether return int dtype with INCWeightOnlyLinear.
                                         Defaults to False.
        """
        # apply quantization and clip
        logger.info("Quantizing the AWQ optimized fp32 model")
        from .rtn import RTNQuantizer

        rtn_quantizer = RTNQuantizer(quant_config=self.weight_config)

        self.model = rtn_quantizer.quantize(
            self.model,
            bits=self.bits,
            group_size=self.group_size,
            scheme=self.scheme,
            return_int=return_int,
            use_full_range=self.use_full_range,
        )
        logger.info("AWQ quantization is done.")

    def update_block_input(self, input_list):
        """Update block input for next block inference.

        Args:
            input_list (list): A list of previous block outputs to serve as input to the next block.
        """
        for i, inp in enumerate(input_list):
            if len(self.total_block_args[i]) > 0:
                self.total_block_args[i][0] = inp
            elif "hidden_states" in self.total_block_kwargs[i]:
                self.total_block_kwargs[i]["hidden_states"] = inp
            else:  # pragma: no cover
                assert False, "cannot find hidden_states position for next block"

    def block_inference(self, model):
        """Collect output of block.

        Args:
            model (torch.nn.Module): input model.

        Returns:
            output(list):  a list of block output.
        """
        total_out = []
        for args, kwargs in zip(self.total_block_args, self.total_block_kwargs):
            # to avoid layer_past: Dynamic_cache when transformers higher than 4.45.1
            if "layer_past" in kwargs.keys() and kwargs["layer_past"] is not None:
                kwargs["layer_past"] = None
            out = model(*args, **kwargs)
            if isinstance(out, tuple):  # pragma: no cover
                out = out[0]
            total_out.append(out)
        return total_out

    def module_inference(self, model, inputs):
        """Collect output of module.

        Args:
            model (torch.nn.Module): input model.
            inputs (list): a list of module input.

        Returns:
            output(list):  a list of module output.
        """
        total_out = []
        for inp in inputs:
            out = model(inp)
            if isinstance(out, tuple):  # pragma: no cover
                out = out[0]
            total_out.append(out)
        return total_out


class AWQQuantizer(Quantizer):
    """AWQ Quantizer."""

    def __init__(self, quant_config: OrderedDict = {}, absorb_layer_dict: dict = {}):
        """Init an AWQQuantizer object.

        Args:
            quant_config (OrderedDict, optional): quantization config for ops. Defaults to {}.
            absorb_layer_dict (dict): The layer dict that scale can be absorbed, default is {}.
        """
        super().__init__(quant_config)
        self.absorb_layer_dict = absorb_layer_dict

    @torch.no_grad()
    def prepare(self, model, *args, **kwargs):
        """Prepare a given model to get hidden states and kwargs of first block.

        Args:
            model: A float torch model.

        Returns:
            A prepared model.
        """
        assert isinstance(model, torch.nn.Module), "AWQ algorithm only supports torch module"
        model = replace_forward(model)
        return model

    @torch.no_grad()
    def convert(
        self,
        model,
        bits=4,
        group_size=32,
        scheme="asym",
        example_inputs=None,
        use_auto_scale=True,
        use_mse_search=True,
        folding=False,
        return_int=False,
        use_full_range=False,
        data_type="int",
        *args,
        **kwargs,
    ):
        """Converts a prepared model to a quantized model.

        Args:
            model: torch model.
            bits: num bits. Defaults to 4.
            group_size: how many elements share one scale/zp. Defaults to 32.
            scheme: sym or asym. Defaults to "asym".
            example_inputs: example_inputs. Defaults to None.
            use_auto_scale: whether enable scale for salient weight. Defaults to True.
            use_mse_search: whether enable clip for weight by checking mse. Defaults to True.
            folding: False will allow insert mul before linear when the scale cannot be absorbed
                by last layer, else won't. Defaults to False.
            return_int: Choose return fp32 or int32 model. Defaults to False.
            use_full_range: Choose sym range whether use -2**(bits-1). Defaults to False.
            data_type: data type. Defaults to "int".

        Returns:
            model: fake quantized model
        """
        model = recover_forward(model)
        total_block_args = getattr(model, "total_block_args", [])
        total_block_kwargs = getattr(model, "total_block_kwargs", [])
        delattr(model, "total_block_args")
        delattr(model, "total_block_kwargs")

        awq = ActAwareWeightQuant(
            model,
            example_inputs=example_inputs,
            data_type=data_type,
            bits=bits,
            group_size=group_size,
            scheme=scheme,
            use_full_range=use_full_range,
            weight_config=self.quant_config,
            total_block_args=total_block_args,
            total_block_kwargs=total_block_kwargs,
            absorb_layer_dict=self.absorb_layer_dict,
        )
        qdq_model = awq.quantize(
            use_auto_scale=use_auto_scale,
            use_mse_search=use_mse_search,
            folding=folding,
            return_int=return_int,
        )
        return qdq_model
