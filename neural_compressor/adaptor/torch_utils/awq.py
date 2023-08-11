import torch
import copy
from neural_compressor.adaptor.torch_utils.util import (
    fetch_module, 
    get_example_input, 
    get_absorb_layers,
    get_module_input_output,
)
from .model_wrapper import MulLinear
from ...utils import logger
from .smooth_quant import model_forward, set_module
from functools import partial


def _get_block_prefix(model, module_types=[torch.nn.ModuleList]):
    for n, m in model.named_modules():
        if type(m) in module_types:
            block_prefix = n
            block_num = len(m)
            logger.debug(f"block_prefix: {block_prefix}")
            break
    assert block_num > 0, "block num should't be zero!"
    return block_prefix, block_num

def _get_absorb_per_block(model, example_inputs, folding=False, weight_config={}):
    """Get absorbed layer per block. 

    Args:
        model (torch.nn.Module): input model
        example_inputs: example_inputs

    Returns:
        block_absorb_dict: dict of absorbed layer per block. eg. {0, [[absorbed_1, xx], [xx]], ...}
    """
    block_absorb_dict = {} # record absorbed layer per block
    absorb_layer_dict = {} # record absorb layers for absorbed layers
    absorb_to_layer, no_absorb_layers = get_absorb_layers(
        model, example_inputs, 
        supported_layers=['Linear'], folding=False
    )
    logger.debug(f"The no absorb layers: {no_absorb_layers}")
    # skip ops when algorithm is not AWQ
    skip_op_set = set()
    for k, v in absorb_to_layer.items():
        for vv in v:
            if vv in weight_config and weight_config[vv]['algorithm'] != 'AWQ':
                skip_op_set.add(k)
    for k in no_absorb_layers:
        if k in weight_config and weight_config[k]['algorithm'] != 'AWQ':
            skip_op_set.add(k)
    for k in skip_op_set:
        if k in absorb_to_layer:
            absorb_to_layer.pop(k)
        if k in no_absorb_layers:
            no_absorb_layers.remove(k)
    if len(skip_op_set) > 0:
        logger.info(f"{skip_op_set} are skipped when running AWQ optimization")

    block_prefix, block_num = _get_block_prefix(model)
    for i in range(block_num):
        block_absorb_dict[i] = []
        block_name = block_prefix + '.' + str(i) + '.'
        for k, v in absorb_to_layer.items():
            name_list =tuple(vv for vv in v if block_name in vv)
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



def calibration(model, dataloader=None, n_samples=128, calib_func=None):
    """ Calibration with dataloader or calib_func

    Args:
        model (torch.nn.Module): input model
        dataloader: dataloader. Defaults to None.
        n_samples (int, optional): n_samples. Defaults to 128.
        calib_func: calib_func. Defaults to None.
    """
    # calibration with dataloader or calib_func
    if calib_func is not None:
        calib_func(model)
    else:
        import math
        batch_size = dataloader.batch_size
        iters = int(math.ceil(n_samples / batch_size))
        if n_samples % batch_size != 0:
            logger.info("calibration samples increase from {} to {} due to batch_size is {}".format(
                n_samples, iters*batch_size, batch_size,
            ))
        model_forward(model, dataloader, iters, next(model.parameters()).device)

def _get_hidden_states(model, dataloader=None, n_samples=128, calib_func=None):
    # Step 1: replace block_forward to collect block inputs and avoid entire inference
    total_block_args = []
    total_block_kwargs = []
    def forward(layer, *args, **kwargs):
        # update total_hidden_states, total_block_kwargs, per batch
        total_block_args.append(list(args))
        total_block_kwargs.append(kwargs)
        raise ValueError

    block_prefix, block_num = _get_block_prefix(model)
    block_list = fetch_module(model, block_prefix)
    first_block = block_list[0]
    block_forward_cache = first_block.forward
    first_block.forward = partial(forward, first_block)

    # Step 2: replace model_forward to avoid ValueError
    model_forward_cache = model.forward
    def model_forward(model, *args, **kwargs):
        nonlocal model_forward_cache
        try:
            model_forward_cache(*args, **kwargs)
        except ValueError:
            pass
    model.forward = partial(model_forward, model)

    # Step 3: execute calibration
    calibration(model, dataloader=dataloader, n_samples=128, calib_func=calib_func)
    logger.info("The hidden_states collection is done.")

    # Step 4: recover model and block forward
    model.forward = model_forward_cache
    first_block.forward = block_forward_cache
    return total_block_args, total_block_kwargs


@torch.no_grad()
def _get_weight_scale(weight, q_group_size=-1):
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
    def __init__(self, model, example_inputs=None, calib_func=None, dataloader=None, n_samples=128,
                 bits=4,  group_size=32, scheme='asym', sym_full_range=False, weight_config={},):
        self.example_inputs = example_inputs
        if example_inputs is None:
            assert dataloader is not None, "datalaoder or example_inputs is required."
            self.example_inputs = get_example_input(dataloader)
        # Step 1: get hidden states and kwargs of first block. 
        self.total_block_args, self.total_block_kwargs = _get_hidden_states(
            model, dataloader=dataloader, n_samples=n_samples, calib_func=calib_func
        )
        # Step 2: get block list and block prefix, number
        self.block_prefix, self.block_num = _get_block_prefix(model)
        self.block_list = fetch_module(model, self.block_prefix)
        self.bits = bits
        self.group_size = group_size
        self.scheme = scheme
        self.sym_full_range = sym_full_range
        self.weight_config = weight_config
        self.model = model

    def quantize(self, auto_scale=True, mse_range=True, folding=False, return_int=False):
        # Step 1: get absorbed module list per block, includes self-absorption
        # block_absorb_dict is split per block, includes all absorb relationship.
        # absorb_layer_dict is the inverse of block_absorb_dict for all blocks
        self.block_absorb_dict, self.absorb_layer_dict = _get_absorb_per_block(
            self.model, self.example_inputs, 
            # for only mse_range, folding is useless.
            folding = folding if auto_scale else False 
        )
        # process per block
        for i, module_list in self.block_absorb_dict.items():
            logger.info(f"Processing block: {i}")
            # Step 1: fetch all input values of each linear for scale calculation
            # use the first linear for QKV tuple
            block_name = self.block_prefix + '.' + str(i)
            block = fetch_module(self.model, block_name)
            module_hook_config = {
                v[0].split(block_name + '.')[1]: ['input'] for v in module_list
            }
            def block_calibration(model):
                for args, kwargs in zip(self.total_block_args, self.total_block_kwargs):
                    model(*args, **kwargs)
            input_values = get_module_input_output(
                block, module_hook_config, calib_func=block_calibration,
            )
            # Step 3: search best scale for linears in one block and apply it
            if auto_scale:
                scale_info = self.search_scale(block, block_name, module_list, input_values)
            # Step 2: update self.total_block_args, self.total_block_kwargs for next block
            out_list = self.block_inference(block)
            self.update_block_input(out_list)
            # Step 4: get input of next block before update scale
            # weights of linear is updated by scale
            if auto_scale:
                self.apply_scale(scale_info)
            # Step 5: search best clip range for linears in one block and save to weight_config
            if mse_range:
                self.search_clip(block_name, module_list, input_values)
        # Step 6: apply clip range in weight_config when quantizing model weights
        self.apply_quantize_with_clip(return_int)
        return self.model

    def search_scale(self, block, block_name, module_list, input_values):
        from .weight_only import quant_weight
        scale_info = {}
        logger.info("Searching best scales with AWQ algorithm")
        for module_tuple in module_list:
            logger.info(f"[SCALE] Processing module: {module_tuple}")
            # Step 1: Initailize quantization configuration.
            if module_tuple[0] in self.weight_config:
                cur_bits = self.weight_config[module_tuple[0]]['bits']
                cur_group_size = self.weight_config[module_tuple[0]]['group_size']
                cur_scheme = self.weight_config[module_tuple[0]]['scheme']
            else:
                cur_bits, cur_group_size, cur_scheme = self.bits, self.group_size, self.scheme
            # Step 2: update module name in block
            module_name_list = [i.split(block_name + '.')[1] for i in module_tuple]
            # Step 3: collect w_max and x_max for scale calculation.
            weight = torch.cat(
                [fetch_module(block, _m).weight for _m in module_name_list], dim=0
            )
            w_max = _get_weight_scale(weight, q_group_size=cur_group_size)
            del weight
            input_val = input_values[module_name_list[0]]['input']
            x_max = _get_act_scale(input_val)
            absorbed_modules = {_m: fetch_module(block, _m) for _m in module_name_list}
            # Step 4: collect origin output for MSE and state_dict for recover.
            org_stat = {_m: module.state_dict() for _m, module in absorbed_modules.items()}
            if len(module_tuple) > 1:
                # use block inference for multi-modules
                org_out = self.block_inference(block)
            else:
                module = absorbed_modules[module_name_list[0]]
                org_out = self.module_inference(module, input_val)
            # Step 5: collect origin output for MSE and state_dict for recover.
            best_error = float('inf')
            best_scales = None
            best_scale_alpha = None
            n_grid = 20
            history = []
            # Step 6: set different alpha for scale and compare the MSE loss.
            for ratio in range(n_grid):
                ratio = ratio * 1 / n_grid
                scales = (x_max.pow(ratio) / w_max.pow(1-ratio)
                        ).clamp(min=1e-4).view(-1)
                scales = scales / (scales.max() * scales.min()).sqrt()
                for name, module in absorbed_modules.items():
                    module.weight.data = module.weight.data.mul(scales.view(1, -1))
                    module.weight.data = quant_weight(
                        module.weight.data,
                        num_bits=cur_bits, 
                        group_size=cur_group_size, 
                        scheme=cur_scheme,
                        full_range=self.sym_full_range,
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
                if absorb_module.bias is not None:
                    absorb_module.bias.div_(scale.view(-1))
                for name in module_tuple:
                    absorbed_module = fetch_module(self.model, name)
                    absorbed_module.weight.mul_(scale.view(1, -1))

    def search_clip(self, block_name, module_list, input_values):
        from .weight_only import quant_weight
        logger.info("Searching the best clip range with AWQ algorithm")
        for module_tuple in module_list:
            input_val = input_values[module_tuple[0].split(block_name + '.')[1]]['input']
            # process linear modules one by one
            for module_name in module_tuple:
                logger.info(f"[CLIP] Processing module: {module_name}")
                # Step 1: Initailize quantization configuration.
                if module_name in self.weight_config:
                    cur_bits = self.weight_config[module_name]['bits']
                    cur_group_size = self.weight_config[module_name]['group_size']
                    cur_scheme = self.weight_config[module_name]['scheme']
                else:
                    cur_bits, cur_group_size, cur_scheme = self.bits, self.group_size, self.scheme
                # Step 2: update module name
                module = fetch_module(self.model, module_name)
                # Step 3: collect origin output for MSE and state_dict for recover.
                org_stat = module.state_dict()
                org_out = self.module_inference(module, input_val)
                # Step 4:  set different clip range for weight and compare the MSE loss.
                logger.info("Searching the best clip range with AWQ algorithm")
                best_error = float('inf')
                best_clip_ratio = None
                n_grid = 100
                max_shrink = 0.1
                history = []
                for i_s in range(int(max_shrink * n_grid)):
                    ratio = (1 - i_s / n_grid) # 1, 0.91-1.0
                    # MulLinear can also work with @weight.setter
                    module.weight.data = quant_weight(
                        module.weight.data,
                        num_bits=cur_bits, 
                        group_size=cur_group_size, 
                        scheme=cur_scheme,
                        full_range=self.sym_full_range,
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
                        'bits': cur_bits,
                        'group_size': cur_group_size,
                        'scheme': cur_scheme
                    }
                self.weight_config[module_name]['quantile'] = best_clip_ratio
                if isinstance(module, MulLinear):
                    self.weight_config[module_name+'.linear'] = self.weight_config[module_name]
                    self.weight_config.pop(module_name)
                logger.debug("The best clip ratio for {}:{}".format(module_name, best_clip_ratio))

    def apply_quantize_with_clip(self, return_int=False):
        # apply quantization and clip
        logger.info("Quantizing the AWQ optimized fp32 model")
        from .weight_only import rtn_quantize
        self.model = rtn_quantize(
            self.model, 
            num_bits=self.bits, 
            group_size=self.group_size,
            scheme=self.scheme,
            weight_config=self.weight_config, 
            return_int=return_int,
            sym_full_range=self.sym_full_range,
        )
        logger.info("AWQ quantization is done.")

    def update_block_input(self, input_list):
        for i, inp in enumerate(input_list):
            if len(self.total_block_args[i]) > 0:
                self.total_block_args[i][0] = inp
            elif 'hidden_states' in self.total_block_kwargs[i]:
                self.total_block_kwargs[i]['hidden_states'] = inp
            else:
                assert False, "cannot find hidden_states position for next block"

    def block_inference(self, model):
        total_out = []
        for args, kwargs in zip(self.total_block_args, self.total_block_kwargs):
            out = model(*args, **kwargs)
            if isinstance(out, tuple):
                out = out[0]
            total_out.append(out)
        return total_out

    def module_inference(self, model, inputs):
        total_out = []
        for inp in inputs:
            out = model(inp)
            if isinstance(out, tuple):
                out = out[0]
            total_out.append(out)
        return total_out
