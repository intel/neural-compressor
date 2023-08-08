import torch
import copy
from neural_compressor.adaptor.torch_utils.util import fetch_module, get_example_input
from ...utils import logger
from .weight_only import get_absorb_layers
from .smooth_quant import model_forward
from functools import partial


def _get_block_prefix(model, module_types=[torch.nn.ModuleList]):
    for n, m in model.named_modules():
        if type(m) in module_types:
            block_prefix = n
            block_num = len(m)
    logger.debug(f"block_prefix: {block_prefix}")
    assert block_num > 0, "block num should't be zero!"
    return block_prefix, block_num

def _get_absorb_per_block(model, example_inputs):
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
    block_prefix, block_num = _get_block_prefix(model)
    for i in range(block_num):
        block_absorb_dict[i] = []
        block_name = block_prefix + '.' + str(i) + '.'
        for k, v in absorb_to_layer.items():
            name_list =[vv for vv in v if block_name in vv]
            if len(name_list) > 0:
                block_absorb_dict[i].append(name_list)
                absorb_layer_dict[tuple(name_list)] = k
        for k in no_absorb_layers:
            if block_name in k:
                block_absorb_dict[i].append([k])
                absorb_layer_dict[tuple([k])] = k
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
        total_block_args.append(args)
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




def awq_optmization(model, bits=4, group_size=32, scheme='asym', weight_config={}, 
                    dataloader=None, n_samples=128, calib_func=None, example_inputs=None):
    """_summary_

    Args:
        model (_type_): _description_
        bits (int, optional): _description_. Defaults to 4.
        group_size (int, optional): _description_. Defaults to 32.
        scheme (str, optional): _description_. Defaults to 'asym'.
        weight_config (dict, optional): _description_. Defaults to {}.
        dataloader (_type_, optional): _description_. Defaults to None.
        n_samples (int, optional): _description_. Defaults to 128.
        calib_func (_type_, optional): _description_. Defaults to None.
        example_inputs (_type_, optional): _description_. Defaults to None.
    """
    weight_config = copy.deepcopy(weight_config)
    if example_inputs is None:
        from .util import get_example_input
        example_inputs = get_example_input(dataloader)
    block_absorb_dict, absorb_layer_dict = _get_absorb_per_block(model, example_inputs)
    total_block_args, total_block_kwargs = _get_hidden_states(model, dataloader=None, n_samples=128, calib_func=None)
    block_prefix, block_num = _get_block_prefix(model)
    block_list = fetch_module(model, block_prefix)
    for i, module_list in block_absorb_dict.items():
        block = fetch_module(model, block_prefix + '.' + str(i))
        total_out = []
        for args, kwargs in zip(total_block_args, total_block_kwargs):
            out = block(*args, **kwargs)
            total_out.append(out)
