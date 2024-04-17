import torch
import torch.nn as nn
from typing import List, Type, Tuple
from collections.abc import Iterable
import pandas as pd
import os

from habana_frameworks.torch.hpex.kernels.Fp8Ops import cast_to_fp8_v2
from habana_frameworks.torch.hpex.kernels.Fp8Ops import cast_from_fp8
import habana_frameworks.torch as ht

from .._quant_common.quant_config import Fp8cfg, QuantMode, ScaleMethod
from . import QuantizedLinear, QuantizedConv2d, QuantizedModule


# get global config
config = Fp8cfg().cfg

# After statistics are collected - calculates scale from statistics. 
# reference - https://arxiv.org/pdf/2208.09225.pdf
# max scale returns the scale which is the largest power of 2 such that abs_max is inside of dynamic range
def statistics_to_scale(statistics, min_exp_bias= float('-inf'), max_exp_bias= float('inf')):
    if config['scale_method'] == ScaleMethod.MAX:
        if config['fp8_config'] == torch.float8_e4m3fn:
            exp_width = 4
        elif config['fp8_config'] == torch.float8_e5m2:
            exp_width = 5
        else:
            print("Warning! 'fp8_config' not indicated, using e4m3 by default")
            exp_width = 4

        orig_bias = 2**(exp_width-1)-1
        man_width = 7 - exp_width
        average_stat = torch.max(statistics)
        bias = (2 ** ((2 ** exp_width) - 2)) / (average_stat.clamp(min=1e-8) / (2 - (2 ** (-1 * man_width))))
        bias = (torch.log2(bias)).floor()
        bias = bias.clamp(min=min_exp_bias, max=max_exp_bias)
        scale = 2.**(bias - orig_bias)

    elif config['scale_method'] == ScaleMethod.WITHOUT_SCALE or config['scale_method'] == ScaleMethod.UNIT_SCALE:
        scale = 1

    else:
        raise TypeError("Select scale method in config.py")
    return scale


# Receives as input a model, and a path to the collected statistics
# Initiates the quantization, by setting the scale using statistics_to_scale
# and turning model.quant to true for all modules not in blocklist
def quant_model(model, types=QuantizedModule, path = '', blocklist = [], std = 0):
    print(config)
    if not path:
        path = config['dump_stats_path']
    stats = torch.load(path)
    if blocklist == []:
        blocklist = config['blocklist']

    named_modules = get_named_modules(model,types=types)
    for module in named_modules:
        curr_stats = stats[module[0]]
        scale = statistics_to_scale(torch.Tensor(curr_stats))
        module[1].set_scale(scale)

        if config['check_std']['check']:
            module[1].set_std(torch.Tensor(curr_stats))

        module[1].start_quant()

    if config['check_std']['check']:
        std = config['check_std']['check']
        block_by_std(model, std)

    if not config['fake_quant']:
        prequantize_weights(model)

def prequantize_weights(model):
    named_modules = get_named_modules(model,types = [QuantizedLinear, QuantizedConv2d])
    for module in named_modules:
        module[1].quantize_weights()


# receieves a model and a limit std. removes a tensor from quantization if the
# std of a_max of a certain tensor in measurement stage surpassed limit std (by turning model.quant to false)
def block_by_std(model, std):
    named_modules = get_named_modules(model, types=QuantizedModule)
    for name, module in named_modules:
        if module.std > std:
            print('removed ' + name + ' from quantization because of std')
            module.stop_quant()

# Iterates over all modules and returns a filter with name and module of all modules with type in types
def get_named_modules(module:torch.nn.Module, types:Tuple[Type[torch.nn.Module]]=None) -> List[torch.nn.Module]:
    named_modules = module.named_modules()
    if types is not None:
        if not isinstance(types, Iterable): types = (types,)
        named_modules = filter(lambda p: type(p[1]) in types, named_modules)
        return named_modules
    return []

# dump collected mse's into a excel file
def dump_mse(model, types=QuantizedModule, path=None):
    if config['collect_mse']['collect']:
        if not path:
            path =  os.path.join(config['dump_results']['out_path'],'mse_results.xlsx')
        named_modules = get_named_modules(model,types=types)
        tmp_list = []
        for module in named_modules:
            
            mse = [module[0]] + [x.item() for x in module[1].mse]
            tmp_list.append(mse)

        df = pd.DataFrame(tmp_list)
        if os.path.exists(path):
            os.remove(path)

        df.to_excel(path)
