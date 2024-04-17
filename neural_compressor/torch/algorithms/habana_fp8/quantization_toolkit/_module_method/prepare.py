import os
import glob
import torch
import copy 

from .._quant_common.helper_modules import BMM
from .._quant_common.quant_config import Fp8cfg

from . import QuantizedLinear, QuantizedConv2d, QuantizedBmm

from . import get_named_modules, QuantizedModule
config = Fp8cfg().cfg

if 'LoRACompatibleConv' in config['allowlist']['types']:
    try:
        from diffusers.models.lora import LoRACompatibleConv
        from .modules import QuantizedLoRACompatibleConv
    except:
        print('no LoRACompatibleConv module found')
        
if 'LoRACompatibleLinear' in config['allowlist']['types']:
    try:
        from diffusers.models.lora import LoRACompatibleLinear
        from .modules import QuantizedLoRACompatibleLinear
    except:
        print('no LoRACompatibleLinear module found')



# monkey patch - wrap all modules of type "types" with the measured version
# TODO: with this method, hooks and buffers aren't transferred to the Quantized model, could consider a different approach later
def mkpatch(model, types):
    for name,module in model.named_children():
        if type(module) is  torch.nn.Linear and torch.nn.Linear in types:
            QL = QuantizedLinear(current_module = module)
            setattr(model,name,QL)
        if type(module) is torch.nn.Conv2d and torch.nn.Conv2d in types:
            QC = QuantizedConv2d(current_module = module)
            setattr(model,name,QC)
        if type(module) is BMM and BMM in types:
            QB = QuantizedBmm()
            setattr(model,name,QB)
        if type(module) is ScaledDotProductAttention:
            module.__class__ = QuantizedScaledDotProductAttention
            module.set_qmodule()

        try:
            if type(module) is LoRACompatibleConv:
                module.__class__ = QuantizedLoRACompatibleConv
                module.set_qmodule()
        except:
            pass

        try:
            if type(module) is LoRACompatibleLinear:
                module.__class__ = QuantizedLoRACompatibleLinear
                module.set_qmodule()
        except:
            pass

        mkpatch(module, types)
    
# set name of module to be accesible from within the model, for convenience
def set_names(model):
    named_modules = get_named_modules(model, types=QuantizedModule)
    for name, module in named_modules:
        module.set_name(name)

# dump config used in this experiment, and the model after quantization
def dump_config(model):
    out_path = config['dump_results']['out_path']
    save_path = f'{out_path}config.txt'
    if os.path.exists(save_path):
        num_config = len(glob.glob(f'{out_path}config*'))
        os.rename(save_path,f'{out_path}config_{num_config}.txt')
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    
    with open(f'{out_path}config.txt','w') as data:
        data.write(str(config))
        data.write(str(model)) 

# Used to remove modules from Quantization. Receives a list of modules, iterates over the model
# and sets quant to false.
def remove_modules(model):
    modules = config['blocklist']['names']
    named_modules = get_named_modules(model, types=QuantizedModule)
    for name, module in named_modules:
        if name in modules:
            print('removed ' + name + ' from quantization because of blocklist')
            module.stop_quant()

# run mkpatch and set_names
def init_model(model):
    types = config['allowlist']['types']
    types = [eval(module) for module in types]
    mkpatch(model, types)
    set_names(model)
    remove_modules(model)
    if config['dump_results']['dump']:
        dump_config(model)

