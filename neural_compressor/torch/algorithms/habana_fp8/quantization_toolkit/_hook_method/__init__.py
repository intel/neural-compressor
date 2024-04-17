
import os

import torch.distributed

import torch.nn as nn
from .._quant_common.quant_config import Fp8cfg, QuantMode
config = Fp8cfg()
from .common import mod_default_dict
assert len(config.cfg['mod_dict']) == 0, f"Custom modules are not supported: {config.cfg['mod_dict'].keys()}. Please add it in the code."
def update_mod_dict():
  config.cfg['mod_dict'].update({k: mod_default_dict[k].type for k in mod_default_dict})

from .measure import prepare_model as prepare_model_for_measure
from .quantize import quantize_hooks
from .scale import scaling_params, scale_method_mapping
from .._quant_common.helper_modules import *




def is_substr(substr_list, target):
  return any([x in target for x in substr_list])

def prepare_model(model):
  update_mod_dict()
  allowlist=set(config.cfg['mod_dict'].keys())
  blocklist=set()
  for type_st in config.cfg['blocklist']['types']:
    blocklist.add(type_st)
  allowlist.difference_update(blocklist)
  allowlist_tuple=tuple(allowlist)
  mod_list=[]
  for name, mod in model.named_modules():
    mod_type=mod.__class__.__name__
    if (mod_type in allowlist_tuple) and (is_substr(config.cfg['allowlist']['names'], name) or len(config.cfg['allowlist']['names'])==0) and (not is_substr(config.cfg['blocklist']['names'], name)):
      mod_list.append(name)
    if config.cfg['verbose']:
        print(f"Module list: {mod_list}")
  print(f"Total modules : {len(mod_list)}")
  if (config.cfg['mode']==QuantMode.MEASURE) or (config.cfg['mode']==QuantMode.SHAPE):
    return prepare_model_for_measure(model, mod_list)
  elif config.cfg['mode']==QuantMode.QUANTIZE:
    scaling_method_name = scale_method_mapping[(config.cfg['scale_method'], config.cfg['observer'])]
    scaling_params[scaling_method_name].update(config.cfg['scale_params'])
    config.cfg['scale_params'] = scaling_params[scaling_method_name]
    return quantize_hooks(model, mod_list)

