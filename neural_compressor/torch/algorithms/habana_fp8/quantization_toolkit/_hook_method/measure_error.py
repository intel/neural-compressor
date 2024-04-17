import torch
# from habana_quantization_toolkit._quant_common.quant_config import Fp8cfg, QuantMode
# from habana_quantization_toolkit._hook_method.common import *
# from habana_quantization_toolkit._quant_common.helper_modules import MatMul
from neural_compressor.torch.algorithms.habana_fp8.quantization_toolkit._quant_common.quant_config import Fp8cfg, QuantMode
from neural_compressor.torch.algorithms.habana_fp8.quantization_toolkit._hook_method.common import *
from neural_compressor.torch.algorithms.habana_fp8.quantization_toolkit._quant_common.helper_modules import MatMul
import functools
config = Fp8cfg().cfg
# import habana_quantization_toolkit
from neural_compressor.torch.algorithms.habana_fp8 import quantization_toolkit as habana_quantization_toolkit


def average_error(mod, input_names, base_fname, hp_dtype):
  cnt = 0
  err = torch.zeros(1, device="hpu")
  p = torch.zeros(1, device="hpu")
  while os.path.exists(base_fname+'_'+input_names[0]+'_iter'+str(cnt)+'.pt'):
    inputs = []
    for input_name in input_names:
      inputs.append(torch.load(base_fname+'_'+input_name+'_iter'+str(cnt)+'.pt').to(device="hpu", dtype=hp_dtype))
    output_ref=torch.load(base_fname+'_output_iter'+str(cnt)+'.pt').to(device="hpu", dtype=hp_dtype)
    with torch.no_grad():
      output=mod(*inputs)
      err+=torch.norm(output.to(torch.float32)-output_ref.to(torch.float32))**2
      p+=torch.norm(output_ref.to(torch.float32))**2
    # print(err, p, err/p)
    cnt+=1
  return err/cnt, p/cnt, err/p

def named_modules(name, mod):
  d = {name: mod}
  return zip(d.keys(), d.values())

err_dict = {}

hp_dtype = config['hp_dtype']
lp_dtype = config['fp8_config']
fname_base = config['dump_stats_path']+'_hooks_save'
gmod_list=load_json(fname_base + '_mod_list.json')
folder_name = os.path.join(config['dump_stats_base_path'], 'tensors')
for mod_name in gmod_list:
  file_base_name = os.path.join(folder_name, mod_name + '_module.pt')
  state = torch.load(file_base_name)
  if 'weight' in state:
    mod = torch.nn.Linear(in_features=state['weight'].shape[1], out_features=state['weight'].shape[0], bias='bias' in state)
    input_names=['input0']
  else:
    mod = MatMul()
    input_names = ['input0', 'input1']
  mod.load_state_dict(state)
  mod=mod.to(dtype=hp_dtype, device="hpu").eval()
  mod.named_modules=functools.partial(named_modules, name=mod_name, mod=mod)
  habana_quantization_toolkit.prep_model(mod)  # fp8 additions
  err_dict[mod_name] = average_error(mod, input_names, os.path.join(folder_name, mod_name), hp_dtype)
for mod_name in err_dict:
  print(mod_name, torch.sqrt(err_dict[mod_name][2]).item())
print(err_dict)
