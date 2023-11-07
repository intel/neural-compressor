##########################
## End-user
##########################
import torch

from neural_compressor.common.utility import print_nested_dict, print_with_note
from neural_compressor.torch.quantization.config import RTNWeightQuantConfig

qconfig = RTNWeightQuantConfig(weight_bits=4, weight_dtype="nf4")
print(qconfig)
qconfig_dict = qconfig.to_dict()
new_config_from_dict = RTNWeightQuantConfig.from_dict(qconfig_dict)
print("new config from dict", new_config_from_dict)
# import pdb; pdb.set_trace()
print_with_note("the fist")
print_nested_dict(qconfig.to_dict())
## For advanced user
global_config = RTNWeightQuantConfig(weight_bits=4, weight_dtype="nf4")
qconfig.set_global(global_config)
conv_config = RTNWeightQuantConfig(weight_bits=6, weight_dtype="nf4")
qconfig.set_operator_type(torch.nn.Linear, conv_config)
conv1_config = RTNWeightQuantConfig(weight_bits=4, weight_dtype="int8")
qconfig.set_operator_name("model.linear1", conv1_config)
# import pdb; pdb.set_trace()
print_with_note("the second")
print_nested_dict(qconfig.to_dict())
qconfig.to_json_file("final_q_config.json")
## quantize model with specific config
from neural_compressor.torch.quantization.quantize import quantize


class UserMolde(torch.nn.Module):
    pass


print_with_note("the end")
print(qconfig)

qconfig_dict = qconfig.to_dict()
new_config_from_dict = RTNWeightQuantConfig.from_dict(qconfig_dict)
print("new config from dict", new_config_from_dict)
q_model = quantize(UserMolde(), qconfig)


print_with_note("get_all_registered_configs")

from neural_compressor.torch.quantization.config import get_all_registered_configs

torch_configs = get_all_registered_configs()
print(torch_configs)


print_with_note("parse from dict")
config = {"rtn_weight_only_quant": qconfig_dict}
q_model = quantize(UserMolde(), config)
