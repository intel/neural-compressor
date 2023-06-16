
from ...utils import logger
from ...utils.utility import LazyImport

tqdm = LazyImport("tqdm")
torch = LazyImport("torch")

def quant_weight_asym(weight, num_bits=4):
    maxq = torch.tensor(2 ** num_bits - 1)
    zeros = torch.zeros(weight.shape[0], device=weight.device)
    wmin = torch.minimum(weight.min(1)[0], zeros)
    wmax = torch.maximum(weight.max(1)[0], zeros)
    tmp = (wmin == 0) & (wmax == 0)
    wmin[tmp] = -1
    wmax[tmp] = +1
    scale = (wmax - wmin) / maxq
    zp = torch.round(-wmin / scale)
    scale.unsqueeze_(dim=-1)
    zp.unsqueeze_(dim=-1)
    q = torch.clamp(torch.round(weight / scale) + zp, 0, maxq)
    return scale * (q - zp)


def quant_weight_sym(weight, num_bits=4):
    assert num_bits > 1, "symmetric schema only supports num_bits > 1"
    maxq = torch.tensor(2 ** (num_bits - 1) - 1)
    minq = torch.tensor(2 ** (num_bits - 1))
    wmax = torch.abs(weight).max(1)[0]
    tmp = (wmax == 0)
    wmax[tmp] = +1
    scale = wmax / maxq
    scale.unsqueeze_(dim=-1)
    q = torch.clamp(torch.round(weight / scale), minq, maxq)
    return scale * q


def quant_weight_actor(weight, num_bits, schema):
    if schema == "sym":
        return quant_weight_sym(weight, num_bits)
    else:
        return quant_weight_asym(weight, num_bits)


def quant_weight(weight, num_bits=4, group_size=-1, schema="asym"):
    if group_size == -1 or weight.shape[1] < group_size:
        return quant_weight_actor(weight, num_bits, schema=schema)

    orig_shape = weight.shape
    if weight.shape[1] % group_size == 0:
        weight = weight.view(-1, group_size)
        weight = quant_weight_actor(weight, num_bits, schema=schema)
        weight = weight.view(orig_shape)
        return weight
    else:
        split_index = weight.shape[1] // group_size * group_size
        weight1 = weight[:, :split_index]
        weight1 = weight1.view(-1, group_size)
        weight1 = quant_weight_actor(weight1, num_bits, schema=schema)
        weight1 = weight1.view(orig_shape[0], split_index)
        weight2 = weight[:, split_index:]
        weight2 = quant_weight_actor(weight2, num_bits, schema=schema)
        weight = torch.cat([weight1, weight2], dim=1)
        return weight


def quant_model_weight_only(model, num_bits, group_size=-1, schema="asym", w_layers_config={}):
    assert isinstance(model, torch.nn.Module), "only support torch module"
    supported_layers = ['Linear', 'Conv2d']
    if w_layers_config == {}:
        logger.info(f"quant model with weight oly config->num_bits:{num_bits}, group_size:{group_size}, {schema}")
    for n, m in model.named_modules():
        if m.__class__.__name__ not in supported_layers:
            continue
        if n in w_layers_config:
            num_bits = w_layers_config[n][0]
            group_size = w_layers_config[n][1]
            schema = w_layers_config[n][2]
        if num_bits <= 0:
            logger.info(f"skip {n}")
            continue
        if w_layers_config != {}:
            logger.info(f"{n} num_bits:{num_bits}, group_size:{group_size}, {schema}")
        if m.__class__.__name__ == "Conv2d":
            weight = m.weight
            orig_shape = weight.shape
            weight = weight.permute(1, 0, 2, 3)
            weight = weight.reshape(weight.shape[0], -1)
        else:
            weight = m.weight
        q_weight = quant_weight(weight, num_bits, group_size, schema)
        if m.__class__.__name__ == "Conv2d":
            q_weight = q_weight.reshape(orig_shape[1], orig_shape[0], orig_shape[2], orig_shape[3])
            q_weight = q_weight.permute(1, 0, 2, 3)
        m.weight.data.copy_(q_weight)
    return model

#
# class QuantizerW:
#     def __init__(self, config: QuantizationConfig):
#         self.default_dtype_value = "int8"
#         self.supported_layers = ['Linear', "Conv2d"]
#         self.default_group_value = {
#             "int4": 128,  ##TODO better default value
#             "others": -1
#         }
#         self.default_schema = "asym"
#
#         if "torch" in config.framework:
#             self.quant_actor = TorchQuantizerW
#         else:
#             assert False, "only support torch backend now"
#
#         w_dtype = config.kwargs.get("w_dtype", self.default_dtype_value)
#         w_group_size = self.default_group_value.get(w_dtype, self.default_group_value['others'])
#         w_group_size = config.kwargs.get("w_group_size", w_group_size)
#         w_schema = config.kwargs.get("w_schema", self.default_schema)
#         self.num_bits, self.w_group_size, self.w_schema = self.convert_config(config.kwargs, w_dtype,
#                                                                               w_group_size, w_schema)
#         self.w_layers_config = config.kwargs.get("w_layers_config", {})
#         for key, value in self.w_layers_config.items():
#             num_bits_tmp, w_group_value_tmp, w_schema = self.convert_config(value, w_dtype, w_group_size, w_schema)
#             self.w_layers_config[key] = [num_bits_tmp, w_group_value_tmp, w_schema]
#
#     def convert_config(self, config, default_w_dtype, default_w_group_size, w_schema):
#         w_dtype = config.get("w_dtype", default_w_dtype)
#         if w_dtype.upper() == "FP32" or w_dtype.upper() == "FP16" or w_dtype.upper() == "BF16":
#             num_bits = -1
#             w_group_size = -1
#             return num_bits, w_group_size, w_schema
#         if "int" not in w_dtype:
#             assert False, "weight_only quantization only support int/FP32/FP16/BF16 dtype currently"
#         try:
#             num_bits = int(w_dtype[-1])
#         except:
#             assert False, "please provide correct w_dtype,e.g. int8, int4 and etc."
#         w_group_size = int(config.get("w_group_size", default_w_group_size))
#         w_schema = config.get("w_schema", w_schema)
#         return num_bits, w_group_size, w_schema
#
#     def fit(self, model):
#         return self.quant_actor.fit(model, self.num_bits, self.w_group_size, self.w_schema, self.w_layers_config,
#                                     self.supported_layers)