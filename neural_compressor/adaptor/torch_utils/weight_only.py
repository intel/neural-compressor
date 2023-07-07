from ...utils import logger
from ...utils.utility import LazyImport

tqdm = LazyImport("tqdm")
torch = LazyImport("torch")


def qdq_weight_asym(weight, num_bits=4):
    """quant and dequant tensor with asym schema
    :param weight:  input weight
    :param num_bits:  num_bits
    :return: qdq weight
    """
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


def qdq_weight_sym(weight, num_bits=4):
    """quant and dequant tensor with sym schema
    :param weight:  input weight
    :param num_bits:  num_bits
    :return: qdq weight
    """
    # assert num_bits > 1, "symmetric scheme only supports num_bits > 1"
    maxq = torch.tensor(2 ** (num_bits - 1) - 1).to(weight.device)
    minq = torch.tensor(-2 ** (num_bits - 1)).to(weight.device)
    if num_bits == 1:
        maxq = torch.tensor(2 ** (num_bits - 1))
        minq = torch.tensor(2 ** (num_bits - 1) - 1)

    wmax = torch.abs(weight).max(1)[0]
    tmp = (wmax == 0)
    wmax[tmp] = +1
    scale = wmax / ((maxq - minq) / 2)
    scale.unsqueeze_(dim=-1)
    q = torch.clamp(torch.round(weight / scale), minq, maxq)
    return scale * q


def qdq_weight_actor(weight, num_bits, scheme):
    """quant and dequant tensor per channel
    :param weight: input weight
    :param num_bits: num_bits
    :param scheme: sym or asym
    :return: qdq weight
    """
    assert num_bits > 0, "num_bits should be larger than 0"
    if scheme == "sym":
        return qdq_weight_sym(weight, num_bits)
    else:
        return qdq_weight_asym(weight, num_bits)

def quant_weight(weight, num_bits=4, group_size=-1, scheme="asym"):
    """quant and dequant tensor with group size
    :param weight: input weight
    :param num_bits: num_bits
    :param group_size: how many elements share one scale/zp
    :param scheme:  sym or asym
    :return: qdq weight
    """
    if group_size == -1 or weight.shape[1] < group_size:
        return qdq_weight_actor(weight, num_bits, scheme=scheme)

    orig_shape = weight.shape
    if weight.shape[1] % group_size == 0:
        weight = weight.reshape(-1, group_size)
        weight = qdq_weight_actor(weight, num_bits, scheme=scheme)
        weight = weight.reshape(orig_shape)
        return weight
    else:
        split_index = weight.shape[1] // group_size * group_size
        weight1 = weight[:, :split_index]
        weight1 = weight1.reshape(-1, group_size)
        weight1 = qdq_weight_actor(weight1, num_bits, scheme=scheme)
        weight1 = weight1.reshape(orig_shape[0], split_index)
        weight2 = weight[:, split_index:]
        weight2 = qdq_weight_actor(weight2, num_bits, scheme=scheme)
        weight = torch.cat([weight1, weight2], dim=1)
        return weight


def rtn_quantize(model, num_bits, group_size=-1, scheme="asym", w_layers_config={}):
    """ quant the model with round to nearst method
    :param model: torch module
    :param num_bits:  num bits
    :param group_size: how many elements share one scale/zp
    :param scheme: sym or asym
    :param w_layers_config:  specific layer wise configirations {"layer_name":[num_bits,group_size,schema]}
    :return:
    """
    assert isinstance(model, torch.nn.Module), "only support torch module"
    assert num_bits > 0, "bit for weight only should large than zero!"
    ##supported_layers = ['Linear', 'Conv2d']
    supported_layers = ['Linear']
    for n, m in model.named_modules():
        if m.__class__.__name__ not in supported_layers:
            continue
        if n in w_layers_config:  # pragma: no cover
            num_bits = w_layers_config[n][0]
            group_size = w_layers_config[n][1]
            scheme = w_layers_config[n][2]
        logger.debug(f"RTN quantized module:{n, m}")
        logger.debug(f"RTN quantization config: num_bits={num_bits}, group_size={group_size}, scheme={scheme}")
        if num_bits <= 0:
            logger.info(f"skip {n}")
            continue
        if m.__class__.__name__ == "Conv2d":
            weight = m.weight
            orig_shape = weight.shape
            weight = weight.permute(1, 0, 2, 3)
            weight = weight.reshape(weight.shape[0], -1)
        else:
            weight = m.weight
        q_weight = quant_weight(weight, num_bits, group_size, scheme)
        if m.__class__.__name__ == "Conv2d":
            q_weight = q_weight.reshape(orig_shape[1], orig_shape[0], orig_shape[2], orig_shape[3])
            q_weight = q_weight.permute(1, 0, 2, 3)
        m.weight.data.copy_(q_weight)
    return model
