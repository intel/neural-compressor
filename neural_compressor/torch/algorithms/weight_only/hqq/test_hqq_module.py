from hqq_utils import HQQTensorHandle, HQQModuleConfig, QuantTensorConfig, HQQLinear
import torch
from hqq.core.common.utils import is_divisible, compare_two_tensor
from copy import deepcopy


######################
#### Test
#####################


def hqq_base_quant_config(
    nbits=4,
    group_size=64,
    quant_zero=True,
    quant_scale=False,
    scale_quant_group_size=128,
):
    assert (
        nbits in HQQTensorHandle.SUPPORTED_BITS
    ), "nbits value not supported. Check Quantizer.SUPPORTED_BITS."
    if group_size is not None:
        assert is_divisible(
            group_size, 8
        ), "Invalid group_size param: the value should be a multiple of 8."
    weight_quant_params = {
        "nbits": nbits,
        "channel_wise": True,
        "group_size": group_size,
        "optimize": True,
        "round_zero": True if nbits == 4 else False,
    }
    scale_quant_params = (
        {
            "nbits": 8,
            "channel_wise": True,
            "group_size": scale_quant_group_size,
            "optimize": False,
        }
        if (quant_scale)
        else None
    )
    zero_quant_params = (
        {"nbits": 8, "channel_wise": False, "group_size": None, "optimize": False}
        if (quant_zero)
        else None
    )
    return {
        "weight_quant_params": weight_quant_params,
        "scale_quant_params": scale_quant_params,
        "zero_quant_params": zero_quant_params,
    }


# Alias: follow similar Auto-GPTQ naming
BaseQuantizeConfig = hqq_base_quant_config


def create_hqq_quant_config_from_hqq_official_api(
    nbits=4,
    group_size=64,
    quant_zero=True,
    quant_scale=False,
    scale_quant_group_size=128,
):
    hqq_offical_config = hqq_base_quant_config(
        nbits=nbits,
        group_size=group_size,
        quant_zero=quant_zero,
        quant_scale=quant_scale,
        scale_quant_group_size=scale_quant_group_size,
    )
    hqq_quant_config = HQQModuleConfig(
        weight_quant_config=QuantTensorConfig(**hqq_offical_config["weight_quant_params"]),
        scale_quant_config=QuantTensorConfig(**hqq_offical_config["scale_quant_params"])
        if hqq_offical_config["scale_quant_params"] is not None
        else None,
        zero_quant_config=QuantTensorConfig(**hqq_offical_config["zero_quant_params"])
        if hqq_offical_config["zero_quant_params"] is not None
        else None,
    )
    print(
        f"[create_hqq_quant_config_from_hqq_official_api] hqq_quant_config: {hqq_quant_config}"
    )
    print(
        f"[create_hqq_quant_config_from_hqq_official_api] hqq_offical_config: {hqq_offical_config}"
    )
    return hqq_quant_config, hqq_offical_config




hqq_quant_config, hqq_offical_config = create_hqq_quant_config_from_hqq_official_api()
in_features = 64
out_features = 128
float_linear = torch.nn.Linear(in_features, out_features)
float_linear_copy = deepcopy(float_linear)

hqq_linear = HQQLinear.from_float(float_linear, quant_config=hqq_quant_config)

from hqq.core.common.modules import HQQLinear as HQQLinear_official

hqq_linear_official = HQQLinear_official(
    float_linear_copy, quant_config=hqq_offical_config
)


compare_two_tensor(
    hqq_linear.q_weight.val, hqq_linear_official.W_q, msg="The quantized weight"
)
compare_two_tensor(
    hqq_linear.q_weight.scale, hqq_linear_official.meta["scale"], msg="float scale"
)
compare_two_tensor(
    hqq_linear.q_weight.zero.val,
    hqq_linear_official.meta["zero_q"],
    rtol=1,  # TODO: to solve it !!!
    msg="The quantized zero",
)

input = torch.randn(1, in_features)
float_output = float_linear(input)
print(hqq_linear.q_weight)
input_half = deepcopy(input).half()
hqq_output = hqq_linear(input_half)
print(hqq_linear.q_weight)
hqq_output_2 = hqq_linear(input_half)
print(hqq_linear.q_weight)
hqq_offical_output = hqq_linear_official(input_half)

