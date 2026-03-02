import os
import re

import habana_frameworks.torch.core as htcore
import pytest
import torch

from neural_compressor.torch.algorithms.fp8_quant._core.scale_methods.scale_method_config import (
    CfgStr,
    ScaleGranularity,
    ScaleMethodConfig,
    ScaleMethodString,
    ScaleRoundMethod,
    ScaleValueType,
    get_scale_method_from_config,
    load_scale_method_config_by_mod_map,
    scale_method_config_mapping,
)
from neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules import Matmul
from neural_compressor.torch.quantization import FP8Config, convert, prepare

from ..tester import *

torch.manual_seed(1)


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.add_module("layers/1/linear", torch.nn.Linear(1, 1))
        self.add_module("layers/1/matmul", Matmul())
        self.add_module("layers/2/linear", torch.nn.Linear(1, 1))

    def forward(self, x):
        x = self._modules["layers/1/linear"](x)
        x = self._modules["layers/1/matmul"](x, x.clone().T)
        x = x.unsqueeze(0)
        x = self._modules["layers/2/linear"](x)
        return x


fp8_config = {
    "mode": "AUTO",
    "observer": "maxabs",
    "dump_stats_path": "./inc_output/measure",
}


def check_tests_to_skip(scale_method, scale_value_type_weight=None, scale_value_type_activation=None):
    if (
        scale_value_type_weight == ScaleValueType.DUMMY_SCALES
        or scale_value_type_activation == ScaleValueType.DUMMY_SCALES
    ):
        pytest.xfail("Dummy scales is not a scale method")
    if scale_method in SUPPORTED_DYNAMIC_SCALES:
        pytest.xfail("Key error")


@pytest.mark.parametrize("scale_granularity_weight", ScaleGranularity)
@pytest.mark.parametrize("scale_value_type_weight", ScaleValueType)
@pytest.mark.parametrize("scale_round_method_weight", ScaleRoundMethod)
@pytest.mark.parametrize("scale_granularity_activation", ScaleGranularity)
@pytest.mark.parametrize("scale_value_type_activation", ScaleValueType)
@pytest.mark.parametrize("scale_round_method_activation", ScaleRoundMethod)
def test_scale_method_as_dict(
    scale_granularity_weight: ScaleGranularity,
    scale_value_type_weight: ScaleValueType,
    scale_round_method_weight: ScaleRoundMethod,
    scale_granularity_activation: ScaleGranularity,
    scale_value_type_activation: ScaleValueType,
    scale_round_method_activation: ScaleRoundMethod,
):

    weight_scale_method_config = ScaleMethodConfig(
        granularity=scale_granularity_weight,
        scale_value_type=scale_value_type_weight,
        rounding_method=scale_round_method_weight,
    )
    activation_scale_method_config = ScaleMethodConfig(
        granularity=scale_granularity_activation,
        scale_value_type=scale_value_type_activation,
        rounding_method=scale_round_method_activation,
    )
    scale_method_config = {
        CfgStr.WEIGHT: weight_scale_method_config,
        CfgStr.ACTIVATION: activation_scale_method_config,
    }

    scale_method_string = get_scale_method_from_config(scale_method_config)
    check_tests_to_skip(scale_method_string, scale_value_type_weight, scale_value_type_activation)

    def run():
        os.environ["SCALE_METHOD_CONFIG_DUMP_PATH"] = "./inc_output/scale_method_config_map.json"
        model = M().eval().to("hpu").to(torch.bfloat16)
        htcore.hpu_inference_initialize()

        fp8_config["scale_method"] = {
            "default": {
                "weight": {
                    "granularity": scale_granularity_weight.name,
                    "scale_value_type": scale_value_type_weight.name,
                    "rounding_method": scale_round_method_weight.name,
                },
                "activation": {
                    "granularity": scale_granularity_activation.name,
                    "scale_value_type": scale_value_type_activation.name,
                    "rounding_method": scale_round_method_activation.name,
                },
            },
        }
        quant_config = FP8Config.from_dict(fp8_config)

        model = prepare(model, quant_config)
        model(torch.tensor([1]).to("hpu").to(torch.bfloat16))
        model = convert(model, quant_config)

        expected_configs = {
            "layers/1/linear": scale_method_config,
            "layers/1/matmul": scale_method_config,
            "layers/2/linear": scale_method_config,
        }
        scale_method_config_by_mod_map = load_scale_method_config_by_mod_map(
            "./inc_output/scale_method_config_map.json"
        )
        for name, _ in model.named_modules():
            if name in expected_configs:
                expected = expected_configs[name]
                for key in [CfgStr.WEIGHT, CfgStr.ACTIVATION]:
                    assert scale_method_config_by_mod_map[name][key].scale_value_type == expected[key].scale_value_type
                    assert scale_method_config_by_mod_map[name][key].granularity == expected[key].granularity
                    assert scale_method_config_by_mod_map[name][key].rounding_method == expected[key].rounding_method

    if scale_method_string == None:
        return run_with_raised_exception(run, ValueError, "Unsupported config: scale method config for")
    return run()


def run_scale_method_test(fp8_config, expected_configs):
    os.environ["SCALE_METHOD_CONFIG_DUMP_PATH"] = "./inc_output/scale_method_config_map.json"
    model = M().eval().to("hpu").to(torch.bfloat16)
    htcore.hpu_inference_initialize()

    quant_config = FP8Config.from_dict(fp8_config)
    model = prepare(model, quant_config)
    model(torch.tensor([1]).to("hpu").to(torch.bfloat16))
    model = convert(model, quant_config)

    scale_method_config_by_mod_map = load_scale_method_config_by_mod_map("./inc_output/scale_method_config_map.json")
    for name, _ in model.named_modules():
        if name in expected_configs:
            expected = expected_configs[name]
            for key in [CfgStr.WEIGHT, CfgStr.ACTIVATION]:
                assert scale_method_config_by_mod_map[name][key].scale_value_type == expected[key].scale_value_type
                assert scale_method_config_by_mod_map[name][key].granularity == expected[key].granularity
                assert scale_method_config_by_mod_map[name][key].rounding_method == expected[key].rounding_method


def extract_weight_activation_configs(*scale_methods):
    """Given any number of ScaleMethodString enums, returns a list of (weight_cfg, activation_cfg) tuples."""
    result = []
    for sm in scale_methods:
        cfg = scale_method_config_mapping[sm]
        result.append((cfg[CfgStr.WEIGHT], cfg[CfgStr.ACTIVATION]))
    return result


def make_scale_method_config_and_expected(
    override_key: str,
    override_names: list,
    scale_methods: list,
    default_granularity: str = "pts",
    default_scale_value_type: str = "maxabs",
    default_rounding_method: str = "hw_aligned",
):
    """Constructs a scale_method config dict for fp8_config, with a default and an override section,
    and also returns the expected_configs dict for test assertions by mapping config to each module."""
    scale_method = {
        "default": {
            "weight": {
                "granularity": default_granularity,
                "scale_value_type": default_scale_value_type,
                "rounding_method": default_rounding_method,
            },
            "activation": {
                "granularity": default_granularity,
                "scale_value_type": default_scale_value_type,
                "rounding_method": default_rounding_method,
            },
        },
        override_key: {},
    }
    configs = extract_weight_activation_configs(*scale_methods)
    config_map = {}

    # Build the configuration first
    for name, (weight_cfg, activation_cfg), sm in zip(override_names, configs, scale_methods):
        scale_method[override_key][name] = {
            "weight": {
                "granularity": weight_cfg.granularity.name,
                "scale_value_type": weight_cfg.scale_value_type.name,
                "rounding_method": weight_cfg.rounding_method.name,
            },
            "activation": {
                "granularity": activation_cfg.granularity.name,
                "scale_value_type": activation_cfg.scale_value_type.name,
                "rounding_method": activation_cfg.rounding_method.name,
            },
        }
        # Store the scale method for this override
        config_map[name] = scale_method_config_mapping[sm]

    # Now build expected_configs by examining each module in the model
    expected_configs = {}
    model = M()
    for node_name, module in model.named_modules():
        if not node_name:  # Skip the root module
            continue

        # Get layer_type (class name)
        layer_type = module.__class__.__name__

        # Extract layer index if present
        layer_match = re.search(CfgStr.LAYERS_SLASH_PATTERN.value, node_name)
        layer_index = layer_match.group(1) if layer_match else None

        # Determine which scale method would be used for this module
        if override_key == "nodes" and node_name in config_map:
            # Direct node match
            expected_configs[node_name] = config_map[node_name]
        elif override_key == "layers" and layer_index and layer_index in config_map:
            # Layer index match
            expected_configs[node_name] = config_map[layer_index]
        elif override_key == "layer_types" and layer_type in config_map:
            # Layer type match
            expected_configs[node_name] = config_map[layer_type]
        else:
            # Default scale method config
            default_config = scale_method_config_mapping[ScaleMethodString.MAXABS_HW]
            expected_configs[node_name] = default_config
    return scale_method, expected_configs


@pytest.mark.parametrize("scale_method_1", ScaleMethodString)
@pytest.mark.parametrize("scale_method_2", ScaleMethodString)
def test_scale_method_by_node(scale_method_1, scale_method_2):
    check_tests_to_skip(scale_method_1)
    check_tests_to_skip(scale_method_2)
    fp8_config["scale_method"], expected_configs = make_scale_method_config_and_expected(
        "nodes", ["layers/1/linear", "layers/1/matmul"], [scale_method_1, scale_method_2]
    )
    expected_configs["layers/2/linear"] = scale_method_config_mapping[ScaleMethodString.MAXABS_HW]
    run_scale_method_test(fp8_config, expected_configs)


@pytest.mark.parametrize("scale_method_1", ScaleMethodString)
@pytest.mark.parametrize("scale_method_2", ScaleMethodString)
def test_scale_method_by_layer(scale_method_1, scale_method_2):
    check_tests_to_skip(scale_method_1)
    check_tests_to_skip(scale_method_2)
    fp8_config["scale_method"], expected_configs = make_scale_method_config_and_expected(
        "layers", ["1", "2"], [scale_method_1, scale_method_2]
    )
    run_scale_method_test(fp8_config, expected_configs)


@pytest.mark.parametrize("scale_method_1", ScaleMethodString)
@pytest.mark.parametrize("scale_method_2", ScaleMethodString)
def test_scale_method_by_layer_type(scale_method_1, scale_method_2):
    check_tests_to_skip(scale_method_1)
    check_tests_to_skip(scale_method_2)
    fp8_config["scale_method"], expected_configs = make_scale_method_config_and_expected(
        "layer_types", ["Linear", "Matmul"], [scale_method_1, scale_method_2]
    )
    run_scale_method_test(fp8_config, expected_configs)
