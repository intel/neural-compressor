from unittest import mock
from pathlib import Path

import tomllib

import pytest
import torch
from nvfp4_hw.patch import register as register_nvfp4_hardware
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm_qdq_plugin import register
from vllm_qdq_plugin.quantization.inc_nvfp4_e5m3_linear import INCNvfp4UE5M3LinearMethod
from vllm_qdq_plugin.quantization.inc_nvfp4_e5m3_moe import INCNvfp4UE5M3MoEMethod


def test_vllm_uses_two_general_plugin_entry_points():
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())

    assert pyproject["project"]["entry-points"]["vllm.general_plugins"] == {
        "qdq": "vllm_qdq_plugin:register",
        "inc_nvfp4": "nvfp4_hw.patch:register",
    }


@pytest.mark.parametrize("enabled", [False, True])
def test_register_applies_all_qdq_patches(enabled: bool):
    with (
        mock.patch("vllm_qdq_plugin.envs.VLLM_QDQ", enabled),
        mock.patch("vllm_qdq_plugin.patch.apply_patches") as apply_gemm_patches,
        mock.patch("vllm_qdq_plugin.quantization.patch.apply_patches") as apply_quantization_patches,
    ):
        register()

    assert apply_gemm_patches.call_count == int(enabled)
    apply_quantization_patches.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
def test_hardware_register_provides_non_qdq_quantization(enabled: bool):
    with (
        mock.patch("vllm_qdq_plugin.envs.VLLM_QDQ", enabled),
        mock.patch("nvfp4_hw.patch.apply_patches") as apply_hardware_patches,
        mock.patch("vllm_qdq_plugin.quantization.patch.apply_patches") as apply_quantization_patches,
    ):
        register_nvfp4_hardware()

    expected_calls = int(not enabled)
    assert apply_hardware_patches.call_count == expected_calls
    apply_quantization_patches.assert_not_called()


def test_qdq_apply_patches_includes_quantization():
    from vllm_qdq_plugin import patch

    with (
        mock.patch.object(patch, "_patch_mla_kv_b_proj_dtype"),
        mock.patch.object(patch, "_patch_marlin_gemm", return_value=("_test_qdq_patch", object(), object())),
        mock.patch.object(patch, "_patch_moe_marlin_gemm", return_value=("_test_qdq_patch", object(), object())),
        mock.patch("vllm_qdq_plugin.quantization.patch.apply_patches") as apply_quantization_patches,
    ):
        patch.apply_patches()

    apply_quantization_patches.assert_called_once_with()


def test_hardware_apply_patches_includes_quantization_once():
    from nvfp4_hw import patch
    from vllm.model_executor.layers.quantization.inc import inc as inc_module
    from vllm.model_executor.layers.quantization.inc.config_parser import INCConfigParser
    from vllm.model_executor.layers.quantization.inc.schemes import factory

    config = mock.Mock(SUPPORTED_DTYPES=set(), SUPPORTED_FORMATS=set())
    with (
        mock.patch.object(patch, "_PATCHED", False),
        mock.patch("vllm.model_executor.layers.quantization.inc.INCConfig", config),
        mock.patch.object(INCConfigParser, "resolve"),
        mock.patch.object(factory, "resolve_scheme"),
        mock.patch.object(inc_module, "resolve_scheme", create=True),
        mock.patch("vllm_qdq_plugin.quantization.patch.apply_patches") as apply_quantization_patches,
    ):
        patch.apply_patches()
        patch.apply_patches()

    apply_quantization_patches.assert_called_once_with(enable_nvfp4_qdq=False)


def test_quantization_patch_upgrades_hardware_registration_to_qdq():
    from vllm.model_executor.layers.quantization.inc import inc as inc_module
    from vllm.model_executor.layers.quantization.inc.config_parser import INCConfigParser
    from vllm.model_executor.layers.quantization.inc.schemes import factory
    from vllm_qdq_plugin.quantization import patch
    from vllm_qdq_plugin.quantization.inc_nvfp4_scheme import INCNvfp4QDQScheme

    config = mock.Mock(SUPPORTED_DTYPES=set(), SUPPORTED_FORMATS=set())
    with (
        mock.patch.object(patch, "_PATCHED", False),
        mock.patch.object(patch, "_NVFP4_QDQ_ENABLED", False),
        mock.patch("vllm.model_executor.layers.quantization.inc.INCConfig", config),
        mock.patch.object(INCConfigParser, "resolve"),
        mock.patch.object(factory, "resolve_scheme"),
        mock.patch.object(inc_module, "resolve_scheme", create=True),
    ):
        patch.apply_patches(enable_nvfp4_qdq=False)
        patch.apply_patches(enable_nvfp4_qdq=True)

        layer_config = mock.Mock(data_type="nv_fp", bits=4)
        assert isinstance(factory.resolve_scheme(layer_config), INCNvfp4QDQScheme)

    assert config._vllm_qdq_nvfp4_enabled is True


@pytest.mark.parametrize("enabled", [False, True])
def test_dense_method_captures_vllm_qdq(enabled: bool):
    layer_config = type("LayerConfig", (), {"group_size": 16})()

    with mock.patch("vllm_qdq_plugin.envs.VLLM_QDQ", enabled):
        method = INCNvfp4UE5M3LinearMethod(layer_config)

    assert method.enable_qdq is enabled


@pytest.mark.parametrize("enabled", [False, True])
def test_dense_activation_qdq_obeys_vllm_qdq(enabled: bool):
    method = object.__new__(INCNvfp4UE5M3LinearMethod)
    method.group_size = 16
    method.enable_qdq = enabled
    layer = type(
        "Layer",
        (),
        {
            "workspace": object(),
            "weight": object(),
            "weight_scale": object(),
            "weight_global_scale": object(),
            "output_size_per_partition": 4,
            "input_size_per_partition": 4,
        },
    )()
    input_tensor = torch.zeros((2, 4), dtype=torch.bfloat16)
    qdq_output = input_tensor + 1

    with (
        mock.patch(
            "vllm_qdq_plugin.qdq.nvfp4_e5m3.nvfp4_e5m3_qdq",
            return_value=qdq_output,
        ) as qdq,
        mock.patch(
            "vllm_qdq_plugin.quantization.inc_nvfp4_e5m3_linear.apply_fp4_marlin_linear",
            side_effect=lambda **kwargs: kwargs["input"],
        ),
    ):
        actual = method.apply_weights(layer, input_tensor)

    assert qdq.call_count == int(enabled)
    assert torch.equal(actual, qdq_output if enabled else input_tensor)


@pytest.mark.parametrize("enabled", [False, True])
def test_moe_activation_qdq_obeys_vllm_qdq(enabled: bool):
    method = object.__new__(INCNvfp4UE5M3MoEMethod)
    method.group_size = 16
    method.enable_qdq = enabled
    method.moe = type(
        "MoeConfig",
        (),
        {
            "swiglu_limit": None,
            "swiglu_alpha": None,
            "swiglu_beta": None,
            "activation_situ_beta": None,
            "activation_situ_linear_beta": None,
        },
    )()
    layer = type(
        "Layer",
        (),
        {
            "apply_router_weight_on_input": False,
            "w13_weight": object(),
            "w2_weight": object(),
            "w13_weight_scale": object(),
            "w2_weight_scale": object(),
            "global_num_experts": 2,
            "activation": MoEActivation.SILU,
            "expert_map": None,
            "w13_weight_scale_2": object(),
            "w2_weight_scale_2": object(),
            "workspace": object(),
        },
    )()
    input_tensor = torch.zeros((2, 16), dtype=torch.bfloat16)
    captured = {}

    def fake_fused_marlin_moe(**kwargs):
        captured["hidden_states"] = kwargs["hidden_states"]
        activation_output = torch.empty_like(input_tensor)
        kwargs["activation_func"](
            None,
            activation_output,
            torch.zeros_like(input_tensor),
            topk_ids=kwargs["topk_ids"],
            expert_map=None,
        )
        captured["activation_output"] = activation_output
        return activation_output

    with (
        mock.patch(
            "vllm_qdq_plugin.quantization.inc_nvfp4_e5m3_moe.nvfp4_e5m3_qdq",
            side_effect=lambda value, *args, **kwargs: value + 1,
        ) as qdq,
        mock.patch(
            "vllm_qdq_plugin.quantization.inc_nvfp4_e5m3_moe.apply_moe_activation",
            side_effect=lambda _, output, activation_input, **kwargs: output.copy_(activation_input),
        ),
        mock.patch(
            "vllm_qdq_plugin.quantization.inc_nvfp4_e5m3_moe.fused_marlin_moe",
            side_effect=fake_fused_marlin_moe,
        ),
    ):
        method.apply(
            layer,
            input_tensor,
            torch.ones((2, 1)),
            torch.zeros((2, 1), dtype=torch.int32),
            None,
            None,
        )

    expected = input_tensor + 1 if enabled else input_tensor
    assert qdq.call_count == (2 if enabled else 0)
    assert torch.equal(captured["hidden_states"], expected)
    assert torch.equal(captured["activation_output"], expected)
