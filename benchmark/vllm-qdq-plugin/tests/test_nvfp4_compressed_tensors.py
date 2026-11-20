from types import SimpleNamespace
from unittest import mock

import torch
from vllm.model_executor.kernels.linear.nvfp4.marlin import MarlinNvFp4LinearKernel
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4 import (
    CompressedTensorsW4A4Fp4,
)

from vllm_qdq_plugin.qdq.nvfp4 import _nvfp4_qdq_reference
from vllm_qdq_plugin.quantization.inc_nvfp4_moe import INCNvfp4QDQMoEMethod
from vllm_qdq_plugin.quantization.patch import apply_patches


class TestCompressedTensorsNvfp4Qdq:
    @classmethod
    def setup_class(cls) -> None:
        apply_patches(enable_nvfp4_qdq=True)

    def test_dense_qdq_uses_checkpoint_scale_on_marlin_fallback(self) -> None:
        kernel = object.__new__(MarlinNvFp4LinearKernel)
        kernel.apply_weights = mock.Mock(side_effect=lambda layer, x, bias=None: x)
        scheme = object.__new__(CompressedTensorsW4A4Fp4)
        scheme.kernel = kernel
        checkpoint_scale = torch.tensor(448.0 * 6.0 / 3.1, dtype=torch.float32)
        layer = SimpleNamespace(input_global_scale=1.0 / checkpoint_scale, input_global_scale_inv=checkpoint_scale)
        x = torch.linspace(-3.1, 3.1, 64, dtype=torch.bfloat16).reshape(2, 32)

        with mock.patch.object(CompressedTensorsConfig, "_vllm_qdq_nvfp4_enabled", True):
            output = scheme.apply_weights(layer, x)

        kernel.apply_weights.assert_called_once()
        assert torch.equal(output, _nvfp4_qdq_reference(x, checkpoint_scale))
        assert not torch.equal(output, x)

    def test_dense_qdq_skips_non_marlin_kernel(self) -> None:
        kernel = SimpleNamespace(apply_weights=mock.Mock(side_effect=lambda layer, x, bias=None: x))
        scheme = object.__new__(CompressedTensorsW4A4Fp4)
        scheme.kernel = kernel
        layer = SimpleNamespace(input_global_scale=torch.tensor([1.0], dtype=torch.float32))
        x = torch.randn((2, 32), dtype=torch.bfloat16)

        with mock.patch.object(CompressedTensorsConfig, "_vllm_qdq_nvfp4_enabled", True):
            output = scheme.apply_weights(layer, x)

        kernel.apply_weights.assert_called_once()
        assert output is x

    def test_compressed_tensors_moe_uses_qdq_method(self) -> None:
        quant_args = SimpleNamespace(group_size=16)
        scheme_dict = {"weights": quant_args, "input_activations": quant_args}
        quant_config = SimpleNamespace(
            _add_fused_moe_to_target_scheme_map=mock.Mock(),
            get_scheme_dict=mock.Mock(return_value=scheme_dict),
            _is_nvfp4_format=mock.Mock(return_value=True),
        )
        layer = SimpleNamespace(moe_config=object())

        with (
            mock.patch.object(CompressedTensorsConfig, "_vllm_qdq_nvfp4_enabled", True),
            mock.patch(
                "vllm_qdq_plugin.quantization.inc_nvfp4_moe.INCNvfp4QDQMoEMethod"
            ) as qdq_method,
        ):
            CompressedTensorsMoEMethod.get_moe_method(quant_config, layer, "model.layers.3.mlp.experts")

        qdq_method.assert_called_once_with(layer.moe_config, 16)

    def test_moe_uses_stored_input_scales_directly(self) -> None:
        method = object.__new__(INCNvfp4QDQMoEMethod)
        method.moe = SimpleNamespace(is_act_and_mul=True)
        method.group_size = 16
        packed_w13 = torch.zeros((2, 4, 16), dtype=torch.uint8)
        packed_w2 = torch.zeros((2, 2, 16), dtype=torch.uint8)
        layer = SimpleNamespace(
            w13_weight_packed=torch.nn.Parameter(packed_w13, requires_grad=False),
            w2_weight_packed=torch.nn.Parameter(packed_w2, requires_grad=False),
            w13_weight_scale=torch.nn.Parameter(torch.ones((2, 4, 2), dtype=torch.float8_e4m3fn)),
            w2_weight_scale=torch.nn.Parameter(torch.ones((2, 2, 2), dtype=torch.float8_e4m3fn)),
            w13_weight_global_scale=torch.nn.Parameter(torch.ones((2, 2))),
            w2_weight_global_scale=torch.nn.Parameter(torch.ones(2)),
            w13_input_global_scale=torch.nn.Parameter(torch.full((2, 2), 4.0)),
            w2_input_global_scale=torch.nn.Parameter(torch.full((2,), 8.0)),
        )
        prepared = (
            packed_w13,
            layer.w13_weight_scale,
            torch.ones((2, 2)),
            packed_w2,
            layer.w2_weight_scale,
            torch.ones(2),
        )

        with mock.patch(
            "vllm_qdq_plugin.quantization.inc_nvfp4_moe.prepare_nvfp4_moe_layer_for_marlin",
            return_value=prepared,
        ) as prepare:
            method.process_weights_after_loading(layer)

        prepare.assert_called_once()
        assert torch.equal(layer.w13_input_scale, torch.tensor(4.0))
        assert torch.equal(layer.w2_input_scale, torch.tensor(8.0))