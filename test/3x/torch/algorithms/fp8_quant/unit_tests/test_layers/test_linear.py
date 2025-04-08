import typing

import pytest
import torch

from neural_compressor.torch.algorithms.fp8_quant._quant_common.quant_config import ScaleMethod, ScaleFormat, _hw_aligned_scale_methods
from neural_compressor.torch.algorithms.fp8_quant._core.scale_handler import scale_to_scalar

from ...test_hpu_utils import *
from ...tester import *


def get_test_vectors(*, dtype: torch.dtype, N: int, D_in: int, atol: float = 0.02, rtol: float = 0.01) -> typing.Iterable[TestVector]:
    yield TestVector(
        inputs=[torch.ones(N, D_in, dtype=dtype, device="hpu", requires_grad=False)],
        atol=atol,
    )
    yield TestVector(
        inputs=[(torch.ones(N, D_in, dtype=dtype, device="hpu") * torch.tensor(list(range(0, N*D_in)), dtype=dtype) / (N*D_in)).requires_grad_(False)],
        atol=atol,
    )
    yield TestVector(
        inputs=[(torch.ones(N, D_in, dtype=dtype, device="hpu", requires_grad=False) * torch.tensor(list(range(0, N*D_in)), dtype=dtype)).requires_grad_(False)],
        rtol=rtol,
    )

def check_tests_to_skip(scale_method, scale_format, dynamic_quantization):
    if scale_method in SCALE_METHODS_KEY_ERROR:
        pytest.xfail("KeyError")
    # TODO [SW-215692]: Fix segfault
    if scale_format == ScaleFormat.CONST or dynamic_quantization:
        if scale_method in [ScaleMethod.MAXABS_HW_OPT_WEIGHT, ScaleMethod.MAXABS_POW2_OPT_WEIGHT]:
            pytest.xfail("Segfault")


@pytest.mark.parametrize("hp_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("lp_dtype", [torch.float8_e4m3fn], ids=["fp8_e4m3fn"])
@pytest.mark.parametrize("scale_method", ScaleMethod)
@pytest.mark.parametrize("device_type", device_type)
@pytest.mark.parametrize("scale_format", ScaleFormat)
@pytest.mark.parametrize("use_hpu_graphs", [True, False], ids=["use_hpu_graphs", "no_hpu_graphs"])
@pytest.mark.parametrize("dynamic_quantization", [True, False], ids=["dynamic_quantization", "static_quantization"])
def test_linear_accuracy(
    hp_dtype: torch.dtype,
    lp_dtype: torch.dtype,
    scale_method: ScaleMethod,
    device_type: str,
    scale_format: ScaleFormat,
    use_hpu_graphs: bool,
    dynamic_quantization: bool
):
    check_tests_to_skip(scale_method, scale_format, dynamic_quantization)
    quant_modes = QUANT_MODES_DEFAULT
    atol = 0.022
    rtol = 0.175
    if scale_method == ScaleMethod.MAXABS_ARBITRARY:
        atol = 0.078
    if scale_method in SCALE_METHODS_QUANT_ONLY or dynamic_quantization:
        quant_modes = QUANT_MODES_QUANT_ONLY
        if scale_method == ScaleMethod.HW_ALIGNED_SINGLE_SCALE:
            atol = 1.0
            rtol = 1.0
        elif scale_method == ScaleMethod.UNIT_SCALE:
            atol = 1.0
            rtol = 0.5
    N = 1
    D_in = 8
    H = 5
    def run():
        run_accuracy_test(
            module_class=torch.nn.Linear,
            module_kwargs={
                "in_features": D_in,
                "out_features": H,
                "bias": False,
                "dtype": hp_dtype,
            },
            lp_dtype=lp_dtype,
            scale_method=scale_method,
            test_vectors=get_test_vectors(dtype=hp_dtype, N=N, D_in=D_in, atol=atol, rtol=rtol),
            quant_modes=quant_modes,
            device_type=device_type,
            scale_format=scale_format,
            use_hpu_graphs=use_hpu_graphs,
            dynamic_quantization=dynamic_quantization
        )
    if get_device_type() != device_type_id[device_type] and scale_method != ScaleMethod.MAXABS_HW:
        return run_with_raised_exception(run, ValueError, "Unsupported config: scale_method: ")
    elif device_type_id[device_type] != get_device_type():
        if not (device_type_id[device_type] == get_gaudi2_type() and is_gaudi3()):
            return run_with_raised_exception(run, ValueError, "Unsupported config: device_for_scales=")
    elif scale_method == ScaleMethod.ACT_MAXABS_PCS_POW2_WEIGHT_MAXABS_PTS_POW2_HW and not dynamic_quantization:
            return run_with_raised_exception(run, ValueError, "Unsupported config: scale method ScaleMethod.ACT_MAXABS_PCS_POW2_WEIGHT_MAXABS_PTS_POW2_HW")
    # TODO [SW-222725]: support HW aligned rounding in dynamic quantization
    elif dynamic_quantization and scale_method in _hw_aligned_scale_methods:
        return run_with_raised_exception(run, ValueError, "is not supported in dynamic quantization")
    return run()


#TODO [SW-225078]: Reeanable test, find a way to test scales in dynamic quantization
@pytest.mark.skip("[SW-225078] Find a way to test scales in dynamic quantization")
@pytest.mark.parametrize("hp_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("lp_dtype", [torch.float8_e4m3fn], ids=["fp8_e4m3fn"])
@pytest.mark.parametrize("scale_method", ScaleMethod)
@pytest.mark.parametrize("device_type", device_type)
@pytest.mark.parametrize("scale_format", ScaleFormat)
@pytest.mark.parametrize("use_hpu_graphs", [True, False], ids=["use_hpu_graphs", "no_hpu_graphs"])
def test_linear_dynamic_quantization(
    hp_dtype: torch.dtype,
    lp_dtype: torch.dtype,
    scale_method: ScaleMethod,
    device_type: str,
    scale_format: ScaleFormat,
    use_hpu_graphs: bool
):
    check_tests_to_skip(scale_method, scale_format, True)
    N = 1
    D_in = 8
    H = 5
    module_class=torch.nn.Linear
    module_kwargs={
        "in_features": D_in,
        "out_features": H,
        "bias": False,
        "dtype": hp_dtype,
    }
    def run():
        test_vectors=get_test_vectors(dtype=hp_dtype, N=N, D_in=D_in)
        import neural_compressor.torch.algorithms.fp8_quant.prepare_quant.prepare_model as prepare_model

        dynamic_quantized_model = WrapModel(module_class, None, **module_kwargs)
        dynamic_quantized_model = setup_quantization(
            dynamic_quantized_model,
            QuantMode.QUANTIZE,
            lp_dtype,
            scale_method,
            device_type,
            scale_format,
            True,
            use_hpu_graphs,
            **module_kwargs,
        )
        previous_input_dynamic_scale = 0

        for vector in test_vectors:
            dynamic_quantized_output = dynamic_quantized_model(*(input.clone() for input in vector.inputs)).to(float)

            current_input_dynamic_scale = dynamic_quantized_model.inner.scale_input
            if isinstance(current_input_dynamic_scale, torch.Tensor):
                current_input_dynamic_scale = scale_to_scalar(current_input_dynamic_scale)
            if scale_method not in SCALE_METHODS_QUANT_ONLY:
                assert previous_input_dynamic_scale != current_input_dynamic_scale, f"input scales in dynamic quantization should differ in different tensors {previous_input_dynamic_scale=} {current_input_dynamic_scale=}"
            previous_input_dynamic_scale = current_input_dynamic_scale

        prepare_model.finish_measurements(dynamic_quantized_model)

    if get_device_type() != device_type_id[device_type] and scale_method != ScaleMethod.MAXABS_HW:
        return run_with_raised_exception(run, ValueError, "Unsupported config: scale_method: ")
    elif device_type_id[device_type] != get_device_type():
        if not (device_type_id[device_type] == get_gaudi2_type() and is_gaudi3()):
            return run_with_raised_exception(run, ValueError, "Unsupported config: device_for_scales=")
    # TODO [SW-222725]: support HW aligned rounding in dynamic quantization
    elif scale_method in _hw_aligned_scale_methods:
        return run_with_raised_exception(run, ValueError, "is not supported in dynamic quantization")
    return run()
