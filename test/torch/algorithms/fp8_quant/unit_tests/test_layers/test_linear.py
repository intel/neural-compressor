import typing

import pytest
import torch

from neural_compressor.torch.algorithms.fp8_quant._core.quant_dequant import QuantDynamicInput
from neural_compressor.torch.algorithms.fp8_quant._core.scale_handler import scale_to_scalar
from neural_compressor.torch.algorithms.fp8_quant._core.scale_methods.scale_method_config import ScaleMethodString
from neural_compressor.torch.algorithms.fp8_quant._quant_common.quant_config import ScaleFormat

from ...test_hpu_utils import *
from ...tester import *


# Test Class to support restoration of calculated scale during runtime with dynamic quantization to test it correctness.
# This is a workaround to avoid saving the scale in the original QuantDynamicInput class as scale saving may cause unwanted graph breaks in torch.compile or issues with hpu_graph.
class TestQuantDynamicInput(QuantDynamicInput):
    def __init__(self, input_scales_creator, lp_dtype, hp_dtype, *args, **kwargs):
        super(TestQuantDynamicInput, self).__init__(input_scales_creator, lp_dtype, hp_dtype, *args, **kwargs)
        self.input_scale = None

    def forward(self, x):
        ret, scale = super().forward(x)
        # We save the calculated scale during this forward pass to test it correctness.
        self.input_scale = scale
        return ret, scale


def get_test_vectors(
    *, dtype: torch.dtype, N: int, D_in: int, atol: float = 0.02, rtol: float = 0.01
) -> typing.Iterable[TestVector]:
    yield TestVector(
        inputs=[torch.ones(N, D_in, dtype=dtype, device="hpu", requires_grad=False)],
        atol=atol,
    )
    yield TestVector(
        inputs=[
            (
                torch.ones(N, D_in, dtype=dtype, device="hpu")
                * torch.tensor(list(range(0, N * D_in)), dtype=dtype)
                / (N * D_in)
            ).requires_grad_(False)
        ],
        atol=atol,
    )
    yield TestVector(
        inputs=[
            (
                torch.ones(N, D_in, dtype=dtype, device="hpu", requires_grad=False)
                * torch.tensor(list(range(0, N * D_in)), dtype=dtype)
            ).requires_grad_(False)
        ],
        rtol=rtol,
    )


def check_tests_to_skip(scale_method, scale_format, dynamic_quantization, device_type=None):
    # TODO [SW-215692]: Fix segfault
    if scale_format == ScaleFormat.CONST or dynamic_quantization:
        if scale_method in [ScaleMethodString.MAXABS_HW_OPT_WEIGHT, ScaleMethodString.MAXABS_POW2_OPT_WEIGHT]:
            pytest.xfail("Segfault")
    # TODO [SW-225900] HW_ALIGNED_SINGLE_SCALE on gaudi3 fails in test_linear unit test
    if scale_method == ScaleMethodString.HW_ALIGNED_SINGLE_SCALE and device_type == GAUDI3:
        pytest.xfail("NoAccuracy")


@pytest.mark.parametrize("hp_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("lp_dtype", [torch.float8_e4m3fn], ids=["fp8_e4m3fn"])
@pytest.mark.parametrize("scale_method", ScaleMethodString)
@pytest.mark.parametrize("device_type", device_type)
@pytest.mark.parametrize("scale_format", ScaleFormat)
@pytest.mark.parametrize("use_hpu_graphs", [True, False], ids=["use_hpu_graphs", "no_hpu_graphs"])
@pytest.mark.parametrize("dynamic_quantization", [True, False], ids=["dynamic_quantization", "static_quantization"])
def test_linear_accuracy(
    hp_dtype: torch.dtype,
    lp_dtype: torch.dtype,
    scale_method: ScaleMethodString,
    device_type: str,
    scale_format: ScaleFormat,
    use_hpu_graphs: bool,
    dynamic_quantization: bool,
):
    check_tests_to_skip(scale_method, scale_format, dynamic_quantization, device_type)
    quant_modes = QUANT_MODES_DEFAULT
    atol = 0.022
    rtol = 0.175
    if scale_method == ScaleMethodString.MAXABS_ARBITRARY:
        atol = 0.078
        rtol = 0.3
    if scale_method in SCALE_METHODS_QUANT_ONLY or dynamic_quantization:
        quant_modes = QUANT_MODES_QUANT_ONLY
        if scale_method == ScaleMethodString.HW_ALIGNED_SINGLE_SCALE:
            atol = 1.0
            rtol = 1.0
        elif scale_method == ScaleMethodString.UNIT_SCALE:
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
            dynamic_quantization=dynamic_quantization,
        )

    if scale_method == ScaleMethodString.MAXABS_HW:
        if device_type_id[device_type] == get_gaudi3_type() and is_gaudi2():
            # Gaudi3 scales not supported on Gaudi2 so "device_for_scales:Gaudi3" is not supported on Gaudi2 run
            return run_with_raised_exception(run, ValueError, "Unsupported config: device_for_scales=")
    else:
        if get_device_type() != device_type_id[device_type]:
            # In scale_method different than MAXABS_HW, we don't support device_for_scales so this scale_method config fails
            return run_with_raised_exception(run, ValueError, "Unsupported config: scale_method")

    if dynamic_quantization:
        if scale_method in HW_ALIGNED_SCALE_METHODS or scale_method in QUANT_ONLY_SCALE_METHODS:
            # When in dynamic quantization we don't support hw aligned scale methods and unit scale
            return run_with_raised_exception(run, ValueError, "Unsupported config: scale_method")
    else:
        if scale_method in SUPPORTED_DYNAMIC_SCALES:
            # When in static quantization we don't support dynamic scale method
            return run_with_raised_exception(run, ValueError, "Unsupported config: scale_method")
    return run()


@pytest.mark.parametrize("hp_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("lp_dtype", [torch.float8_e4m3fn], ids=["fp8_e4m3fn"])
@pytest.mark.parametrize("scale_method", ScaleMethodString)
@pytest.mark.parametrize("device_type", device_type)
@pytest.mark.parametrize("scale_format", ScaleFormat)
@pytest.mark.parametrize("use_hpu_graphs", [True, False], ids=["use_hpu_graphs", "no_hpu_graphs"])
def test_linear_dynamic_quantization(
    hp_dtype: torch.dtype,
    lp_dtype: torch.dtype,
    scale_method: ScaleMethodString,
    device_type: str,
    scale_format: ScaleFormat,
    use_hpu_graphs: bool,
):
    if not use_hpu_graphs and (hp_dtype == torch.bfloat16) and device_type == GAUDI2:
        pytest.xfail("[SW-242200] Temporary skip them since the time usage is more than expected.")
    check_tests_to_skip(scale_method, scale_format, True)
    N = 1
    D_in = 8
    H = 5
    module_class = torch.nn.Linear
    module_kwargs = {
        "in_features": D_in,
        "out_features": H,
        "bias": False,
        "dtype": hp_dtype,
    }

    def run():
        test_vectors = get_test_vectors(dtype=hp_dtype, N=N, D_in=D_in)
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
        test_quant_dynamic_input = TestQuantDynamicInput(
            dynamic_quantized_model.inner.quant_input.input_scales_creator,
            dynamic_quantized_model.inner.quant_input.lp_dtype,
            dynamic_quantized_model.inner.quant_input.hp_dtype,
        )
        dynamic_quantized_model.inner.quant_input = test_quant_dynamic_input

        for vector in test_vectors:
            dynamic_quantized_output = dynamic_quantized_model(*(input.clone() for input in vector.inputs)).to(float)
            # We save the calculated scale after the dynamic_quantized_model run the current input and calculates new scale.
            # In next iteration, we will have a new scale stored in the class.
            current_input_dynamic_scale = dynamic_quantized_model.inner.quant_input.input_scale

            if isinstance(current_input_dynamic_scale, torch.Tensor):
                current_input_dynamic_scale = scale_to_scalar(current_input_dynamic_scale)
            if scale_method not in SCALE_METHODS_QUANT_ONLY:
                assert (
                    previous_input_dynamic_scale != current_input_dynamic_scale
                ), f"input scales in dynamic quantization should differ in different tensors {previous_input_dynamic_scale=} {current_input_dynamic_scale=}"
            previous_input_dynamic_scale = current_input_dynamic_scale

    if device_type_id[device_type] == get_gaudi3_type() and is_gaudi2() and scale_method == ScaleMethodString.MAXABS_HW:
        return run_with_raised_exception(run, ValueError, "Unsupported config: device_for_scales=")
    if (
        (get_device_type() != device_type_id[device_type])
        or scale_method in HW_ALIGNED_SCALE_METHODS
        or scale_method in QUANT_ONLY_SCALE_METHODS
    ):
        return run_with_raised_exception(run, ValueError, "Unsupported config: scale_method")

    return run()
