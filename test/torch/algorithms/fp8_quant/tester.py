from __future__ import annotations

import itertools
import logging
import os
import random
import typing
from dataclasses import dataclass
from typing import Dict

import habana_frameworks.torch as ht
import torch
from pytest import raises as pytest_raises

from neural_compressor.torch.algorithms.fp8_quant._core.patching_common import mod_default_dict
from neural_compressor.torch.algorithms.fp8_quant._core.utils import should_quantize
from neural_compressor.torch.algorithms.fp8_quant._quant_common.quant_config import (
    Fp8cfg,
    QuantMode,
    ScaleFormat,
    ScaleMethodString,
    get_hqt_config,
)
from neural_compressor.torch.quantization import FP8Config, convert, prepare  # user level API

from .test_hpu_utils import get_device_name

SUPPORTED_DYNAMIC_SCALES = [ScaleMethodString.ACT_MAXABS_PCS_POW2_WEIGHT_MAXABS_PTS_POW2_HW]

HW_ALIGNED_SCALE_METHODS = [
    ScaleMethodString.MAXABS_HW,
    ScaleMethodString.MAXABS_HW_OPT_WEIGHT,
    ScaleMethodString.ACT_MAXABS_HW_WEIGHTS_PCS_MAXABS_POW2,
    ScaleMethodString.ACT_MAXABS_HW_WEIGHTS_PCS_OPT_POW2,
]

QUANT_ONLY_SCALE_METHODS = [ScaleMethodString.UNIT_SCALE, ScaleMethodString.HW_ALIGNED_SINGLE_SCALE]

# TODO [SW-196641]: fix the following issues:
SCALE_METHODS_SEGFAULT = [
    ScaleMethodString.ACT_MAXABS_HW_WEIGHTS_PCS_OPT_POW2,
    ScaleMethodString.ACT_MAXABS_POW2_WEIGHTS_PCS_OPT_POW2,
    ScaleMethodString.MAXABS_HW_OPT_WEIGHT,
    ScaleMethodString.MAXABS_POW2_OPT_WEIGHT,
]

SCALE_METHODS_COMPILATION_ERROR = [
    ScaleMethodString.ACT_MAXABS_HW_WEIGHTS_PCS_MAXABS_POW2,
    ScaleMethodString.ACT_MAXABS_POW2_WEIGHTS_PCS_MAXABS_POW2,
]
SCALE_METHODS_QUANT_ONLY = [ScaleMethodString.UNIT_SCALE, ScaleMethodString.HW_ALIGNED_SINGLE_SCALE]

QUANT_MODES_DEFAULT = [QuantMode.MEASURE, QuantMode.QUANTIZE]
QUANT_MODES_QUANT_ONLY = [QuantMode.QUANTIZE]

DTYPE_TO_HPDTYPE_STR = {
    torch.bfloat16: "BF16",
    torch.float16: "FP16",
    torch.float32: "FP32",
}

RUNTIME_SCALE_PATCHING_SUPPORTED_METHODS_LIST = [
    ScaleMethodString.UNIT_SCALE,
    ScaleMethodString.HW_ALIGNED_SINGLE_SCALE,
    ScaleMethodString.MAXABS_HW,
    ScaleMethodString.MAXABS_POW2,
    ScaleMethodString.MAXABS_HW_OPT_WEIGHT,
    ScaleMethodString.MAXABS_POW2_OPT_WEIGHT,
    ScaleMethodString.MAXABS_ARBITRARY,
]


# Expects to get an exception. If there's no exception, the test will fail
def run_with_raised_exception(test_to_run, error, error_str):
    with pytest_raises(Exception) as exc:
        test_to_run()
    assert error_str in str(exc.value)
    assert exc.type == error


@dataclass
class TestVector:
    # Mark to pytest that it is not a tester class
    __test__ = False

    inputs: typing.Sequence[torch.Tensor]
    atol: typing.Optional[float] = None
    rtol: typing.Optional[float] = None


M = typing.TypeVar("M", bound=torch.nn.Module)


def _assert_quantized_correctly(*, reference_model: WrapModel, quantized_model: WrapModel):
    """In quantized mode, assert the reference model is not quantized, and the quantized model is.

    Otherwise, assert that both are not quantized.
    """
    for reference_name in mod_default_dict.keys():
        # Modules that don't support dynamic quantization currently won't be patched
        # preventing the tests from failing
        # TODO [SW-217813]: Remove this when we support dynamic quantization in all modules, and remove supported_dynamic_ops
        config_internal = get_hqt_config(quantized_model)
        if not should_quantize(config_internal, reference_name, ""):
            continue
        quantized_name = mod_default_dict[reference_name].patched_module.__name__

        assert not reference_model.has_name(quantized_name)
        assert not quantized_model.has_name(reference_name), f"{reference_name=} should not be in the quantized model"

        if reference_model.has_name(reference_name):
            assert quantized_model.has_name(quantized_name), f"{quantized_name=} should be in the quantized model"


import habana_frameworks.torch.core as htcore


def setup_quantization(
    quantized_model,
    mode,
    lp_dtype,
    scale_method,
    device_type,
    scale_format,
    dynamic_quantization,
    use_hpu_graphs,
    **module_kwargs,
):
    config = get_API_level_config(
        mode=mode,
        lp_dtype=lp_dtype,
        scale_method=scale_method,
        device_type=device_type,
        scale_format=scale_format,
        dynamic_quantization=dynamic_quantization,
        **module_kwargs,
    )
    if mode == QuantMode.MEASURE:
        prepare(quantized_model, config)
    elif mode == QuantMode.QUANTIZE:
        convert(quantized_model, config)
    else:
        raise (ValueError(), "Unexpected mode value - {}".format(mode))

    if use_hpu_graphs:
        quantized_model = ht.hpu.wrap_in_hpu_graph(quantized_model)

    return quantized_model


def run_accuracy_test(
    *,
    module_class: typing.Type[M],
    module_args: typing.Sequence = (),
    module_kwargs: typing.Mapping = {},
    lp_dtype: torch.dtype,
    scale_method: ScaleMethodString,
    measure_vectors: typing.Optional[typing.Iterable[TestVector]] = None,
    test_vectors: typing.Iterable[TestVector],
    seed: typing.Optional[int] = None,
    quant_modes: typing.Iterable[list] = QUANT_MODES_DEFAULT,
    device_type: str = get_device_name(),
    scale_format: ScaleFormat = ScaleFormat.SCALAR,
    use_hpu_graphs: bool = True,
    dynamic_quantization: bool = False,
):
    """Run both the reference and the quantized versions of this module,
    and compare the outputs on every test vector.

    First the measure vectors are used for measurements.

    This test also makes asserts the quantization actually happened.
    This may be moved to another tests in the future.

    Args:
        module_class: The reference module class to test.
            This should be the direct module to test, e.g. Matmul, Linear, etc.
        module_args: The positional arguments to pass to the module constructor. Default is empty.
        module_kwargs: The keyword arguments to pass to the module constructor. Default is empty.
        lp_dtype: The dtype to quantize to.
        scale_method: The scaling method to use.
        measure_vectors: An iterable of vectors, each contains a sequence of inputs.
            If not given, `itertools.tee()` for `test_vectors` will be used.
            That is, all the test vectors will be used for the measurements.
        test_vectors: An iterable of test vectors, each contains a sequence of inputs and tolerance
        seed: The random seed to use. If not given, will use a default seed derived from the module name.
        quant_modes: An iterable of quantization modes.
        device_type: Override device type
        scale_format: The scale format to use: Const tensor or scalar (default: scalar)
        use_hpu_graphs: Wrap in hpu graph (default: True)
        dynamic_quantization: Use dynamic quantization (default: False)
    """

    # If no measure vectors given - use the same dataset as for the test vectors
    # Use `help(itertools.tee)` for more info
    if measure_vectors is None:
        measure_vectors, test_vectors = itertools.tee(test_vectors)

    for mode in quant_modes:
        import neural_compressor.torch.algorithms.fp8_quant.prepare_quant.prepare_model as prepare_model

        reference_model = WrapModel(module_class, seed, *module_args, **module_kwargs)
        quantized_model = WrapModel(module_class, seed, *module_args, **module_kwargs)

        quantized_model = setup_quantization(
            quantized_model,
            mode,
            lp_dtype,
            scale_method,
            device_type,
            scale_format,
            dynamic_quantization,
            use_hpu_graphs,
            **module_kwargs,
        )

        _assert_quantized_correctly(reference_model=reference_model, quantized_model=quantized_model)

        vectors = {
            QuantMode.MEASURE: measure_vectors,
            QuantMode.QUANTIZE: test_vectors,
        }[mode]

        for vector in vectors:
            reference_output = reference_model(*(input.clone() for input in vector.inputs)).cpu().to(float)
            quantized_output = quantized_model(*(input.clone() for input in vector.inputs)).cpu().to(float)

            # Override tolerance values given by the caller
            tolerance = {
                key: getattr(vector, key) for key in ["atol", "rtol"] if getattr(vector, key, None) is not None
            }

            # Accuracy check against the reference module
            assert torch.allclose(reference_output, quantized_output, **tolerance), (
                f"Test vector fails in accuracy test: "
                f"\n  inputs={vector.inputs}"
                f"\n  {reference_output=}"
                f"\n  {quantized_output=}"
                f"\n  {lp_dtype=}"
                f"\n  {scale_method.name=}"
            )

        prepare_model.finish_measurements(quantized_model)


def _set_optional_seed(*, module_class: typing.Type[M], seed: typing.Optional[int]):
    """Set random seed to a unique reproducible value derived from the module.

    Args:
        module_class: The module class to test.
            This should be the direct module to test, e.g. Matmul, Linear, etc.
        seed: The random seed to use. If not given, will use a default seed derived from the module name.
    """
    if seed is None:
        import hashlib

        # We use sha256 to ensure a deterministic has, as opposed to `builtins.hash`, which sadly is not so.
        seed = int.from_bytes(
            bytes=hashlib.sha256(module_class.__name__.encode("utf-8")).digest()[:4],
            byteorder="big",
        )

    logging.info(f"Using {seed=}")

    random.seed(seed)
    torch.manual_seed(seed)


class WrapModel(torch.nn.Module):
    """Wrap an inner module.
    If we do not wrap the inner module, it will not be quantized properly.

    Maybe we can change this behavior in the future.
    """

    def __init__(
        self,
        module_class: typing.Type[M],
        seed: typing.Optional[int],
        /,
        *args,
        **kwargs,
    ):
        super().__init__()
        _set_optional_seed(module_class=module_class, seed=seed)
        self.inner = module_class(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs)

    def has_name(self, module_name: str) -> bool:
        return any(module._get_name() == module_name for module in self.modules())


dir_path = os.path.dirname(os.path.realpath(__file__))
TEST_ONLY_OUTPUT_DIRECTORY = f"{dir_path}/test/3x/torch/algorithms/fp8_quant/output/"


def get_test_unique_dump_path(scale_method: ScaleMethodString):
    # This is a unique id of the test including the parameters, thanks to pytest.
    # TODO: make sure this globally-ever unique (probably add global init timestamp)
    unique_test_id = os.environ.get("PYTEST_CURRENT_TEST")
    return os.path.join(TEST_ONLY_OUTPUT_DIRECTORY, unique_test_id)


def _get_test_only_config(
    *,
    mode: QuantMode,
    scale_method: ScaleMethodString,
    lp_dtype: torch.dtype,
    device_type: str = get_device_name(),
    scale_format: ScaleFormat = ScaleFormat.SCALAR,
    dynamic_quantization: bool = False,
    **kwargs,
) -> Dict:
    """Should NOT be used externally.

    Return a new config used only for the tests.
    """

    # TODO: replace this with a version that does not use strings but direct values.
    #  It is currently needed because of how Fp8cfg.parse() works.
    fp8_cfg = {
        "method": "HOOKS",
        "mode": mode.name,
        "observer": "maxabs",
        "fp8_config": str(lp_dtype).replace("torch.float8_", "")[:4],
        "scale_method": scale_method.name,
        "dump_stats_path": get_test_unique_dump_path(scale_method),
        "device_for_scales": device_type,
        "scale_format": scale_format.name,
        "dynamic_quantization": str(dynamic_quantization),
    }
    if "dtype" in kwargs:
        fp8_cfg["hp_dtype"] = DTYPE_TO_HPDTYPE_STR[kwargs["dtype"]]

    return fp8_cfg


def get_internal_config(
    *,
    mode: QuantMode,
    scale_method: ScaleMethodString,
    lp_dtype: torch.dtype,
    device_type: str = get_device_name(),
    **kwargs,
) -> Fp8cfg:
    return Fp8cfg.parse(
        _get_test_only_config(
            mode=mode, scale_method=scale_method, lp_dtype=lp_dtype, device_type=device_type, **kwargs
        )
    )


def get_API_level_config(
    *,
    mode: QuantMode,
    scale_method: ScaleMethodString,
    lp_dtype: torch.dtype,
    device_type: str = get_device_name(),
    scale_format: ScaleFormat = ScaleFormat.SCALAR,
    dynamic_quantization: bool = False,
    **kwargs,
) -> FP8Config:
    return FP8Config.from_dict(
        _get_test_only_config(
            mode=mode,
            scale_method=scale_method,
            lp_dtype=lp_dtype,
            device_type=device_type,
            scale_format=scale_format,
            dynamic_quantization=dynamic_quantization,
            **kwargs,
        )
    )
