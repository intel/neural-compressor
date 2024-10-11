from __future__ import annotations

import itertools
import logging
import os
import random
import typing
from dataclasses import dataclass

import torch

from neural_compressor.torch.algorithms.fp8_quant._core.common import mod_default_dict
from neural_compressor.torch.algorithms.fp8_quant._quant_common.quant_config import Fp8cfg, QuantMode, ScaleMethod

# TODO [SW-196641]: fix the following issues:
SCALE_METHODS_SEGFAULT = [
    ScaleMethod.ACT_MAXABS_HW_WEIGHTS_PCS_OPT_POW2,
    ScaleMethod.ACT_MAXABS_POW2_WEIGHTS_PCS_OPT_POW2,
    ScaleMethod.MAXABS_HW_OPT_WEIGHT,
    ScaleMethod.MAXABS_POW2_OPT_WEIGHT,
]
SCALE_METHODS_KEY_ERROR = [
    ScaleMethod.MAX,
    ScaleMethod.SMOOTHQUANT_WEIGHTS_OUTPUT_CHANNEL_MAXABS_POW2,
    ScaleMethod.WEAKSMOOTHQUANT_WEIGHTS_OUTPUT_CHANNEL_MAXABS_POW2,
    ScaleMethod.SMOOTHQUANT_OPT,
]
SCALE_METHODS_COMPILATION_ERROR = [
    ScaleMethod.ACT_MAXABS_HW_WEIGHTS_PCS_MAXABS_POW2,
    ScaleMethod.ACT_MAXABS_POW2_WEIGHTS_PCS_MAXABS_POW2,
]
SCALE_METHODS_QUANT_ONLY = [ScaleMethod.UNIT_SCALE, ScaleMethod.HW_ALIGNED_SINGLE_SCALE]

QUANT_MODES_DEFAULT = [QuantMode.MEASURE, QuantMode.QUANTIZE]
QUANT_MODES_QUANT_ONLY = [QuantMode.QUANTIZE]


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
        quantized_name = mod_default_dict[reference_name].patched_module.__name__

        assert not reference_model.has_name(quantized_name)
        assert not quantized_model.has_name(reference_name), f"{reference_name=} should not be in the quantized model"

        if reference_model.has_name(reference_name):
            assert quantized_model.has_name(quantized_name), f"{quantized_name=} should be in the quantized model"


def run_accuracy_test(
    *,
    module_class: typing.Type[M],
    module_args: typing.Sequence = (),
    module_kwargs: typing.Mapping = {},
    lp_dtype: torch.dtype,
    scale_method: ScaleMethod,
    measure_vectors: typing.Optional[typing.Iterable[TestVector]] = None,
    test_vectors: typing.Iterable[TestVector],
    seed: typing.Optional[int] = None,
    quant_modes: typing.Iterable[list] = QUANT_MODES_DEFAULT,
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
    """

    # If no measure vectors given - use the same dataset as for the test vectors
    # Use `help(itertools.tee)` for more info
    if measure_vectors is None:
        measure_vectors, test_vectors = itertools.tee(test_vectors)

    for mode in quant_modes:
        import neural_compressor.torch.algorithms.fp8_quant.prepare_quant.prepare_model as prepare_model

        reference_model = WrapModel(module_class, seed, *module_args, **module_kwargs)
        quantized_model = WrapModel(module_class, seed, *module_args, **module_kwargs)

        config = _get_test_only_config(
            mode=mode,
            lp_dtype=lp_dtype,
            scale_method=scale_method,
        )
        prepare_model._prep_model_with_predefined_config(quantized_model, config=config)

        _assert_quantized_correctly(reference_model=reference_model, quantized_model=quantized_model)

        vectors = {
            QuantMode.MEASURE: measure_vectors,
            QuantMode.QUANTIZE: test_vectors,
        }[mode]

        for vector in vectors:
            reference_output = reference_model(*(input.clone() for input in vector.inputs)).to(float)
            quantized_output = quantized_model(*(input.clone() for input in vector.inputs)).to(float)

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


def get_test_unique_dump_path(scale_method: ScaleMethod):
    # This is a unique id of the test including the parameters, thanks to pytest.
    # TODO: make sure this globally-ever unique (probably add global init timestamp)
    unique_test_id = os.environ.get("PYTEST_CURRENT_TEST")
    return os.path.join(TEST_ONLY_OUTPUT_DIRECTORY, unique_test_id)


def _get_test_only_config(
    *,
    mode: QuantMode,
    scale_method: ScaleMethod,
    lp_dtype: torch.dtype,
) -> Fp8cfg:
    """Should NOT be used externally.

    Return a new config used only for the tests.
    """

    # TODO: replace this with a version that does not use strings but direct values.
    #  It is currently needed because of how Fp8cfg.parse() works.
    return Fp8cfg.parse(
        {
            "method": "HOOKS",
            "mode": mode.name,
            "observer": "maxabs",
            "fp8_config": str(lp_dtype).replace("torch.float8_", "")[:4],
            "scale_method": scale_method.name,
            "dump_stats_path": get_test_unique_dump_path(scale_method),
        }
    )
