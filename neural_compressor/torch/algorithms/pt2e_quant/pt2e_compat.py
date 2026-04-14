"""Compatibility imports for PT2E quantization APIs."""

from importlib import import_module

from neural_compressor.torch.utils import TORCH_VERSION_2_11_0, get_torch_version


def _load_pt2e_modules():
    if get_torch_version() >= TORCH_VERSION_2_11_0:
        try:
            pt2e_module = import_module("torchao.quantization.pt2e")
            quantizer_module = import_module("torchao.quantization.pt2e.quantizer.x86_inductor_quantizer")
            xnnpack_module = import_module("torchao.quantization.pt2e.quantizer.xnnpack_quantizer")
            ao_quantization_module = None
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "torch>=2.11 requires `torchao` for PT2E quantization. Please install torchao."
            ) from exc
    else:
        pt2e_module = import_module("torch.ao.quantization.quantize_pt2e")
        quantizer_module = import_module("torch.ao.quantization.quantizer.x86_inductor_quantizer")
        xnnpack_module = import_module("torch.ao.quantization.quantizer.xnnpack_quantizer")
        ao_quantization_module = import_module("torch.ao.quantization")
    return pt2e_module, quantizer_module, xnnpack_module, ao_quantization_module


_PT2E_MODULE, xiq, xpq, _AO_QUANTIZATION_MODULE = _load_pt2e_modules()

prepare_pt2e = _PT2E_MODULE.prepare_pt2e
convert_pt2e = _PT2E_MODULE.convert_pt2e
move_exported_model_to_eval = (
    _PT2E_MODULE.move_exported_model_to_eval
    if get_torch_version() >= TORCH_VERSION_2_11_0
    else _AO_QUANTIZATION_MODULE.move_exported_model_to_eval
)
X86InductorQuantizer = xiq.X86InductorQuantizer
QuantizationConfig = xiq.QuantizationConfig

__all__ = [
    "prepare_pt2e",
    "convert_pt2e",
    "move_exported_model_to_eval",
    "xiq",
    "xpq",
    "X86InductorQuantizer",
    "QuantizationConfig",
]
