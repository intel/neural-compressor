# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Intel Neural Compressor Pytorch quantization base API."""

import copy
import warnings
from typing import Any, Callable, Dict, Tuple

import torch

from neural_compressor.common.base_config import BaseConfig, ComposableConfig, config_registry
from neural_compressor.common.utils import FAKE_QUANT, Mode, call_counter, log_process
from neural_compressor.torch.quantization.config import (
    AutoRoundConfig,
    FP8Config,
    HybridGPTQConfig,
    INT8StaticQuantConfig,
    SmoothQuantConfig,
)
from neural_compressor.torch.utils import is_ipex_available, logger
from neural_compressor.torch.utils.utility import WHITE_MODULE_LIST, algos_mapping, get_model_info

FRAMEWORK_NAME = "torch"


class _AutoRoundModelReference:
    """A lightweight container for AutoRound string model references."""

    def __init__(self, model_reference: str, quant_config: BaseConfig, example_inputs: Any = None):
        self.model_reference = model_reference
        self.quant_config = quant_config
        self.example_inputs = example_inputs
        self.is_prepared = True


def _is_autoround_string_model_reference(model: Any, quant_config: BaseConfig) -> bool:
    """Return True when AutoRound is called with a string model reference."""
    return isinstance(quant_config, AutoRoundConfig) and isinstance(model, str)


def need_apply(configs_mapping: Dict[Tuple[str, callable], BaseConfig], algo_name):
    """Check whether to apply this algorithm according to configs_mapping.

    Args:
        configs_mapping (Dict[Tuple[str, callable], BaseConfig]): configs mapping
        algo_name (str): algo name

    Returns:
        Bool: True or False.
    """
    return any(config.name == algo_name for config in configs_mapping.values())


def preprocess_quant_config(model, quant_config, mode="prepare", example_inputs=None, run_fn=None):
    """Preprocess the quantization configuration.

    Args:
        model (torch.nn.Module): Float model to be quantized.
        quant_config (BaseConfig | dict): Quantization configuration or configuration dictionary.
        mode (str, optional): Quantization mode to prepare for. Defaults to "prepare".
        example_inputs (Any, optional): Example inputs for tracing the model. Defaults to None.
        run_fn (Callable | None): Calibration function for collecting statistics. Defaults to None.

    Returns:
        tuple[torch.nn.Module, dict]: Updated model and configuration mapping.
    """
    registered_configs = config_registry.get_cls_configs()
    if isinstance(quant_config, dict):
        quant_config = ComposableConfig.from_dict(quant_config, config_registry=registered_configs[FRAMEWORK_NAME])
        logger.info(f"Parsed a config dict to construct the quantization config: {quant_config}.")
    else:
        assert isinstance(
            quant_config, BaseConfig
        ), f"Please pass a dict or config instance as the quantization configuration, but got {type(quant_config)}."
    logger.debug("Quantize model with config:")
    logger.debug(quant_config.to_dict())

    if is_ipex_available and (
        isinstance(quant_config, INT8StaticQuantConfig) or isinstance(quant_config, SmoothQuantConfig)
    ):
        if mode == "quantize":
            if isinstance(quant_config, SmoothQuantConfig):
                from neural_compressor.torch.algorithms.smooth_quant import TorchSmoothQuant

                sq = TorchSmoothQuant(
                    model, dataloader=None, example_inputs=example_inputs, q_func=run_fn, record_max_info=True
                )
                model.sq_info = sq
                model = sq.transform(
                    alpha=quant_config.alpha,
                    folding=quant_config.folding,
                    auto_alpha_args=quant_config.auto_alpha_args,
                    scale_sharing=quant_config.scale_sharing,
                )
        model_info = quant_config.get_model_info(model, example_inputs)
    elif isinstance(quant_config, AutoRoundConfig):
        if _is_autoround_string_model_reference(model, quant_config):
            # Keep optional large objects on config when model is a string reference.
            model_info = quant_config.get_model_info(model=None)
        else:
            for _attr in ("tokenizer", "processor", "image_processor", "template"):
                _backup = getattr(quant_config, _attr, None)
                if _backup is not None:
                    setattr(model, _attr, _backup)
                    delattr(quant_config, _attr)
            model_info = quant_config.get_model_info(model=model)
    else:
        model_info = quant_config.get_model_info(model=model)

    if (
        (hasattr(quant_config, "model_path") and quant_config.model_path == "")
        or isinstance(quant_config, ComposableConfig)
    ) and hasattr(model, "name_or_path"):
        quant_config.model_path = model.name_or_path
    if hasattr(model, "dpq_quantized") and model.dpq_quantized and type(quant_config) == FP8Config:
        quant_config = HybridGPTQConfig.convert_from_fp8(quant_config)
    configs_mapping = quant_config.to_config_mapping(model_info=model_info)
    logger.debug(configs_mapping)
    return model, configs_mapping


@log_process(mode=Mode.QUANTIZE)
@call_counter
def quantize(
    model: torch.nn.Module,
    quant_config: BaseConfig,
    run_fn: Callable = None,
    run_args: Any = None,
    inplace: bool = True,
    example_inputs: Any = None,
) -> torch.nn.Module:
    """The main entry to quantize model with static mode.

    Args:
        model: a float model to be quantized.
        quant_config: a quantization configuration.
        run_fn: a calibration function for calibrating the model. Defaults to None.
        run_args: positional arguments for `run_fn`. Defaults to None.
        example_inputs: used to trace torch model.

    Returns:
        The quantized model.
    """
    q_model = model if inplace else copy.deepcopy(model)
    q_model, configs_mapping = preprocess_quant_config(
        q_model, quant_config, mode="quantize", example_inputs=example_inputs, run_fn=run_fn
    )
    for algo_name, algo_func in algos_mapping.items():
        # select quantization algo according to config
        if need_apply(configs_mapping, algo_name):
            logger.info(f"Start to apply {algo_name} on the model.")
            q_model = algo_func(
                q_model,
                configs_mapping,
                run_fn=run_fn,
                run_args=run_args,
                example_inputs=example_inputs,
                mode=Mode.QUANTIZE,
            )
    # Fake-quant keeps float parameters, so it is not a deployment-quantized
    # model; marking it `is_quantized` would make HF Trainer refuse to train it.
    if getattr(quant_config, "name", None) != FAKE_QUANT:
        setattr(q_model, "is_quantized", True)
    else:
        setattr(q_model, "is_fake_quant", True)
    return q_model


@log_process(mode=Mode.PREPARE)
def prepare(
    model: torch.nn.Module,
    quant_config: BaseConfig,
    inplace: bool = True,
    example_inputs: Any = None,
):
    """Prepare the model for calibration.

    Insert observers into the model so that it can monitor the input and output tensors during calibration.

    Args:
        model (torch.nn.Module): origin model
        quant_config (BaseConfig): path to quantization config
        inplace (bool, optional): It will change the given model in-place if True.
        example_inputs (tensor/tuple/dict, optional): used to trace torch model.

    Returns:
        prepared and calibrated module.
    """
    if _is_autoround_string_model_reference(model, quant_config):
        return _AutoRoundModelReference(model_reference=model, quant_config=quant_config, example_inputs=example_inputs)

    prepared_model = model if inplace else copy.deepcopy(model)
    prepared_model, configs_mapping = preprocess_quant_config(
        prepared_model, quant_config, mode="prepare", example_inputs=example_inputs
    )
    for algo_name, algo_func in algos_mapping.items():
        # select quantization algo according to config
        if need_apply(configs_mapping, algo_name):
            logger.info(f"Start to prepare model with {algo_name}.")
            prepared_model = algo_func(
                prepared_model,
                configs_mapping,
                example_inputs=example_inputs,
                mode=Mode.PREPARE,
            )
            setattr(prepared_model, "is_prepared", True)
    setattr(prepared_model, "quant_config", quant_config)
    setattr(prepared_model, "example_inputs", example_inputs)
    return prepared_model


_LEGACY_QAT_FORMATS = {
    "MXFP8": "mxfp8",
    "MXFP4": "mxfp4",
    "MXFP8_E4M3": "mxfp8",
}


def _legacy_qat_config(mapping):
    """Translate the legacy ``{nn.ModuleType: scheme}`` mapping into a config."""
    from neural_compressor.torch.quantization.fake_quant import FakeQuantConfig, QuantRule

    schemes = set()
    for module_type, scheme in mapping.items():
        if module_type is not torch.nn.Linear:
            raise NotImplementedError(
                f"legacy prepare_qat only supported nn.Linear, got {module_type!r}; "
                "pass a FakeQuantConfig instead"
            )
        schemes.add(scheme)
    if len(schemes) != 1:
        raise ValueError(f"legacy prepare_qat needs one scheme for all modules, got {sorted(schemes)}")
    scheme = schemes.pop()
    if scheme not in _LEGACY_QAT_FORMATS:
        raise ValueError(
            f"unsupported legacy QAT scheme {scheme!r}; known: {sorted(_LEGACY_QAT_FORMATS)}. "
            "Pass a FakeQuantConfig for the new API."
        )
    fmt = _LEGACY_QAT_FORMATS[scheme]
    return FakeQuantConfig(rules=[QuantRule(module=r".*", weight=fmt, input=fmt)])


def prepare_qat(
    model: torch.nn.Module,
    quant_config=None,
    mapping=None,
    inplace: bool = True,
):
    """Prepare a model for quantization-aware training.

    The new API takes a :class:`FakeQuantConfig` and is equivalent to
    ``prepare(model, quant_config)``: quantizers are inserted, static ones wait
    for :func:`calibrate` or a calibration ``run_fn``. The model may be in eval
    mode (calibration and QAT do not require training mode).

    The legacy ``mapping={nn.Linear: "MXFP8"}`` form is accepted for one
    release and translated to a config with a deprecation warning.

    Args:
        model: Model to modify (in place when ``inplace=True``).
        quant_config: :class:`FakeQuantConfig` describing the quantizers.
        mapping: Legacy ``{module_type: scheme}`` mapping.
        inplace: Mutate the input model instead of a copy.
    """
    from neural_compressor.torch.quantization.fake_quant import FakeQuantConfig

    if quant_config is None and mapping is None:
        warnings.warn(
            "prepare_qat() without a config keeps the legacy MXFP8 default for one release; "
            "pass a FakeQuantConfig instead",
            DeprecationWarning,
            stacklevel=2,
        )
        mapping = {torch.nn.Linear: "MXFP8"}
    if mapping is not None:
        warnings.warn(
            "prepare_qat(mapping=...) is deprecated; pass a FakeQuantConfig instead",
            DeprecationWarning,
            stacklevel=2,
        )
        if quant_config is not None:
            raise ValueError("pass either quant_config or mapping, not both")
        quant_config = _legacy_qat_config(mapping)
    if not isinstance(quant_config, FakeQuantConfig):
        raise TypeError(f"prepare_qat expects a FakeQuantConfig, got {type(quant_config)!r}")
    return prepare(model, quant_config, inplace=inplace)


@log_process(mode=Mode.CONVERT)
def convert(
    model: torch.nn.Module,
    quant_config: BaseConfig = None,
    inplace: bool = True,
    **kwargs,
):
    """Convert the prepared model to a quantized model.

    Args:
        model (torch.nn.Module): torch model
        quant_config (BaseConfig, optional): path to quantization config, only required when model is not prepared.
        inplace (bool, optional): It will change the given model in-place if True.

    Returns:
        The quantized model.
    """
    if isinstance(model, _AutoRoundModelReference):
        if quant_config is None:
            quant_config = model.quant_config
        else:
            logger.warning("quant_config will be ignored since the model has been prepared.")
        model = model.model_reference

    q_model = model if inplace else copy.deepcopy(model)

    assert (
        getattr(model, "is_prepared", False) or quant_config is not None
    ), "Please pass quant_config to convert function."

    if getattr(model, "is_prepared", False):
        if quant_config is None:
            quant_config = model.quant_config
        else:
            logger.warning("quant_config will be ignored since the model has been prepared.")
    example_inputs = model.example_inputs if getattr(model, "is_prepared", False) else None

    registered_configs = config_registry.get_cls_configs()
    if isinstance(quant_config, dict):
        quant_config = ComposableConfig.from_dict(quant_config, config_registry=registered_configs[FRAMEWORK_NAME])
        logger.info(f"Parsed a config dict to construct the quantization config: {quant_config}.")
    else:
        assert isinstance(
            quant_config, BaseConfig
        ), f"Please pass a dict or config instance as the quantization configuration, but got {type(quant_config)}."
    if hasattr(model, "dpq_quantized") and model.dpq_quantized and type(quant_config) == FP8Config:
        quant_config = HybridGPTQConfig.convert_from_fp8(quant_config)
    logger.debug("Convert model with config:")
    logger.debug(quant_config.to_dict())

    # select quantization algo according to config
    if is_ipex_available and (
        isinstance(quant_config, INT8StaticQuantConfig) or isinstance(quant_config, SmoothQuantConfig)
    ):
        model_info = quant_config.get_model_info(q_model, example_inputs)
    else:
        model_info = quant_config.get_model_info(model=q_model)
    configs_mapping = quant_config.to_config_mapping(model_info=model_info)
    logger.debug(configs_mapping)

    # TODO: Need to consider composableConfig situation
    for algo_name, algo_func in algos_mapping.items():
        if need_apply(configs_mapping, algo_name):
            logger.info(f"Start to convert model with {algo_name}.")
            q_model = algo_func(
                q_model,
                configs_mapping,
                example_inputs=example_inputs,
                mode=Mode.CONVERT,
                **kwargs,
            )
    if hasattr(q_model, "__dict__"):
        if getattr(quant_config, "name", None) != FAKE_QUANT:
            setattr(q_model, "is_quantized", True)
        else:
            setattr(q_model, "is_fake_quant", True)
    return q_model


def finalize_calibration(model):
    """Generate and save calibration info."""
    from neural_compressor.torch.algorithms.fp8_quant import save_calib_result

    save_calib_result(model)
