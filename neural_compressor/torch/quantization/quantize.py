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
from typing import Any, Callable, Dict, Tuple

import torch

from neural_compressor.common.base_config import BaseConfig, ComposableConfig, config_registry
from neural_compressor.common.utils import Mode, call_counter, log_process
from neural_compressor.torch.quantization.config import INT8StaticQuantConfig, SmoothQuantConfig
from neural_compressor.torch.utils import is_ipex_available, logger
from neural_compressor.torch.utils.utility import WHITE_MODULE_LIST, algos_mapping, get_model_info

FRAMEWORK_NAME = "torch"


def need_apply(configs_mapping: Dict[Tuple[str, callable], BaseConfig], algo_name):
    """Check whether to apply this algorithm according to configs_mapping.

    Args:
        configs_mapping (Dict[Tuple[str, callable], BaseConfig]): configs mapping
        algo_name (str): algo name

    Returns:
        Bool: True or False.
    """
    return any(config.name == algo_name for config in configs_mapping.values())


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
    # select quantization algo according to config

    if is_ipex_available and (
        isinstance(quant_config, INT8StaticQuantConfig) or isinstance(quant_config, SmoothQuantConfig)
    ):
        if isinstance(quant_config, SmoothQuantConfig):
            from neural_compressor.torch.algorithms.smooth_quant import TorchSmoothQuant

            sq = TorchSmoothQuant(
                q_model, dataloader=None, example_inputs=example_inputs, q_func=run_fn, record_max_info=True
            )
            q_model.sq_info = sq
            q_model = sq.transform(
                alpha=quant_config.alpha,
                folding=quant_config.folding,
                auto_alpha_args=quant_config.auto_alpha_args,
                scale_sharing=quant_config.scale_sharing,
            )

        model_info = quant_config.get_model_info(q_model, example_inputs)
    else:
        model_info = quant_config.get_model_info(model=q_model)
    configs_mapping = quant_config.to_config_mapping(model_info=model_info)
    logger.debug(configs_mapping)
    for algo_name, algo_func in algos_mapping.items():
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
    setattr(q_model, "is_quantized", True)
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
    prepared_model = model if inplace else copy.deepcopy(model)
    registered_configs = config_registry.get_cls_configs()
    if isinstance(quant_config, dict):
        quant_config = ComposableConfig.from_dict(quant_config, config_registry=registered_configs[FRAMEWORK_NAME])
        logger.info(f"Parsed a config dict to construct the quantization config: {quant_config}.")
    else:
        assert isinstance(
            quant_config, BaseConfig
        ), f"Please pass a dict or config instance as the quantization configuration, but got {type(quant_config)}."
    logger.debug("Prepare model with config:")
    logger.debug(quant_config.to_dict())

    # select quantization algo according to config
    if is_ipex_available and (
        isinstance(quant_config, INT8StaticQuantConfig) or isinstance(quant_config, SmoothQuantConfig)
    ):
        model_info = quant_config.get_model_info(prepared_model, example_inputs)
    else:
        model_info = quant_config.get_model_info(model=prepared_model)
    configs_mapping = quant_config.to_config_mapping(model_info=model_info)
    logger.debug(configs_mapping)

    # TODO: Need to consider composableConfig situation
    for algo_name, algo_func in algos_mapping.items():
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


@log_process(mode=Mode.CONVERT)
def convert(
    model: torch.nn.Module,
    quant_config: BaseConfig = None,
    inplace: bool = True,
):
    """Convert the prepared model to a quantized model.

    Args:
        model (torch.nn.Module): torch model
        quant_config (BaseConfig, optional): path to quantization config, only required when model is not prepared.
        inplace (bool, optional): It will change the given model in-place if True.

    Returns:
        The quantized model.
    """
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
            )
    setattr(q_model, "is_quantized", True)
    return q_model


def finalize_calibration(model):
    """Generate and save calibration info."""
    from neural_compressor.torch.algorithms.fp8_quant import save_calib_result

    save_calib_result(model)
