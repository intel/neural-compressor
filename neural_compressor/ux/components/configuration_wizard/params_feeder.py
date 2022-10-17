# -*- coding: utf-8 -*-
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Parameters feeder module."""
from typing import Any, Dict, List, Optional

from neural_compressor.experimental.metric.metric import registry_metrics
from neural_compressor.objective import OBJECTIVES
from neural_compressor.strategy import STRATEGIES
from neural_compressor.ux.components.model.repository import ModelRepository
from neural_compressor.ux.utils.exceptions import ClientErrorException
from neural_compressor.ux.utils.utils import (
    _update_metric_parameters,
    check_module,
    filter_transforms,
    load_dataloader_config,
    load_help_nc_params,
    load_metrics_config,
    load_model_config,
    load_precisions_config,
    load_transforms_config,
    normalize_framework,
)


class Feeder:
    """Parameters feeder class."""

    def __init__(self, data: Dict[str, Any]) -> None:
        """Initialize parameters feeder class."""
        self.param: Optional[str] = data.get("param")
        self.config: Dict[str, Any] = data.get("config", {})

    def feed(self) -> Dict[str, Any]:
        """Feed the parameters."""
        param_mapper = {
            "framework": self.get_frameworks,
            "domain": self.get_domains,
            "model": self.get_models,
            "dataloader": self.get_dataloaders,
            "transform": self.get_transforms,
            "objective": self.get_objectives,
            "strategy": self.get_strategies,
            "quantization_approach": self.get_quantization_approaches,
            "metric": self.get_metrics,
            "precision": self.get_precisions,
        }
        if self.param is None:
            raise ClientErrorException("Parameter not defined.")
        get_param = param_mapper.get(self.param, None)
        if get_param is None:
            raise ClientErrorException(
                f"Could not found method for {self.param} parameter.",
            )

        return {
            self.param: get_param(),
        }

    @staticmethod
    def get_frameworks() -> List[dict]:
        """Get list of available frameworks."""
        supported_frameworks = ModelRepository.get_supported_frameworks()
        frameworks = []
        models_config = load_model_config()
        for framework in models_config.keys():
            if framework.startswith("__help__"):
                continue
            if framework not in supported_frameworks:
                continue
            help_msg = models_config.get(f"__help__{framework}", "")
            frameworks.append({"name": framework, "help": help_msg})
        return frameworks

    def get_domains(self) -> List[Dict[str, Any]]:
        """Get list of available domains."""
        framework = self.config.get("framework", None)
        if framework is None:
            raise ClientErrorException("Framework not set.")
        models_config = load_model_config()
        domains = []
        for domain in models_config.get(framework, {}).keys():
            if domain.startswith("__help__"):
                continue
            help_msg = models_config.get(framework, {}).get(f"__help__{domain}", "")
            domains.append(
                {
                    "name": domain,
                    "help": help_msg,
                },
            )
        return domains

    def get_models(self) -> List[Dict[str, Any]]:
        """Get list of models."""
        framework = self.config.get("framework", None)
        if framework is None:
            raise ClientErrorException("Framework not set.")
        domain = self.config.get("domain", None)
        if domain is None:
            raise ClientErrorException("Domain not set.")
        models_config = load_model_config()

        raw_models_dict = models_config.get(framework, {}).get(domain, {})
        models = []
        for model in raw_models_dict.keys():
            if model.startswith("__help__"):
                continue
            help_msg = raw_models_dict.get(f"__help__{model}", "")
            models.append({"name": model, "help": help_msg})
        return models

    def get_dataloaders(self) -> List[Dict[str, Any]]:
        """Get available dataloaders."""
        framework = self.config.get("framework", None)
        if framework is None:
            raise ClientErrorException("Framework not set.")
        for fw_dataloader in load_dataloader_config():
            if fw_dataloader.get("name") == framework:
                return fw_dataloader.get("params", [])
        return []

    def get_transforms(self) -> List[Dict[str, Any]]:
        """Get available transforms."""
        framework = self.config.get("framework", None)
        if framework is None:
            raise ClientErrorException("Framework not set.")
        domain = self.config.get("domain", None)
        transforms = []
        for fw_transforms in load_transforms_config():
            if fw_transforms.get("name") == framework:
                transforms = fw_transforms.get("params", [])
                break
        if domain is not None:
            transforms = filter_transforms(transforms, framework, domain)
        return transforms

    @staticmethod
    def get_objectives() -> List[dict]:
        """Get list of supported objectives."""
        help_dict = load_help_nc_params("objectives")

        objectives = []
        for objective in OBJECTIVES.keys():
            help_msg = help_dict.get(f"__help__{objective}", "")
            objectives.append({"name": objective, "help": help_msg})
        return objectives

    @staticmethod
    def get_strategies() -> List[Dict[str, Any]]:
        """Get list of supported strategies."""
        help_dict = load_help_nc_params("strategies")
        strategies = []
        for strategy in STRATEGIES.keys():
            if "sigopt" == strategy:
                continue
            help_msg = help_dict.get(f"__help__{strategy}", "")
            strategies.append({"name": strategy, "help": help_msg})
        return strategies

    def get_precisions(self) -> List[dict]:
        """Get list of available precisions."""
        framework = self.config.get("framework", None)
        if framework is None:
            raise ClientErrorException("Framework not set.")
        return load_precisions_config().get(framework, [])

    def get_quantization_approaches(self) -> List[Dict[str, Any]]:
        """Get list of supported quantization approaches."""
        approaches = [
            {
                "name": "post_training_static_quant",
                "help": "help placeholder for post_training_static_quant",
            },
        ]
        framework = self.config.get("framework", None)
        if framework in ["pytorch", "onnxrt"]:
            approaches.append(
                {
                    "name": "post_training_dynamic_quant",
                    "help": f"help placeholder for {framework} post_training_dynamic_quant",
                },
            )

        return approaches

    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get list of possible metrics."""
        framework = self.config.get("framework", None)
        if framework is None:
            raise ClientErrorException("Framework not set.")

        framework = normalize_framework(framework)
        if framework == "pytorch":
            check_module("ignite")
        else:
            check_module(framework)

        inc_framework_name = "onnxrt_qlinearops" if framework == "onnxrt" else framework

        fw_metrics = registry_metrics.get(inc_framework_name, None)

        raw_metric_list = list(fw_metrics.keys()) if fw_metrics else []
        raw_metric_list += ["custom"]

        metrics = []

        loaded_metrics = load_metrics_config()

        for metric_name in raw_metric_list:
            if metric_name in [metric_item.get("name", None) for metric_item in loaded_metrics]:
                metric_config = None
                for metric_item in loaded_metrics:
                    if metric_item.get("name") == metric_name:
                        metric_config = metric_item
                if metric_config is not None:
                    metric = _update_metric_parameters(metric_config)
                    metrics.append(metric)
            else:
                metrics.append({"name": metric_name, "help": "", "value": None})

        return metrics


def get_possible_values(data: dict) -> Dict[str, List[Any]]:
    """Get list of possible values for specified scenario with "help" information."""
    feeder = Feeder(data)
    return feeder.feed()
