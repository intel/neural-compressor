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

import os
from typing import Any, Dict, List, Optional

from lpot.ux.utils.exceptions import ClientErrorException
from lpot.ux.utils.utils import (
    check_module,
    is_model_file,
    load_dataloader_config,
    load_model_config,
    load_transforms_config,
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
    def get_frameworks() -> List[str]:
        """Get list of available frameworks."""
        models_config = load_model_config()
        return list(models_config.keys())

    def get_domains(self) -> List[str]:
        """Get list of available domains."""
        if self.config is None:
            raise ClientErrorException("Config not found.")
        framework = self.config.get("framework", None)
        if framework is None:
            raise ClientErrorException("Framework not set.")
        models_config = load_model_config()
        return list(models_config.get(framework, {}).keys())

    def get_models(self) -> List[str]:
        """Get list of models."""
        if self.config is None:
            raise ClientErrorException("Config not found.")
        framework = self.config.get("framework", None)
        if framework is None:
            raise ClientErrorException("Framework not set.")
        domain = self.config.get("domain", None)
        if domain is None:
            raise ClientErrorException("Domain not set.")
        models_config = load_model_config()

        return list(models_config.get(framework, {}).get(domain, {}).keys())

    def get_available_models(self, workspace_path: str) -> List[str]:
        """Get list of available models in workspace."""
        available_models = []
        all_models = self.get_models()
        for filename in os.listdir(workspace_path):
            name = os.path.splitext(filename)[0]
            if (
                os.path.isfile(os.path.join(workspace_path, filename))
                and name in all_models
                and is_model_file(filename)
            ):
                available_models.append(filename)
        return available_models

    def get_dataloaders(self) -> Dict[str, Any]:
        """Get available dataloaders."""
        if self.config is None:
            raise ClientErrorException("Config not found.")
        framework = self.config.get("framework", None)
        if framework is None:
            raise ClientErrorException("Framework not set.")
        dataloaders = load_dataloader_config()
        return dataloaders.get(framework, {})

    def get_transforms(self) -> Dict[str, Any]:
        """Get available transforms."""
        if self.config is None:
            raise ClientErrorException("Config not found.")
        framework = self.config.get("framework", None)
        if framework is None:
            raise ClientErrorException("Framework not set.")
        transforms = load_transforms_config()
        return transforms.get(framework, {})

    @staticmethod
    def get_objectives() -> List[str]:
        """Get list of supported objectives."""
        check_module("lpot")
        from lpot.objective import OBJECTIVES

        objectives = list(OBJECTIVES.keys())
        return objectives

    @staticmethod
    def get_strategies() -> List[str]:
        """Get list of supported strategies."""
        check_module("lpot")
        from lpot.strategy import STRATEGIES

        strategies = list(STRATEGIES.keys())
        return sorted(strategies)

    def get_quantization_approaches(self) -> List[str]:
        """Get list of supported quantization approaches."""
        approaches = [
            "post_training_static_quant",
            "quant_aware_training",
        ]
        framework = self.config.get("framework", None)
        if framework in ["pytorch", "onnxruntime"]:
            approaches.append("post_training_dynamic_quant")

        return approaches

    def get_metrics(self) -> Dict[str, Any]:
        """Get list of possible metrics."""
        check_module("lpot")
        framework = self.config.get("framework", None)
        if framework is None:
            raise ClientErrorException("Framework not set.")

        if framework == "pytorch":
            check_module("ignite")
        else:
            check_module(framework)

        from lpot.metric.metric import framework_metrics

        metric_list = list(framework_metrics.get(framework)().metrics.keys())
        metrics = update_metric_parameters(metric_list)

        return metrics


def update_metric_parameters(metric_list: List[str]) -> Dict[str, Any]:
    """Add parameters to metrics."""
    metrics: Dict[str, Any] = {}
    for metric in metric_list:
        if metric == "topk":
            metrics.update({metric: {"k": [1, 5]}})
        elif metric == "COCOmAP":
            metrics.update({metric: {"anno_path": ""}})
        elif metric in ["MSE", "RMSE", "MAE"]:
            metrics.update({metric: {"compare_label": True}})
        else:
            metrics.update({metric: None})
    return metrics


def get_possible_values(data: dict) -> Dict[str, List[Any]]:
    """
    Get list of possible values for specified scenario.

    Example expected data:
    {
        "param": "dataloader",
        "config": {
            "framework": "tensorflow"
        }
    }
    """
    feeder = Feeder(data)
    return feeder.feed()
