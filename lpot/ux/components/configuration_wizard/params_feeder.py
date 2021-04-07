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
from typing import Any, Dict, List, Optional, Union

from lpot.ux.utils.exceptions import ClientErrorException
from lpot.ux.utils.utils import (
    check_module,
    framework_extensions,
    is_model_file,
    load_dataloader_config,
    load_help_lpot_params,
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
    def get_frameworks() -> List[dict]:
        """Get list of available frameworks."""
        frameworks = []
        models_config = load_model_config()
        for framework in models_config.keys():
            if framework.startswith("__help__"):
                continue
            if framework not in framework_extensions.keys():
                continue
            help_msg = models_config.get(f"__help__{framework}", "")
            frameworks.append({"name": framework, "help": help_msg})
        return frameworks

    def get_domains(self) -> List[Dict[str, Any]]:
        """Get list of available domains."""
        if self.config is None:
            raise ClientErrorException("Config not found.")
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
        if self.config is None:
            raise ClientErrorException("Config not found.")
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

    def get_dataloaders(self) -> List[Dict[str, Any]]:
        """Get available dataloaders."""
        if self.config is None:
            raise ClientErrorException("Config not found.")
        framework = self.config.get("framework", None)
        if framework is None:
            raise ClientErrorException("Framework not set.")
        for fw_dataloader in load_dataloader_config():
            if fw_dataloader.get("name") == framework:
                return fw_dataloader.get("params", [])
        return []

    def get_transforms(self) -> List[Dict[str, Any]]:
        """Get available transforms."""
        if self.config is None:
            raise ClientErrorException("Config not found.")
        framework = self.config.get("framework", None)
        if framework is None:
            raise ClientErrorException("Framework not set.")
        for fw_transforms in load_transforms_config():
            if fw_transforms.get("name") == framework:
                return fw_transforms.get("params", [])
        return []

    @staticmethod
    def get_objectives() -> List[dict]:
        """Get list of supported objectives."""
        check_module("lpot")
        from lpot.objective import OBJECTIVES

        help_dict = load_help_lpot_params("objectives")

        objectives = []
        for objective in OBJECTIVES.keys():
            help_msg = help_dict.get(f"__help__{objective}", "")
            objectives.append({"name": objective, "help": help_msg})
        return objectives

    @staticmethod
    def get_strategies() -> List[Dict[str, Any]]:
        """Get list of supported strategies."""
        check_module("lpot")
        from lpot.strategy import STRATEGIES

        help_dict = load_help_lpot_params("strategies")
        strategies = []
        for strategy in STRATEGIES.keys():
            help_msg = help_dict.get(f"__help__{strategy}", "")
            strategies.append({"name": strategy, "help": help_msg})
        return strategies

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
        check_module("lpot")
        framework = self.config.get("framework", None)
        if framework is None:
            raise ClientErrorException("Framework not set.")

        if framework == "pytorch":
            check_module("ignite")
        else:
            check_module(framework)
        from lpot.experimental.metric.metric import framework_metrics

        help_dict = load_help_lpot_params("metrics")
        if framework == "onnxrt":
            raw_metric_list = list(
                framework_metrics.get("onnxrt_qlinearops")().metrics.keys(),
            )
        else:
            raw_metric_list = list(framework_metrics.get(framework)().metrics.keys())
        raw_metric_list += ["custom"]
        metrics_updated = update_metric_parameters(raw_metric_list)
        for metric, value in metrics_updated.copy().items():
            if isinstance(value, dict):
                for key in value.copy().keys():
                    help_msg_key = f"__help__{key}"
                    metrics_updated[metric][help_msg_key] = help_dict.get(
                        metric,
                        {},
                    ).get(help_msg_key, "")
            metrics_updated[f"__help__{metric}"] = help_dict.get(
                f"__help__{metric}",
                "",
            )
        return self._parse_help_in_dict(metrics_updated)

    def _parse_help_in_dict(self, data: dict) -> list:
        parsed_list = []
        for key, value in data.items():
            if key.startswith("__help__"):
                continue
            if isinstance(value, dict):
                parsed_list.append(
                    {
                        "name": key,
                        "help": data.get(f"__help__{key}", ""),
                        "params": self._parse_help_in_dict(value),
                    },
                )
            else:
                parsed_list.append(
                    {
                        "name": key,
                        "help": data.get(f"__help__{key}", ""),
                        "value": value,
                    },
                )
        return parsed_list


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
    return convert_to_v1_api(feeder.feed())


def convert_to_v1_api(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert new API into old (without "help")."""
    data_v1 = {}
    for key, value in data.items():
        if isinstance(value, list):
            data_v1[key] = _convert_to_v1_api_list(value)
        else:
            data_v1[key] = value
    return data_v1


def _convert_to_v1_api_list(data: list) -> Union[List[str], Dict[str, Any]]:
    """Convert values in list with "help" args into dict or list, based on content."""
    data_v1_dict = {}
    data_v1_list = []
    for item in data:
        if isinstance(item, dict):
            if "params" in item.keys():
                params = item["params"]
                if isinstance(params, list):
                    data_v1_dict[item["name"]] = _convert_to_v1_api_list(params)
                else:
                    raise TypeError(
                        f"Type of params could be only type of list, not {type(params)}.",
                    )
            elif "value" in item.keys():
                data_v1_dict[item["name"]] = item["value"]
            else:
                data_v1_list.append(item["name"])
    if data_v1_dict and not data_v1_list:
        return data_v1_dict
    elif data_v1_list and not data_v1_dict:
        return data_v1_list
    else:
        raise Exception("Could not determine return type, error in input data.")


def get_possible_values_v2(data: dict) -> Dict[str, List[Any]]:
    """Get list of possible values for specified scenario with "help" information."""
    feeder = Feeder(data)
    return feeder.feed()
