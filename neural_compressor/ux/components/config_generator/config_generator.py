# -*- coding: utf-8 -*-
# Copyright (c) 2022 Intel Corporation
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
"""Configuration generator class."""

from abc import abstractmethod
from copy import deepcopy
from typing import List, Optional

from neural_compressor.ux.utils.json_serializer import JsonSerializer
from neural_compressor.ux.utils.utils import get_predefined_config_path, normalize_string
from neural_compressor.ux.utils.workload.dataloader import Dataloader, Filter
from neural_compressor.ux.utils.workload.model import Model


class ConfigGenerator(JsonSerializer):
    """Configuration generator class."""

    def __init__(self, workload_directory: str, configuration_path: str, data: dict):
        """Initialize configuration generator."""
        super().__init__()
        self.workdir: str = workload_directory
        self.config_path: str = configuration_path
        self.framework: str = data["framework"]
        self.model_name: str = data["model"]["name"]
        self.model_path: str = data["model"]["input_graph"]
        self.model_domain: str = data["model"]["domain"]
        self.model_domain_flavour: str = data["model"]["domain_flavour"]
        self.model_inputs: List[str] = data.get("model", {}).get("input_nodes", [])
        self.model_outputs: List[str] = data.get("model", {}).get("output_nodes", [])
        self.batch_size: int = data["batch_size"]
        self.dataset: dict = data["dataloader"]["dataset"]
        self.dataset_type = data["dataset_type"]
        self.transforms: List[dict] = data["dataloader"]["transforms"]
        self.filter: dict = data["dataloader"]["filter"]
        self.metric: dict = data["dataloader"]["metric"]
        self.predefined_config_path: str = self.get_predefined_config_path()

    def get_predefined_config_path(self) -> str:
        """Get path to predefined config."""
        return get_predefined_config_path(
            self.framework,
            self.model_domain,
            self.model_domain_flavour,
        )

    @abstractmethod
    def generate(self) -> None:
        """Generate yaml config file."""
        raise NotImplementedError

    def generate_model_config(self) -> Model:
        """Generate model configuration."""
        model = Model()
        model.framework = self.framework
        model.name = normalize_string(self.model_name)
        model.inputs = self.model_inputs
        model.outputs = self.model_outputs
        return model

    def generate_dataloader_config(self, batch_size: Optional[int] = None) -> Dataloader:
        """Generate dataloader configuration."""
        dataloader = Dataloader()
        if batch_size is None:
            batch_size = self.batch_size
        dataloader.batch_size = batch_size

        if self.transforms:
            dataloader.set_transforms_from_list(
                deepcopy(self.transforms),
            )

        dataloader.set_dataset(
            deepcopy(self.dataset),
        )

        if self.filter:
            dataloader.filter = Filter(
                deepcopy(self.filter),
            )
        return dataloader
