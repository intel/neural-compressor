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
"""DB Manager utils module."""
from typing import List

from sqlalchemy.orm import session

from neural_compressor.ux.components.db_manager.db_models.dataloader import Dataloader
from neural_compressor.ux.utils.consts import Frameworks
from neural_compressor.ux.utils.exceptions import InternalException
from neural_compressor.ux.utils.utils import load_dataloader_config, load_transforms_config


def update_dataloaders_params(
    db_session: session.Session,
    framework_id: int,
    dataloaders_to_update: List[str],
    framework_dataloaders_config: List[dict],
) -> None:
    """Update dataloaders' params."""
    framework_dataloaders = Dataloader.list_by_framework(db_session, framework_id)["dataloaders"]

    for dataloader_to_update in dataloaders_to_update:
        dataloader_id: int = list(
            filter(
                lambda dl: (dl["id"] if dl["name"] == dataloader_to_update else None),
                framework_dataloaders,
            ),
        )[0].get("id")

        dataloader_params = []
        for dl in framework_dataloaders_config:
            if dl["name"] == dataloader_to_update:
                dataloader_params = dl["params"]
        Dataloader.update_params(db_session, dataloader_id, dataloader_params)


def get_framework_dataloaders_config(framework: Frameworks) -> List[dict]:
    """Get dataloaders config for specified framework."""
    dataloaders_config = load_dataloader_config()
    for fwk_dataloaders in dataloaders_config:
        if fwk_dataloaders["name"] == framework.value:
            return fwk_dataloaders["params"]
    raise InternalException(f"Could not find dataloaders config for {framework.value} framework.")


def get_framework_transforms_config(framework: Frameworks) -> List[dict]:
    """Get transformations config for specified framework."""
    transforms_config = load_transforms_config()
    for fwk_transforms in transforms_config:
        if fwk_transforms["name"] == framework.value:
            return fwk_transforms["params"]
    raise InternalException(
        f"Could not find transformations config for {framework.value} framework.",
    )
