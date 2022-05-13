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
"""The Dataset class."""

import json
from typing import Any, Optional, Union

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Session, relationship, session
from sqlalchemy.sql import func

from neural_compressor.ux.components.db_manager.db_manager import Base
from neural_compressor.ux.utils.exceptions import ClientErrorException
from neural_compressor.ux.utils.logger import log

log.debug("Initializing Dataset table")


class Dataset(Base):
    """INC Bench datasets' table representation."""

    __tablename__ = "dataset"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("project.id"))
    name = Column(String(50), nullable=False)
    dataset_type = Column(String(50), nullable=False)
    parameters = Column(String, default=None, nullable=True)
    transforms = Column(String, default=None, nullable=True)
    filter = Column(String, default=None, nullable=True)
    metric = Column(String, default=None, nullable=True)
    template_path = Column(String, default=None, nullable=True)
    created_at = Column(DateTime, nullable=False, default=func.now())

    project: Any = relationship("Project", back_populates="datasets")

    @staticmethod
    def add(
        db_session: session.Session,
        project_id: int,
        dataset_name: str,
        dataset_type: str,
        parameters: Optional[dict] = None,
        transforms: Optional[dict] = None,
        filter: Optional[dict] = None,
        metric: Optional[dict] = None,
        template_path: Optional[str] = None,
    ) -> int:
        """
        Add dataset to database.

        Returns id of added dataset.
        """
        parsed_params: Union[Optional[dict], str] = parameters
        if parameters is not None:
            parsed_params = json.dumps(parameters)

        parsed_transforms: Union[Optional[dict], str] = transforms
        if transforms is not None:
            parsed_transforms = json.dumps(transforms)

        parsed_filter: Union[Optional[dict], str] = filter
        if filter is not None:
            parsed_filter = json.dumps(filter)

        parsed_metric: Union[Optional[dict], str] = metric
        if metric is not None:
            parsed_metric = json.dumps(metric)

        new_dataset = Dataset(
            project_id=project_id,
            name=dataset_name,
            dataset_type=dataset_type,
            parameters=parsed_params,
            transforms=parsed_transforms,
            filter=parsed_filter,
            metric=parsed_metric,
            template_path=template_path,
        )
        db_session.add(new_dataset)
        db_session.flush()

        return int(new_dataset.id)

    @staticmethod
    def delete_dataset(
        db_session: session.Session,
        dataset_id: int,
        dataset_name: str,
    ) -> Optional[int]:
        """Remove dataset from database."""
        dataset = (
            db_session.query(Dataset)
            .filter(Dataset.id == dataset_id)
            .filter(Dataset.name == dataset_name)
            .one_or_none()
        )
        if dataset is None:
            return None
        db_session.delete(dataset)
        db_session.flush()

        return int(dataset.id)

    @staticmethod
    def details(db_session: Session, dataset_id: int) -> dict:
        """Get dataset details."""
        dataset = db_session.query(Dataset).filter(Dataset.id == dataset_id).one_or_none()
        if dataset is None:
            raise ClientErrorException("Could not found specified dataset in database.")

        return Dataset.build_info(dataset)

    @staticmethod
    def list(db_session: Session, project_id: int) -> dict:
        """Get dataset list for specified project from database."""
        datasets = []
        dataset_instances = db_session.query(
            Dataset.id,
            Dataset.name,
            Dataset.dataset_type,
            Dataset.created_at,
        ).filter(Dataset.project_id == project_id)
        for dataset in dataset_instances:
            datasets.append(
                {
                    "id": dataset.id,
                    "name": dataset.name,
                    "dataset_type": dataset.dataset_type,
                    "created_at": str(dataset.created_at),
                },
            )
        return {"datasets": datasets}

    @staticmethod
    def update_template_path(
        db_session: session.Session,
        dataset_id: int,
        template_path: str,
    ) -> dict:
        """Update template path for dataset."""
        dataset = db_session.query(Dataset).filter(Dataset.id == dataset_id).one()
        dataset.template_path = template_path
        db_session.add(dataset)
        db_session.flush()

        return {
            "id": dataset.id,
            "template_path": dataset.template_path,
        }

    @staticmethod
    def build_info(
        dataset: Any,
    ) -> dict:
        """Build dataset info."""
        parameters = dataset.parameters
        if parameters:
            parameters = json.loads(dataset.parameters)

        transforms = dataset.transforms
        if transforms:
            transforms = json.loads(dataset.transforms)

        filter = dataset.filter
        if filter:
            filter = json.loads(dataset.filter)

        metric = dataset.metric
        if metric:
            metric = json.loads(dataset.metric)

        return {
            "id": dataset.id,
            "project_id": dataset.project_id,
            "name": dataset.name,
            "dataset_type": dataset.dataset_type,
            "parameters": parameters,
            "transforms": transforms,
            "filter": filter,
            "metric": metric,
            "template_path": dataset.template_path,
            "created_at": str(dataset.created_at),
        }
