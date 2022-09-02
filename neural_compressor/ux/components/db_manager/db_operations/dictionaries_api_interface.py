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
# pylint: disable=no-member
"""INC Bench Dictionaries API interface."""

from sqlalchemy.orm import sessionmaker

from neural_compressor.ux.components.db_manager.db_manager import DBManager
from neural_compressor.ux.components.db_manager.db_models.dataloader import Dataloader
from neural_compressor.ux.components.db_manager.db_models.domain import Domain
from neural_compressor.ux.components.db_manager.db_models.domain_flavour import DomainFlavour
from neural_compressor.ux.components.db_manager.db_models.framework import Framework
from neural_compressor.ux.components.db_manager.db_models.metric import Metric
from neural_compressor.ux.components.db_manager.db_models.optimization_type import OptimizationType
from neural_compressor.ux.components.db_manager.db_models.precision import Precision
from neural_compressor.ux.components.db_manager.db_models.transform import Transform
from neural_compressor.ux.utils.exceptions import ClientErrorException

db_manager = DBManager()
Session = sessionmaker(bind=db_manager.engine)


class DictionariesAPIInterface:
    """Interface for queries connected with dictonaries."""

    @staticmethod
    def list_domains(data: dict) -> dict:
        """List model domains."""
        with Session.begin() as db_session:
            domains_list = Domain.list(db_session)
        return domains_list

    @staticmethod
    def list_domain_flavours(data: dict) -> dict:
        """List model domain flavours."""
        with Session.begin() as db_session:
            domain_flavours_list = DomainFlavour.list(db_session)
        return domain_flavours_list

    @staticmethod
    def list_optimization_types(data: dict) -> dict:
        """List optimization types."""
        with Session.begin() as db_session:
            optimization_types = OptimizationType.list(db_session)
        return optimization_types

    @staticmethod
    def list_optimization_types_for_precision(data: dict) -> dict:
        """List optimization types."""
        precision_name = data.get("precision", None)
        if precision_name is None:
            raise ClientErrorException("Precision name not specified.")
        with Session.begin() as db_session:
            optimization_types = OptimizationType.list_for_precision(db_session, precision_name)
        return optimization_types

    @staticmethod
    def list_precisions(data: dict) -> dict:
        """List precisions."""
        with Session.begin() as db_session:
            precisions = Precision.list(db_session)
        return precisions

    @staticmethod
    def list_dataloaders(data: dict) -> dict:
        """List all dataloaders."""
        with Session.begin() as db_session:
            dataloaders = Dataloader.list(db_session)
        return dataloaders

    @staticmethod
    def list_dataloaders_by_framework(data: dict) -> dict:
        """List dataloaders for specified framework."""
        framework = data.get("framework", None)
        if framework is None:
            raise ClientErrorException("Could not find framework name.")

        with Session.begin() as db_session:
            framework_id = Framework.get_framework_id(db_session, framework)
            fw_dataloaders = Dataloader.list_by_framework(db_session, framework_id)
        return fw_dataloaders

    @staticmethod
    def list_transforms(data: dict) -> dict:
        """List all transforms."""
        with Session.begin() as db_session:
            transforms = Transform.list(db_session)
        return transforms

    @staticmethod
    def list_transforms_by_framework(data: dict) -> dict:
        """List transforms for specified framework."""
        framework = data.get("framework", None)
        if framework is None:
            raise ClientErrorException("Could not find framework name.")

        with Session.begin() as db_session:
            framework_id = Framework.get_framework_id(db_session, framework)
            fw_transforms = Transform.list_by_framework(db_session, framework_id)
        return fw_transforms

    @staticmethod
    def list_transforms_by_domain(data: dict) -> dict:
        """List transforms for specified domain."""
        domain = data.get("domain", None)
        if domain is None:
            raise ClientErrorException("Could not find domain name.")

        with Session.begin() as db_session:
            domain_id = Domain.get_domain_id(db_session, domain)
            fw_transforms = Transform.list_by_domain(db_session, domain_id)
        return fw_transforms

    @staticmethod
    def list_metrics(data: dict) -> dict:
        """List all metrics."""
        with Session.begin() as db_session:
            metrics = Metric.list(db_session)
        return metrics

    @staticmethod
    def list_metrics_by_framework(data: dict) -> dict:
        """List metrics for specified framework."""
        framework = data.get("framework", None)
        if framework is None:
            raise ClientErrorException("Could not find framework name.")

        with Session.begin() as db_session:
            framework_id = Framework.get_framework_id(db_session, framework)
            fw_metrics = Metric.list_by_framework(db_session, framework_id)
        return fw_metrics
