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
"""INC Bench Project API interface."""
import os
import shutil
from copy import copy

from sqlalchemy.orm import session, sessionmaker

from neural_compressor.ux.components.configuration_wizard.configuration_parser import (
    ConfigurationParser,
)
from neural_compressor.ux.components.db_manager.db_manager import DBManager
from neural_compressor.ux.components.db_manager.db_models.dataset import Dataset
from neural_compressor.ux.components.db_manager.db_models.domain import Domain
from neural_compressor.ux.components.db_manager.db_models.domain_flavour import DomainFlavour
from neural_compressor.ux.components.db_manager.db_models.framework import Framework
from neural_compressor.ux.components.db_manager.db_models.model import Model
from neural_compressor.ux.components.db_manager.db_models.precision import Precision
from neural_compressor.ux.components.db_manager.db_models.project import Project
from neural_compressor.ux.components.db_manager.db_operations.model_api_interface import (
    ModelAPIInterface,
)
from neural_compressor.ux.components.db_manager.params_interfaces import ModelAddParamsInterface
from neural_compressor.ux.components.model.repository import ModelRepository
from neural_compressor.ux.utils.consts import WORKSPACE_LOCATION, Domains
from neural_compressor.ux.utils.exceptions import ClientErrorException, InternalException
from neural_compressor.ux.utils.utils import normalize_string

db_manager = DBManager()
Session = sessionmaker(bind=db_manager.engine)


class ProjectAPIInterface:
    """Interface for queries connected with project."""

    @staticmethod
    def get_project_details(data: dict) -> dict:
        """Get project details from database."""
        try:
            project_id: int = int(data.get("id", None))
        except ValueError:
            raise ClientErrorException("Incorrect project id.")
        except TypeError:
            raise ClientErrorException("Could not find project id.")
        with Session.begin() as db_session:
            project_details = Project.project_details(
                db_session=db_session,
                project_id=project_id,
            )
        return project_details

    @staticmethod
    def list_projects(data: dict) -> dict:
        """List projects from database."""
        with Session.begin() as db_session:
            projects_list = Project.list_projects(db_session)
        return projects_list

    @staticmethod
    def create_project(data: dict) -> dict:
        """Create new project and add input model."""
        project_name = data.get("name", None)
        if project_name is None:
            ClientErrorException("Project name not provided.")

        with Session.begin() as db_session:
            project_id = Project.create_project(db_session, project_name)
            data.update({"project_id": project_id})
            data["model"].update({"model_name": "Input model"})
            model_id = ProjectAPIInterface.add_model(db_session, data)
            dataset_id = ProjectAPIInterface.add_dummy_dataset(db_session, data)
        return {
            "project_id": project_id,
            "model_id": model_id,
            "dataset_id": dataset_id,
        }

    @staticmethod
    def delete_project(data: dict) -> dict:
        """Delete project details from database and clean workspace."""
        try:
            project_id: int = int(data.get("id", None))
            project_name: str = str(data.get("name", None))
        except ValueError:
            raise ClientErrorException("Could not parse value.")
        except TypeError:
            raise ClientErrorException("Missing project id or project name.")
        with Session.begin() as db_session:
            removed_project_id = Project.delete_project(
                db_session=db_session,
                project_id=project_id,
                project_name=project_name,
            )

        if removed_project_id is not None:
            normalized_project_name = normalize_string(project_name)
            project_location = os.path.join(
                WORKSPACE_LOCATION,
                f"{normalized_project_name}_{removed_project_id}",
            )
            shutil.rmtree(project_location, ignore_errors=True)

        return {"id": removed_project_id}

    @staticmethod
    def add_model(db_session: session.Session, data: dict) -> int:
        """Create new project with input model."""
        model_data = copy(data.get("model", {}))
        model_data.update({"project_id": data.get("project_id")})
        model_path = model_data.get("path")
        model_data.update({"model_path": model_path})
        model = ModelRepository().get_model(model_path)
        model_data.update({"size": model.size})
        supports_profiling = model.supports_profiling
        model_data.update({"supports_profiling": supports_profiling})
        supports_graph = model.supports_graph
        model_data.update({"supports_graph": supports_graph})
        supports_pruning = model.supports_pruning
        model_data.update({"supports_pruning": supports_pruning})
        framework = model.get_framework_name()
        framework_id = Framework.get_framework_id(db_session, framework)
        model_data.update({"framework_id": framework_id})
        precision = model_data.get("precision", "fp32")
        precision_id = Precision.get_precision_id(db_session, precision)
        model_data.update({"precision_id": precision_id})
        domain = model_data.get("domain")
        if not domain:
            domain = Domains.NONE.value
        domain_id = Domain.get_domain_id(db_session, domain)
        model_data.update({"domain_id": domain_id})
        del model_data["domain"]

        domain_flavour = model.domain.domain_flavour
        domain_flavour_id = DomainFlavour.get_domain_flavour_id(db_session, domain_flavour)
        model_data.update({"domain_flavour_id": domain_flavour_id})
        model_parameters: ModelAddParamsInterface = ModelAPIInterface.parse_model_data(
            model_data,
        )

        model_id = Model.add(
            db_session,
            project_id=model_parameters.project_id,
            name=model_parameters.name,
            path=model_parameters.path,
            framework_id=framework_id,
            size=model_parameters.size,
            precision_id=model_parameters.precision_id,
            domain_id=model_parameters.domain_id,
            domain_flavour_id=model_parameters.domain_flavour_id,
            input_nodes=model_parameters.input_nodes,
            output_nodes=model_parameters.output_nodes,
            supports_profiling=model_parameters.supports_profiling,
            supports_graph=model_parameters.supports_graph,
            supports_pruning=model_parameters.supports_pruning,
        )
        return model_id

    @staticmethod
    def add_dummy_dataset(db_session: session.Session, data: dict) -> int:
        """Add dummy dataset to project."""
        received_shape = data.get("model", {}).get("shape", None)
        model_path = data.get("model", {}).get("path", None)
        project_id = data.get("project_id", None)
        if project_id is None:
            raise InternalException("Could not find project id.")
        if model_path is not None:
            if ".py" == model_path[-3:]:
                dataset_id = Dataset.add(
                    project_id=project_id,
                    db_session=db_session,
                    dataset_name="dummy",
                    dataset_type="dummy_v0",
                    parameters={
                        "input_shape": "NA",
                        "label_shape": [1],
                    },
                )
                return dataset_id
        if received_shape is None:
            return -1
        shape = ConfigurationParser.parse_value(received_shape, [[int]])  # type: ignore
        if len(shape[0]) <= 0:
            return -1

        dataset_id = Dataset.add(
            project_id=project_id,
            db_session=db_session,
            dataset_name="dummy",
            dataset_type="dummy_v2",
            parameters={
                "input_shape": shape,
                "label_shape": [1],
            },
        )
        return dataset_id

    @staticmethod
    def update_project_notes(data: dict) -> dict:
        """Update project notes."""
        try:
            project_id: int = int(data.get("id", None))
        except ValueError:
            raise ClientErrorException("Incorrect project id.")
        except TypeError:
            raise ClientErrorException("Could not find project id.")

        notes: str = data.get("notes", None)
        if notes is None:
            raise ClientErrorException("Notes not provided.")

        with Session.begin() as db_session:
            response_data = Project.update_notes(
                db_session,
                project_id,
                notes,
            )
        return response_data
