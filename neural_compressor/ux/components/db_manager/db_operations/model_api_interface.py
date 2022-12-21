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
"""INC Bench Model API interface."""

from sqlalchemy.orm import sessionmaker

from neural_compressor.ux.components.configuration_wizard.configuration_parser import (
    ConfigurationParser,
)
from neural_compressor.ux.components.db_manager.db_manager import DBManager
from neural_compressor.ux.components.db_manager.db_models.framework import Framework
from neural_compressor.ux.components.db_manager.db_models.model import Model
from neural_compressor.ux.components.db_manager.params_interfaces import ModelAddParamsInterface
from neural_compressor.ux.utils.exceptions import ClientErrorException

db_manager = DBManager()
Session = sessionmaker(bind=db_manager.engine)


class ModelAPIInterface:
    """Interface for queries connected with models."""

    @staticmethod
    def add_model(data: dict) -> int:
        """Add model details."""
        parser = ConfigurationParser()
        parsed_input_data = parser.parse(data)

        model_parameters: ModelAddParamsInterface = ModelAPIInterface.parse_model_data(
            parsed_input_data,
        )
        with Session.begin() as db_session:
            framework_id = Framework.get_framework_id(db_session, model_parameters.framework)
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
    def get_model_details(data: dict) -> dict:
        """Parse input data and get model details."""
        try:
            model_id: int = int(data.get("id", None))
        except ValueError:
            raise ClientErrorException("Incorrect model id.")
        except TypeError:
            raise ClientErrorException("Could not find model id.")

        with Session.begin() as db_session:
            model_details = Model.details(
                db_session,
                model_id,
            )
        return model_details

    @staticmethod
    def list_models(data: dict) -> dict:
        """List models assigned to project."""
        try:
            project_id: int = int(data.get("project_id", None))
        except ValueError:
            raise ClientErrorException("Incorrect project id.")
        except TypeError:
            raise ClientErrorException("Could not find project id.")

        with Session.begin() as db_session:
            models_list = Model.list(
                db_session,
                project_id,
            )

        return models_list

    @staticmethod
    def delete_model(data: dict) -> dict:
        """Delete model from database."""
        try:
            model_id: int = int(data.get("id", None))
            model_name: str = str(data.get("name", None))
        except ValueError:
            raise ClientErrorException("Could not parse value.")
        except TypeError:
            raise ClientErrorException("Missing model id or model name.")
        with Session.begin() as db_session:
            removed_model_id = Model.delete_model(
                db_session=db_session,
                model_id=model_id,
                model_name=model_name,
            )

        return {"id": removed_model_id}

    @staticmethod
    def parse_model_data(data: dict) -> ModelAddParamsInterface:
        """Parse input data for model."""
        model_parameters = ModelAddParamsInterface()
        try:
            model_parameters.project_id = int(data.get("project_id", None))
            model_parameters.name = str(data.get("model_name", None))
            model_parameters.path = str(data.get("model_path", None))
            model_parameters.framework = str(data.get("framework", None))
            model_parameters.size = float(data.get("size", None))
            model_parameters.precision_id = int(data.get("precision_id", None))
            model_parameters.domain_id = int(data.get("domain_id", None))
            model_parameters.domain_flavour_id = int(data.get("domain_flavour_id", None))
            model_parameters.input_nodes = ModelAddParamsInterface.parse_nodes(
                nodes=data.get("input_nodes", None),
            )
            model_parameters.output_nodes = ModelAddParamsInterface.parse_nodes(
                nodes=data.get("output_nodes", None),
            )
            model_parameters.supports_profiling = data.get("supports_profiling", False)
            model_parameters.supports_graph = data.get("supports_graph", False)
            model_parameters.supports_pruning = data.get("supports_pruning", False)

        except ValueError:
            raise ClientErrorException("Could not parse value")
        except TypeError:
            raise ClientErrorException("Could not find required parameter.")

        return model_parameters
