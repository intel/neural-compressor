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
"""INC Bench database operations."""
import os
import shutil
from copy import copy
from shutil import copy as copy_file
from typing import Any, Dict, List, Optional, OrderedDict, Union

from alembic import command
from alembic.config import Config as AlembicConfig
from alembic.script import ScriptDirectory
from sqlalchemy.orm import session, sessionmaker

from neural_compressor.ux.components.benchmark import Benchmarks
from neural_compressor.ux.components.configuration_wizard.configuration_parser import (
    ConfigurationParser,
)
from neural_compressor.ux.components.db_manager.db_manager import DBManager
from neural_compressor.ux.components.db_manager.db_models.benchmark import Benchmark
from neural_compressor.ux.components.db_manager.db_models.benchmark_result import BenchmarkResult
from neural_compressor.ux.components.db_manager.db_models.dataloader import Dataloader
from neural_compressor.ux.components.db_manager.db_models.dataset import Dataset
from neural_compressor.ux.components.db_manager.db_models.domain import Domain
from neural_compressor.ux.components.db_manager.db_models.domain_flavour import DomainFlavour
from neural_compressor.ux.components.db_manager.db_models.framework import Framework
from neural_compressor.ux.components.db_manager.db_models.metric import Metric
from neural_compressor.ux.components.db_manager.db_models.model import Model
from neural_compressor.ux.components.db_manager.db_models.optimization import Optimization
from neural_compressor.ux.components.db_manager.db_models.optimization_type import OptimizationType
from neural_compressor.ux.components.db_manager.db_models.precision import (
    Precision,
    precision_optimization_type_association,
)
from neural_compressor.ux.components.db_manager.db_models.profiling import Profiling
from neural_compressor.ux.components.db_manager.db_models.profiling_result import ProfilingResult
from neural_compressor.ux.components.db_manager.db_models.project import Project
from neural_compressor.ux.components.db_manager.db_models.transform import Transform
from neural_compressor.ux.components.db_manager.db_models.tuning_details import TuningDetails
from neural_compressor.ux.components.db_manager.db_models.tuning_history import TuningHistory
from neural_compressor.ux.components.db_manager.params_interfaces import (
    BenchmarkAddParamsInterface,
    DatasetAddParamsInterface,
    DiagnosisOptimizationParamsInterface,
    ModelAddParamsInterface,
    OptimizationAddParamsInterface,
    ProfilingAddParamsInterface,
    ProfilingResultAddParamsInterface,
    TuningHistoryInterface,
    TuningHistoryItemInterface,
)
from neural_compressor.ux.components.diagnosis.factory import DiagnosisFactory
from neural_compressor.ux.components.diagnosis.op_details import OpDetails
from neural_compressor.ux.components.model.repository import ModelRepository
from neural_compressor.ux.components.model_zoo.download_model import download_model
from neural_compressor.ux.components.names_mapper.names_mapper import MappingDirection, NamesMapper
from neural_compressor.ux.components.optimization.factory import OptimizationFactory
from neural_compressor.ux.components.optimization.optimization import (
    Optimization as OptimizationInterface,
)
from neural_compressor.ux.components.optimization.tune.tuning import (
    TuningDetails as TuningDetailsInterface,
)
from neural_compressor.ux.utils.consts import (
    WORKSPACE_LOCATION,
    Domains,
    ExecutionStatus,
    OptimizationTypes,
    Precisions,
    precision_optimization_types,
)
from neural_compressor.ux.utils.exceptions import ClientErrorException, InternalException
from neural_compressor.ux.utils.logger import log
from neural_compressor.ux.utils.utils import (
    get_predefined_config_path,
    load_model_config,
    load_model_wise_params,
    normalize_string,
)
from neural_compressor.ux.utils.workload.config import Config
from neural_compressor.ux.utils.workload.dataloader import Dataloader as DataloaderConfig
from neural_compressor.ux.utils.workload.dataloader import Dataset as DatasetConfig
from neural_compressor.ux.utils.workload.dataloader import Transform as TransformConfig
from neural_compressor.ux.web.communication import MessageQueue

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
        except ValueError:
            raise ClientErrorException("Could not parse value")
        except TypeError:
            raise ClientErrorException("Could not find required parameter.")

        return model_parameters


class DatasetAPIInterface:
    """Interface for queries connected with datasets."""

    @staticmethod
    def delete_dataset(data: dict) -> dict:
        """Delete dataset from database and clean workspace."""
        try:
            dataset_id: int = int(data.get("id", None))
            dataset_name: str = str(data.get("name", None))
        except ValueError:
            raise ClientErrorException("Could not parse value.")
        except TypeError:
            raise ClientErrorException("Missing dataset id or dataset name.")
        with Session.begin() as db_session:

            removed_dataset_id = Dataset.delete_dataset(
                db_session=db_session,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
            )

        return {"id": removed_dataset_id}

    @staticmethod
    def get_dataset_details(data: dict) -> dict:
        """Parse input data and get dataset details."""
        try:
            dataset_id: int = int(data.get("id", None))
        except ValueError:
            raise ClientErrorException("Incorrect dataset id.")
        except TypeError:
            raise ClientErrorException("Could not find dataset id.")

        with Session.begin() as db_session:
            dataset_details = Dataset.details(
                db_session,
                dataset_id,
            )
        return dataset_details

    @staticmethod
    def list_datasets(data: dict) -> dict:
        """List datasets assigned to project."""
        try:
            project_id: int = int(data.get("project_id", None))
        except ValueError:
            raise ClientErrorException("Incorrect project id.")
        except TypeError:
            raise ClientErrorException("Could not find project id.")

        with Session.begin() as db_session:
            datasets_list = Dataset.list(
                db_session,
                project_id,
            )
        return datasets_list

    @staticmethod
    def add_dataset(data: dict) -> dict:
        """Add dataset to database."""
        parser = ConfigurationParser()
        parsed_input_data = parser.parse(data)
        parsed_dataset_data: DatasetAddParamsInterface = DatasetAPIInterface.parse_dataset_data(
            parsed_input_data,
        )

        with Session.begin() as db_session:
            dataset_id = Dataset.add(
                db_session=db_session,
                project_id=parsed_dataset_data.project_id,
                dataset_name=parsed_dataset_data.dataset_name,
                dataset_type=parsed_dataset_data.dataset_type,
                parameters=parsed_dataset_data.parameters,
                transforms=parsed_dataset_data.transforms,
                filter=parsed_dataset_data.filter,
                metric=parsed_dataset_data.metric,
                template_path=None,
            )

        DatasetAPIInterface.set_template_path(dataset_id, parsed_dataset_data)

        return {
            "dataset_id": dataset_id,
        }

    @staticmethod
    def set_template_path(dataset_id: int, parsed_dataset_data: DatasetAddParamsInterface) -> None:
        """Set template path for dataset."""
        project_id = parsed_dataset_data.project_id
        with Session.begin() as db_session:
            project_name = Project.project_details(db_session, project_id).get("name", None)
        if project_name is None:
            raise ClientErrorException(f"Could not find project with id {project_id}")
        template_path = None
        custom_templates = DatasetAPIInterface.check_if_custom_metric_or_dataloader(
            parsed_dataset_data,
        )

        if any(custom_templates.values()):
            dataloader_path = DatasetAPIInterface.dataloader_path(
                project_id=project_id,
                project_name=project_name,
                dataset_id=dataset_id,
                dataset_name=parsed_dataset_data.dataset_name,
            )
            template_path = DatasetAPIInterface.generate_custom_template(
                dataloader_path,
                custom_templates,
            )

        if template_path:
            with Session.begin() as db_session:
                Dataset.update_template_path(
                    db_session=db_session,
                    dataset_id=dataset_id,
                    template_path=template_path,
                )

    @staticmethod
    def check_if_custom_metric_or_dataloader(dataset_data: DatasetAddParamsInterface) -> dict:
        """Check if dataset contains custom dataloader or metric."""
        is_custom_dataloader = dataset_data.dataset_type == "custom"
        is_custom_metric = dataset_data.metric.get("name") == "custom"
        return {
            "custom_dataloader": is_custom_dataloader,
            "custom_metric": is_custom_metric,
        }

    @staticmethod
    def generate_custom_template(dataloader_path: str, custom: dict) -> Optional[str]:
        """Generate template for custom dataloader or metric."""
        template_type = None
        template_path = None
        if custom.get("custom_metric") and custom.get("custom_dataloader"):
            template_type = "dataloader_and_metric"
        elif custom.get("custom_dataloader"):
            template_type = "dataloader"
        elif custom.get("custom_metric"):
            template_type = "metric"
        if template_type is not None:
            template_path = DatasetAPIInterface.generate_template(dataloader_path, template_type)
        return template_path

    @staticmethod
    def generate_template(dataloader_path: str, template_type: str) -> str:
        """Generate code templates."""
        generated_template_path = os.path.join(dataloader_path, "code_template.py")
        path_to_templates = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "utils",
            "templates",
            f"{template_type}_template.txt",
        )
        os.makedirs(os.path.dirname(generated_template_path))
        copy_file(path_to_templates, generated_template_path)
        return generated_template_path

    @staticmethod
    def dataloader_path(
        project_name: str,
        project_id: int,
        dataset_name: str,
        dataset_id: int,
    ) -> str:
        """Get path for dataset templates."""
        normalized_project_name = normalize_string(project_name)
        project_location = os.path.join(
            WORKSPACE_LOCATION,
            f"{normalized_project_name}_{project_id}",
        )
        normalized_dataset_name = normalize_string(dataset_name)
        return os.path.join(
            project_location,
            "custom_datasets",
            f"{normalized_dataset_name}_{dataset_id}",
        )

    @staticmethod
    def parse_dataset_data(data: dict) -> DatasetAddParamsInterface:
        """Parse input data for dataset."""
        dataset_parameters = DatasetAddParamsInterface()
        try:
            dataset_parameters.project_id = int(data.get("project_id", None))
            dataset_parameters.dataset_name = str(data.get("name", None))
            dataset_parameters.dataset_type = str(data.get("dataloader", {}).get("name", None))
            parameters = data.get("dataloader", {}).get("params", {})
            dataset_parameters.filter = data.get("dataloader", {}).get("filter", None)
            dataset_path = str(data.get("dataset_path", None))
            metric_name = str(data.get("metric", None))
            metric_param = data.get("metric_param", None)
        except ValueError:
            raise ClientErrorException("Could not parse value")
        except TypeError:
            raise ClientErrorException("Could not find required parameter.")

        if isinstance(parameters, dict) and "root" in parameters.keys():
            parameters.update({"root": dataset_path})
        dataset_parameters.parameters = ConfigurationParser().parse_dataloader(
            {
                "params": parameters,
            },
        )["params"]
        dataset_parameters.transforms = data.get("transform", [])
        dataset_parameters.metric = {"name": metric_name, "param": metric_param}

        return dataset_parameters

    @staticmethod
    def get_predefined_dataset(data: dict) -> dict:
        """Get predefined dataset for specified configuration."""
        required_keys = ["framework", "domain", "domain_flavour"]
        if not all(key in data.keys() for key in required_keys):
            raise ClientErrorException(
                f"Could not find required parameter. Required keys are {required_keys}",
            )
        try:
            framework = str(data.get("framework", None))
            domain = str(data.get("domain", None))
            domain_flavour = str(data.get("domain_flavour", None))
        except ValueError:
            raise ClientErrorException("Could not parse value")
        except TypeError:
            raise ClientErrorException("Could not find required parameter.")

        names_mapper = NamesMapper(MappingDirection.ToCore)
        framework = names_mapper.map_name("framework", framework)
        domain = names_mapper.map_name("domain", domain)
        domain_flavour = names_mapper.map_name("domain_flavour", domain_flavour)

        predefined_config_path = get_predefined_config_path(framework, domain, domain_flavour)

        config = Config()
        config.load(predefined_config_path)

        predefined_data: dict = {
            "transform": {},
            "dataloader": {},
            "metric": {},
            "metric_param": {},
        }
        if (
            config.quantization
            and config.quantization.calibration
            and config.quantization.calibration.dataloader
        ):
            dataloader_config: DataloaderConfig = config.quantization.calibration.dataloader
            if dataloader_config.dataset is not None:
                predefined_data.update(
                    {
                        "dataloader": DatasetAPIInterface.prepare_predefined_dataloader(
                            dataloader_config.dataset,
                        ),
                    },
                )
            if dataloader_config.transform is not None:
                predefined_data.update(
                    {
                        "transform": DatasetAPIInterface.prepare_predefined_transform(
                            dataloader_config.transform,
                        ),
                    },
                )

        if config.evaluation and config.evaluation.accuracy and config.evaluation.accuracy.metric:
            predefined_data.update(
                {
                    "metric": config.evaluation.accuracy.metric.name,
                    "metric_param": config.evaluation.accuracy.metric.param,
                },
            )
        return predefined_data

    @staticmethod
    def prepare_predefined_dataloader(
        dataloader_data: DatasetConfig,
    ) -> dict:
        """Prepare predefined transform data."""
        parameters = []
        for param_name, param_value in dataloader_data.params.items():
            parameters.append(
                {
                    "name": param_name,
                    "value": param_value,
                },
            )

        return {
            "name": dataloader_data.name,
            "params": parameters,
        }

    @staticmethod
    def prepare_predefined_transform(
        transforms_data: OrderedDict[str, TransformConfig],
    ) -> List[dict]:
        """Prepare predefined transform data."""
        transforms: List[dict] = []

        for _, transform in transforms_data.items():
            parameters = []
            for param_name, param_value in transform.parameters.items():
                parameters.append(
                    {
                        "name": param_name,
                        "value": param_value,
                    },
                )
            transforms.append(
                {
                    "name": transform.name,
                    "params": parameters,
                },
            )
        return transforms


class OptimizationAPIInterface:
    """Interface for queries connected with optimizations."""

    @staticmethod
    def delete_optimization(data: dict) -> dict:
        """Delete optimization from database and clean workspace."""
        try:
            optimization_id: int = int(data.get("id", None))
            optimization_name: str = str(data.get("name", None))
        except ValueError:
            raise ClientErrorException("Could not parse value.")
        except TypeError:
            raise ClientErrorException("Missing optimization id or optimization name.")
        with Session.begin() as db_session:
            optimization_details = Optimization.details(db_session, optimization_id)
            project_id = optimization_details["project_id"]
            project_details = ProjectAPIInterface.get_project_details({"id": project_id})
            removed_optimization_id = Optimization.delete_optimization(
                db_session=db_session,
                optimization_id=optimization_id,
                optimization_name=optimization_name,
            )

        if removed_optimization_id is not None:
            try:
                project_id = project_details["id"]
                normalized_project_name = normalize_string(project_details["name"])
                normalized_optimization_name = normalize_string(optimization_name)
                optimization_location = os.path.join(
                    WORKSPACE_LOCATION,
                    f"{normalized_project_name}_{project_id}",
                    "optimizations",
                    f"{normalized_optimization_name}_{optimization_id}",
                )
                shutil.rmtree(optimization_location, ignore_errors=True)
            except Exception:
                log.debug("Could not find optimization directory.")

        return {"id": removed_optimization_id}

    @staticmethod
    def get_optimization_details(data: dict) -> dict:
        """Parse input data and get optimization details."""
        try:
            optimization_id: int = int(data.get("id", None))
        except ValueError:
            raise ClientErrorException("Incorrect optimization id.")
        except TypeError:
            raise ClientErrorException("Could not find optimization id.")

        with Session.begin() as db_session:
            optimization_details = Optimization.details(
                db_session,
                optimization_id,
            )
        return optimization_details

    @staticmethod
    def list_optimizations(data: dict) -> dict:
        """List optimizations assigned to project."""
        try:
            project_id = int(data.get("project_id", None))
        except ValueError:
            raise ClientErrorException("Incorrect project id.")
        except TypeError:
            raise ClientErrorException("Could not find project id.")

        with Session.begin() as db_session:
            optimizations_list = Optimization.list(
                db_session,
                project_id,
            )
        return optimizations_list

    @staticmethod
    def update_optimization_status(data: dict) -> dict:
        """Update optimization status."""
        try:
            optimization_id: int = int(data.get("id", None))
        except ValueError:
            raise ClientErrorException("Incorrect optimization id.")
        except TypeError:
            raise ClientErrorException("Could not find optimization id.")

        try:
            status: ExecutionStatus = ExecutionStatus(data.get("status", None))
        except ValueError as err:
            raise ClientErrorException(err)

        with Session.begin() as db_session:
            response_data = Optimization.update_status(
                db_session,
                optimization_id,
                status,
            )
        return response_data

    @staticmethod
    def update_optimized_model(data: dict) -> dict:
        """Update optimized model."""
        try:
            optimization_id: int = int(data.get("id", None))
        except ValueError:
            raise ClientErrorException("Incorrect optimization id.")
        except TypeError:
            raise ClientErrorException("Could not find optimization id.")

        try:
            optimized_model_id = int(data.get("optimized_model_id", None))
        except ValueError as err:
            raise ClientErrorException(err)

        with Session.begin() as db_session:
            response_data = Optimization.update_optimized_model(
                db_session,
                optimization_id,
                optimized_model_id,
            )
        return response_data

    @staticmethod
    def update_optimization_duration(data: dict) -> dict:
        """Update duration of optimization."""
        try:
            optimization_id: int = int(data.get("id", None))
        except ValueError:
            raise ClientErrorException("Incorrect optimization id.")
        except TypeError:
            raise ClientErrorException("Could not find optimization id.")

        try:
            duration = int(data.get("duration", None))
        except ValueError as err:
            raise ClientErrorException(err)

        with Session.begin() as db_session:
            response_data = Optimization.update_duration(
                db_session,
                optimization_id,
                duration,
            )
        return response_data

    @staticmethod
    def update_paths(data: dict) -> dict:
        """Update config path and output log path."""
        response = {}
        try:
            optimization_id: int = int(data.get("id", None))
        except ValueError:
            raise ClientErrorException("Incorrect optimization id.")
        except TypeError:
            raise ClientErrorException("Could not find optimization id.")

        config_path: Optional[str] = data.get("config_path")
        log_path: Optional[str] = data.get("log_path")

        with Session.begin() as db_session:
            config_path_response = Optimization.update_config_path(
                db_session=db_session,
                optimization_id=optimization_id,
                path=config_path,
            )
            response.update(config_path_response)
            log_path_response = Optimization.update_log_path(
                db_session=db_session,
                optimization_id=optimization_id,
                path=log_path,
            )
            response.update(log_path_response)
        return response

    @staticmethod
    def update_execution_command(data: dict) -> dict:
        """Update optimization execution command."""
        try:
            optimization_id: int = int(data.get("id", None))
        except ValueError:
            raise ClientErrorException("Incorrect optimization id.")
        except TypeError:
            raise ClientErrorException("Could not find optimization id.")

        execution_command: Optional[Union[str, List[str]]] = data.get("execution_command")
        if isinstance(execution_command, list):
            execution_command = " ".join(map(str, execution_command))
        with Session.begin() as db_session:
            response_data = Optimization.update_execution_command(
                db_session=db_session,
                optimization_id=optimization_id,
                execution_command=execution_command,
            )
        return response_data

    @staticmethod
    def pin_accuracy_benchmark(data: dict) -> dict:
        """Pin accuracy benchmark to optimization."""
        try:
            optimization_id: int = int(data.get("optimization_id", None))
        except ValueError:
            raise ClientErrorException("Incorrect optimization id.")
        except TypeError:
            raise ClientErrorException("Could not find optimization id.")

        try:
            benchmark_id: int = int(data.get("benchmark_id", None))
        except ValueError:
            raise ClientErrorException("Incorrect benchmark id.")
        except TypeError:
            raise ClientErrorException("Could not find benchmark id.")

        with Session.begin() as db_session:
            response_data = Optimization.pin_accuracy_benchmark(
                db_session=db_session,
                optimization_id=optimization_id,
                benchmark_id=benchmark_id,
            )
        return response_data

    @staticmethod
    def pin_performance_benchmark(data: dict) -> dict:
        """Pin performance benchmark to optimization."""
        try:
            optimization_id: int = int(data.get("optimization_id", None))
        except ValueError:
            raise ClientErrorException("Incorrect optimization id.")
        except TypeError:
            raise ClientErrorException("Could not find optimization id.")

        try:
            benchmark_id: int = int(data.get("benchmark_id", None))
        except ValueError:
            raise ClientErrorException("Incorrect benchmark id.")
        except TypeError:
            raise ClientErrorException("Could not find benchmark id.")

        with Session.begin() as db_session:
            response_data = Optimization.pin_performance_benchmark(
                db_session=db_session,
                optimization_id=optimization_id,
                benchmark_id=benchmark_id,
            )
        return response_data

    @staticmethod
    def add_optimization(data: dict) -> dict:
        """Add optimization to database."""
        parser = ConfigurationParser()
        parsed_input_data = parser.parse(data)
        parsed_optimization_data: OptimizationAddParamsInterface = (
            OptimizationAPIInterface.parse_optimization_data(
                parsed_input_data,
            )
        )
        with Session.begin() as db_session:
            quantization_id = OptimizationType.get_optimization_type_id(
                db_session,
                OptimizationTypes.QUANTIZATION.value,
            )

            add_optimization_method = OptimizationAPIInterface.add_standard_optimization
            if parsed_optimization_data.optimization_type_id == quantization_id:
                add_optimization_method = OptimizationAPIInterface.add_quantization_optimization

            optimization_id = add_optimization_method(db_session, parsed_optimization_data)
        return {
            "optimization_id": optimization_id,
        }

    @staticmethod
    def add_quantization_optimization(
        db_session: session.Session,
        optimization_data: OptimizationAddParamsInterface,
    ) -> int:
        """Add quantization optimization to database."""
        tuning_details = optimization_data.tuning_details
        tuning_details_id = TuningDetails.add(
            db_session=db_session,
            strategy=tuning_details.strategy,
            accuracy_criterion_type=tuning_details.accuracy_criterion.type,
            accuracy_criterion_threshold=tuning_details.accuracy_criterion.threshold,
            objective=tuning_details.objective,
            exit_policy=tuning_details.exit_policy,
            random_seed=tuning_details.random_seed,
        )
        optimization_id = Optimization.add(
            db_session=db_session,
            project_id=optimization_data.project_id,
            name=optimization_data.name,
            precision_id=optimization_data.precision_id,
            optimization_type_id=optimization_data.optimization_type_id,
            dataset_id=optimization_data.dataset_id,
            batch_size=optimization_data.batch_size,
            sampling_size=optimization_data.sampling_size,
            tuning_details_id=tuning_details_id,
            diagnosis_config=optimization_data.diagnosis_config,
        )
        return optimization_id

    @staticmethod
    def add_standard_optimization(
        db_session: session.Session,
        optimization_data: OptimizationAddParamsInterface,
    ) -> int:
        """Add optimization to database."""
        optimization_id = Optimization.add(
            db_session=db_session,
            project_id=optimization_data.project_id,
            name=optimization_data.name,
            precision_id=optimization_data.precision_id,
            optimization_type_id=optimization_data.optimization_type_id,
            dataset_id=optimization_data.dataset_id,
            batch_size=optimization_data.batch_size,
            sampling_size=optimization_data.sampling_size,
        )
        return optimization_id

    @staticmethod
    def add_tuning_history(
        optimization_id: int,
        tuning_history: dict,
    ) -> int:
        """Add tuning history to database."""
        tuning_data: TuningHistoryInterface = OptimizationAPIInterface.parse_tuning_history(
            tuning_history,
        )

        history: List[dict] = [
            history_item.serialize() for history_item in tuning_data.history  # type: ignore
        ]
        with Session.begin() as db_session:
            optimization = Optimization.details(
                db_session,
                optimization_id,
            )
            tuning_details_id = optimization.get("tuning_details", {}).get("id", None)
            tuning_history_id = TuningHistory.add(
                db_session=db_session,
                minimal_accuracy=tuning_data.minimal_accuracy,
                baseline_accuracy=tuning_data.baseline_accuracy,
                baseline_performance=tuning_data.baseline_performance,
                last_tune_accuracy=tuning_data.last_tune_accuracy,
                last_tune_performance=tuning_data.last_tune_performance,
                best_tune_accuracy=tuning_data.best_tune_accuracy,
                best_tune_performance=tuning_data.best_tune_performance,
                history=history,
            )

            TuningDetails.update_tuning_history(db_session, tuning_details_id, tuning_history_id)
        return tuning_history_id

    @staticmethod
    def parse_tuning_history(tuning_history: dict) -> TuningHistoryInterface:
        """Parse input data for tuning history."""
        tuning_data: TuningHistoryInterface = TuningHistoryInterface()
        try:
            tuning_data.minimal_accuracy = tuning_history.get("minimal_accuracy", None)
            tuning_data.baseline_accuracy = tuning_history.get("baseline_accuracy", None)
            tuning_data.baseline_performance = tuning_history.get("baseline_performance", None)
            tuning_data.last_tune_accuracy = tuning_history.get("last_tune_accuracy", None)
            tuning_data.last_tune_performance = tuning_history.get("last_tune_performance", None)
            tuning_data.best_tune_accuracy = tuning_history.get("best_tune_accuracy", None)
            tuning_data.best_tune_performance = tuning_history.get("best_tune_performance", None)

            history = tuning_history.get("history", [])
            tuning_data.history = []
            for history_item in history:
                parsed_history_item = TuningHistoryItemInterface()
                parsed_history_item.accuracy = history_item.get("accuracy", None)
                parsed_history_item.performance = history_item.get("performance", None)
                tuning_data.history.append(parsed_history_item)
        except ValueError:
            raise ClientErrorException("Could not parse value")
        except TypeError:
            raise ClientErrorException("Could not find required parameter.")

        return tuning_data

    @staticmethod
    def parse_optimization_data(data: dict) -> OptimizationAddParamsInterface:
        """Parse input data for optimization."""
        optimization_data = OptimizationAddParamsInterface()
        try:
            optimization_data.project_id = int(data.get("project_id", None))
            optimization_data.name = str(data.get("name", None))
            optimization_data.precision_id = int(data.get("precision_id", None))
            optimization_data.optimization_type_id = int(data.get("optimization_type_id", None))
            optimization_data.dataset_id = int(data.get("dataset_id", None))
            optimization_data.tuning_details = TuningDetailsInterface(data)
        except ValueError:
            raise ClientErrorException("Could not parse value")
        except TypeError:
            raise ClientErrorException("Could not find required parameter.")

        optimization_data.batch_size = int(data.get("batch_size", 100))
        optimization_data.sampling_size = int(data.get("sampling_size", 100))
        optimization_data.diagnosis_config = dict(data.get("diagnosis_config", {}))

        return optimization_data

    @staticmethod
    def clean_status(status_to_clean: ExecutionStatus) -> dict:
        """Clean specified optimization status."""
        with Session.begin() as db_session:
            response = Optimization.clean_status(
                db_session=db_session,
                status_to_clean=status_to_clean,
            )
        return response


class BenchmarkAPIInterface:
    """Interface for queries connected with benchmark."""

    @staticmethod
    def delete_benchmark(data: dict) -> dict:
        """Delete benchmark from database and clean workspace."""
        try:
            benchmark_id: int = int(data.get("id", None))
            benchmark_name: str = str(data.get("name", None))
        except ValueError:
            raise ClientErrorException("Could not parse value.")
        except TypeError:
            raise ClientErrorException("Missing project id or project name.")
        with Session.begin() as db_session:
            benchmark_details = Benchmark.details(db_session, benchmark_id)
            project_id = benchmark_details["project_id"]
            project_details = ProjectAPIInterface.get_project_details({"id": project_id})
            removed_benchmark_id = Benchmark.delete_benchmark(
                db_session=db_session,
                benchmark_id=benchmark_id,
                benchmark_name=benchmark_name,
            )

        if removed_benchmark_id is not None:
            try:
                model_id = benchmark_details["model"]["id"]
                normalized_project_name = normalize_string(project_details["name"])
                normalized_benchmark_name = normalize_string(benchmark_name)
                normalized_model_name = normalize_string(benchmark_details["model"]["name"])
                benchmark_location = os.path.join(
                    WORKSPACE_LOCATION,
                    f"{normalized_project_name}_{project_id}",
                    "models",
                    f"{normalized_model_name}_{model_id}",
                    "benchmarks",
                    f"{normalized_benchmark_name}_{benchmark_id}",
                )
                shutil.rmtree(benchmark_location, ignore_errors=True)
            except Exception:
                log.debug("Could not find benchmark directory.")

        return {"id": removed_benchmark_id}

    @staticmethod
    def get_benchmark_details(data: dict) -> dict:
        """Parse input data and get benchmark details."""
        try:
            benchmark_id: int = int(data.get("id", None))
        except ValueError:
            raise ClientErrorException("Incorrect benchmark id.")
        except TypeError:
            raise ClientErrorException("Could not find benchmark id.")

        with Session.begin() as db_session:
            benchmark_details = Benchmark.details(
                db_session,
                benchmark_id,
            )
        return benchmark_details

    @staticmethod
    def list_benchmarks(data: dict) -> dict:
        """List benchmarks assigned to project."""
        try:
            project_id: int = int(data.get("project_id", None))
        except ValueError:
            raise ClientErrorException("Incorrect project id.")
        except TypeError:
            raise ClientErrorException("Could not find project id.")

        with Session.begin() as db_session:
            benchmarks_list = Benchmark.list(
                db_session,
                project_id,
            )
        return benchmarks_list

    @staticmethod
    def update_benchmark_accuracy(data: dict) -> dict:
        """Update benchmark accuracy."""
        try:
            benchmark_id: int = int(data.get("id", None))
        except ValueError:
            raise ClientErrorException("Incorrect benchmark id.")
        except TypeError:
            raise ClientErrorException("Could not find benchmark id.")

        try:
            accuracy: float = float(data.get("status", None))
        except ValueError as err:
            raise ClientErrorException(err)

        with Session.begin() as db_session:
            response_data = BenchmarkResult.update_accuracy(
                db_session,
                benchmark_id,
                accuracy,
            )
        return response_data

    @staticmethod
    def update_benchmark_performance(data: dict) -> dict:
        """Update benchmark performance."""
        try:
            benchmark_id: int = int(data.get("id", None))
        except ValueError:
            raise ClientErrorException("Incorrect benchmark id.")
        except TypeError:
            raise ClientErrorException("Could not find benchmark id.")

        try:
            performance: float = float(data.get("status", None))
        except ValueError as err:
            raise ClientErrorException(err)

        with Session.begin() as db_session:
            response_data = BenchmarkResult.update_performance(
                db_session,
                benchmark_id,
                performance,
            )
        return response_data

    @staticmethod
    def update_benchmark_status(data: dict) -> dict:
        """Update benchmark status."""
        try:
            benchmark_id: int = int(data.get("id", None))
        except ValueError:
            raise ClientErrorException("Incorrect benchmark id.")
        except TypeError:
            raise ClientErrorException("Could not find benchmark id.")

        try:
            status: ExecutionStatus = ExecutionStatus(data.get("status", None))
        except ValueError as err:
            raise ClientErrorException(err)

        with Session.begin() as db_session:
            response_data = Benchmark.update_status(
                db_session,
                benchmark_id,
                status,
            )
        return response_data

    @staticmethod
    def update_benchmark_duration(data: dict) -> dict:
        """Update duration of benchmark."""
        try:
            benchmark_id: int = int(data.get("id", None))
        except ValueError:
            raise ClientErrorException("Incorrect benchmark id.")
        except TypeError:
            raise ClientErrorException("Could not find benchmark id.")

        try:
            duration = int(data.get("duration", None))
        except ValueError as err:
            raise ClientErrorException(err)

        with Session.begin() as db_session:
            response_data = Benchmark.update_duration(
                db_session,
                benchmark_id,
                duration,
            )
        return response_data

    @staticmethod
    def update_paths(data: dict) -> dict:
        """Update config path and output log path."""
        response = {}
        try:
            benchmark_id: int = int(data.get("id", None))
        except ValueError:
            raise ClientErrorException("Incorrect benchmark id.")
        except TypeError:
            raise ClientErrorException("Could not find benchmark id.")

        config_path: Optional[str] = data.get("config_path")
        log_path: Optional[str] = data.get("log_path")

        with Session.begin() as db_session:
            config_path_response = Benchmark.update_config_path(
                db_session=db_session,
                benchmark_id=benchmark_id,
                path=config_path,
            )
            response.update(config_path_response)
            log_path_response = Benchmark.update_log_path(
                db_session=db_session,
                benchmark_id=benchmark_id,
                path=log_path,
            )
            response.update(log_path_response)
        return response

    @staticmethod
    def update_execution_command(data: dict) -> dict:
        """Update benchmark execution command."""
        try:
            benchmark_id: int = int(data.get("id", None))
        except ValueError:
            raise ClientErrorException("Incorrect benchmark id.")
        except TypeError:
            raise ClientErrorException("Could not find benchmark id.")

        execution_command: Optional[Union[str, List[str]]] = data.get("execution_command")
        if isinstance(execution_command, list):
            execution_command = " ".join(map(str, execution_command))
        with Session.begin() as db_session:
            response_data = Benchmark.update_execution_command(
                db_session=db_session,
                benchmark_id=benchmark_id,
                execution_command=execution_command,
            )
        return response_data

    @staticmethod
    def add_benchmark(data: dict) -> dict:
        """Add benchmark to database."""
        parser = ConfigurationParser()
        parsed_data = parser.parse(data)

        benchmark_params: BenchmarkAddParamsInterface = BenchmarkAPIInterface.parse_benchmark_data(
            parsed_data,
        )

        with Session.begin() as db_session:
            benchmark_id = Benchmark.add(
                db_session=db_session,
                project_id=benchmark_params.project_id,
                name=benchmark_params.name,
                model_id=benchmark_params.model_id,
                dataset_id=benchmark_params.dataset_id,
                mode=benchmark_params.mode,
                batch_size=benchmark_params.batch_size,
                iterations=benchmark_params.iterations,
                number_of_instance=benchmark_params.number_of_instance,
                cores_per_instance=benchmark_params.cores_per_instance,
                warmup_iterations=benchmark_params.warmup_iterations,
                execution_command=benchmark_params.command_line,
            )
        return {
            "benchmark_id": benchmark_id,
        }

    @staticmethod
    def add_result(data: dict) -> None:
        """Add benchmark result to database."""
        try:
            benchmark_id: int = int(data.get("benchmark_id", None))
        except ValueError:
            raise ClientErrorException("Incorrect benchmark id.")
        except TypeError:
            raise ClientErrorException("Could not find benchmark id.")

        accuracy: Optional[float] = None
        performance: Optional[float] = None

        try:
            accuracy = float(data["accuracy"])
        except Exception:
            pass

        try:
            performance = float(data["performance"])
        except Exception:
            pass
        with Session.begin() as db_session:
            BenchmarkResult.add(
                db_session=db_session,
                benchmark_id=benchmark_id,
                accuracy=accuracy,
                performance=performance,
            )

    @staticmethod
    def parse_benchmark_data(data: dict) -> BenchmarkAddParamsInterface:
        """Parse input data for benchmark."""
        benchmark_data = BenchmarkAddParamsInterface()
        try:
            benchmark_data.project_id = int(data.get("project_id", None))
            benchmark_data.name = str(data.get("name", None))
            benchmark_data.model_id = int(data.get("model_id", None))
            benchmark_data.dataset_id = int(data.get("dataset_id", None))
            benchmark_data.mode = str(data.get("mode", Benchmarks.PERF))
            benchmark_data.batch_size = int(data.get("batch_size", 100))
            benchmark_data.iterations = int(data.get("iterations", -1))
            benchmark_data.cores_per_instance = int(data.get("cores_per_instance", 4))
            benchmark_data.warmup_iterations = int(data.get("warmup_iterations", 10))
            benchmark_data.command_line = str(data.get("command_line"))
        except ValueError:
            raise ClientErrorException("Could not parse value")
        except TypeError:
            raise ClientErrorException("Could not find required parameter.")

        try:
            benchmark_data.number_of_instance = int(data.get("number_of_instance", None))
        except TypeError:
            from neural_compressor.ux.utils.hw_info import HWInfo

            hw_info = HWInfo()
            benchmark_data.number_of_instance = (
                hw_info.cores_per_socket // benchmark_data.cores_per_instance
            )

        return benchmark_data

    @staticmethod
    def clean_status(status_to_clean: ExecutionStatus) -> dict:
        """Clean specified optimization status."""
        with Session.begin() as db_session:
            response = Benchmark.clean_status(
                db_session=db_session,
                status_to_clean=status_to_clean,
            )
        return response


class ProfilingAPIInterface:
    """Interface for queries connected with profiling."""

    @staticmethod
    def get_profiling_details(data: dict) -> dict:
        """Parse input data and get profiling details."""
        try:
            profiling_id: int = int(data.get("id", None))
        except ValueError:
            raise ClientErrorException("Incorrect profiling id.")
        except TypeError:
            raise ClientErrorException("Could not find profiling id.")

        with Session.begin() as db_session:
            profiling_details = Profiling.details(
                db_session,
                profiling_id,
            )
        return profiling_details

    @staticmethod
    def delete_profiling(data: dict) -> dict:
        """Delete profiling from database and clean workspace."""
        try:
            profiling_id: int = int(data.get("id", None))
            profiling_name: str = str(data.get("name", None))
        except ValueError:
            raise ClientErrorException("Could not parse value.")
        except TypeError:
            raise ClientErrorException("Missing project id or project name.")
        with Session.begin() as db_session:
            profiling_details = Profiling.details(db_session, profiling_id)
            project_id = profiling_details["project_id"]
            project_details = ProjectAPIInterface.get_project_details({"id": project_id})
            removed_profiling_id = Profiling.delete_profiling(
                db_session=db_session,
                profiling_id=profiling_id,
                profiling_name=profiling_name,
            )

        if removed_profiling_id is not None:
            try:
                model_id = profiling_details["model"]["id"]
                normalized_project_name = normalize_string(project_details["name"])
                normalized_profiling_name = normalize_string(profiling_name)
                normalized_model_name = normalize_string(profiling_details["model"]["name"])
                profiling_location = os.path.join(
                    WORKSPACE_LOCATION,
                    f"{normalized_project_name}_{project_id}",
                    "models",
                    f"{normalized_model_name}_{model_id}",
                    "profilings",
                    f"{normalized_profiling_name}_{profiling_id}",
                )
                shutil.rmtree(profiling_location, ignore_errors=True)
            except Exception:
                log.debug("Could not find profiling directory.")

        return {"id": removed_profiling_id}

    @staticmethod
    def list_profilings(data: dict) -> dict:
        """List profilings assigned to project."""
        try:
            project_id: int = int(data.get("project_id", None))
        except ValueError:
            raise ClientErrorException("Incorrect project id.")
        except TypeError:
            raise ClientErrorException("Could not find project id.")

        with Session.begin() as db_session:
            profilings_list = Profiling.list(
                db_session,
                project_id,
            )
        return profilings_list

    @staticmethod
    def update_profiling_status(data: dict) -> dict:
        """Update profiling status."""
        try:
            profiling_id: int = int(data.get("id", None))
        except ValueError:
            raise ClientErrorException("Incorrect profiling id.")
        except TypeError:
            raise ClientErrorException("Could not find profiling id.")

        try:
            status: ExecutionStatus = ExecutionStatus(data.get("status", None))
        except ValueError as err:
            raise ClientErrorException(err)

        with Session.begin() as db_session:
            response_data = Profiling.update_status(
                db_session,
                profiling_id,
                status,
            )
        return response_data

    @staticmethod
    def update_profiling_duration(data: dict) -> dict:
        """Update duration of profiling."""
        try:
            profiling_id: int = int(data.get("id", None))
        except ValueError:
            raise ClientErrorException("Incorrect profiling id.")
        except TypeError:
            raise ClientErrorException("Could not find profiling id.")

        try:
            duration = int(data.get("duration", None))
        except ValueError as err:
            raise ClientErrorException(err)

        with Session.begin() as db_session:
            response_data = Profiling.update_duration(
                db_session,
                profiling_id,
                duration,
            )
        return response_data

    @staticmethod
    def update_log_path(data: dict) -> dict:
        """Update config path and output log path."""
        try:
            profiling_id: int = int(data.get("id", None))
        except ValueError:
            raise ClientErrorException("Incorrect profiling id.")
        except TypeError:
            raise ClientErrorException("Could not find profiling id.")

        log_path: Optional[str] = data.get("log_path")

        with Session.begin() as db_session:
            response = Profiling.update_log_path(
                db_session=db_session,
                profiling_id=profiling_id,
                path=log_path,
            )

        return response

    @staticmethod
    def update_execution_command(data: dict) -> dict:
        """Update profiling execution command."""
        try:
            profiling_id: int = int(data.get("id", None))
        except ValueError:
            raise ClientErrorException("Incorrect profiling id.")
        except TypeError:
            raise ClientErrorException("Could not find profiling id.")

        execution_command: Optional[Union[str, List[str]]] = data.get("execution_command")
        if isinstance(execution_command, list):
            execution_command = " ".join(map(str, execution_command))
        with Session.begin() as db_session:
            response_data = Profiling.update_execution_command(
                db_session=db_session,
                profiling_id=profiling_id,
                execution_command=execution_command,
            )
        return response_data

    @staticmethod
    def add_profiling(data: dict) -> dict:
        """Add profiling to database."""
        parser = ConfigurationParser()
        parsed_data = parser.parse(data)

        profiling_params: ProfilingAddParamsInterface = ProfilingAPIInterface.parse_profiling_data(
            parsed_data,
        )

        with Session.begin() as db_session:
            profiling_id = Profiling.add(
                db_session=db_session,
                project_id=profiling_params.project_id,
                name=profiling_params.name,
                model_id=profiling_params.model_id,
                dataset_id=profiling_params.dataset_id,
                num_threads=profiling_params.num_threads,
            )
        return {
            "profiling_id": profiling_id,
        }

    @staticmethod
    def add_result(profiling_id: int, data: dict) -> None:
        """Add profiling result to database."""
        profiling_result_data: ProfilingResultAddParamsInterface = (
            ProfilingAPIInterface.parse_profiling_result_data(
                data,
            )
        )
        profiling_result_data.profiling_id = profiling_id

        with Session.begin() as db_session:
            ProfilingResult.add(
                db_session=db_session,
                profiling_id=profiling_result_data.profiling_id,
                node_name=profiling_result_data.node_name,
                total_execution_time=profiling_result_data.total_execution_time,
                accelerator_execution_time=profiling_result_data.accelerator_execution_time,
                cpu_execution_time=profiling_result_data.cpu_execution_time,
                op_run=profiling_result_data.op_run,
                op_defined=profiling_result_data.op_defined,
            )

    @staticmethod
    def bulk_add_results(profiling_id: int, results: List[dict]) -> None:
        """Bulk add profiling results to database."""
        profiling_results: List[ProfilingResultAddParamsInterface] = []
        for result in results:
            parsed_result = ProfilingAPIInterface.parse_profiling_result_data(result)
            parsed_result.profiling_id = profiling_id
            profiling_results.append(parsed_result)

        with Session.begin() as db_session:
            ProfilingResult.delete_results(
                db_session=db_session,
                profiling_id=profiling_id,
            )
            ProfilingResult.bulk_add(
                db_session=db_session,
                profiling_id=profiling_id,
                results=profiling_results,
            )

    @staticmethod
    def parse_profiling_data(data: dict) -> ProfilingAddParamsInterface:
        """Parse input data for profiling."""
        profiling_data = ProfilingAddParamsInterface()
        try:
            profiling_data.project_id = int(data.get("project_id", None))
            profiling_data.name = str(data.get("name", None))
            profiling_data.model_id = int(data.get("model_id", None))
            profiling_data.dataset_id = int(data.get("dataset_id", None))
            profiling_data.num_threads = int(data.get("num_threads", None))
        except ValueError:
            raise ClientErrorException("Could not parse value")
        except TypeError:
            raise ClientErrorException("Could not find required parameter.")

        return profiling_data

    @staticmethod
    def parse_profiling_result_data(data: dict) -> ProfilingResultAddParamsInterface:
        """Parse input data for profiling result."""
        profiling_result_data = ProfilingResultAddParamsInterface()
        try:
            profiling_result_data.node_name = str(data.get("node_name", None))
            profiling_result_data.total_execution_time = int(
                data.get("total_execution_time", None),
            )
            profiling_result_data.accelerator_execution_time = int(
                data.get("accelerator_execution_time", None),
            )
            profiling_result_data.cpu_execution_time = int(data.get("cpu_execution_time", None))
            profiling_result_data.op_run = int(data.get("op_occurence", {}).get("run", None))
            profiling_result_data.op_defined = int(
                data.get("op_occurence", {}).get("defined", None),
            )
        except ValueError:
            raise ClientErrorException("Could not parse value")
        except TypeError:
            raise ClientErrorException("Could not find required parameter.")

        return profiling_result_data

    @staticmethod
    def clean_status(status_to_clean: ExecutionStatus) -> dict:
        """Clean specified optimization status."""
        with Session.begin() as db_session:
            response = Profiling.clean_status(
                db_session=db_session,
                status_to_clean=status_to_clean,
            )
        return response


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


class ExamplesAPIInterface:
    """Interface for queries connected with predefined models."""

    @staticmethod
    def create_project(data: dict) -> None:
        """Create new project for predefined model."""
        mq = MessageQueue()

        try:
            request_id: str = str(data["request_id"])
            mq.post_success(
                "create_example_project_start",
                {"message": "Creating project from examples.", "request_id": request_id},
            )

            framework = str(data.get("framework"))
            model = str(data.get("model"))
            domain = str(data.get("domain"))

            models_config = load_model_config()
            model_info = models_config.get(framework, {}).get(domain, {}).get(model, None)
            if model_info is None:
                raise InternalException(
                    f"Could not find information about {framework} {domain} {model} model",
                )
            try:
                mq.post_success(
                    "create_example_project_progress",
                    {"message": "Downloading example model.", "request_id": request_id},
                )
                model_path = download_model(data)
            except Exception as e:
                mq.post_error(
                    "create_example_project_finish",
                    {"message": str(e), "code": 404, "request_id": request_id},
                )
                raise

            project_name = data.get("name", None)
            if project_name is None:
                ClientErrorException("Project name not provided.")

            with Session.begin() as db_session:
                mq.post_success(
                    "create_example_project_progress",
                    {"message": "Adding project for example model.", "request_id": request_id},
                )
                project_id = Project.create_project(db_session, project_name)
                data.update({"project_id": project_id})
                data.update(
                    {
                        "model": {
                            "model_name": "Input model",
                            "path": model_path,
                            "domain": domain,
                            "shape": model_info.get("input_shape"),
                            "input_nodes": model_info.get("inputs", []),
                            "output_nodes": model_info.get("outputs", []),
                        },
                    },
                )
                mq.post_success(
                    "create_example_project_progress",
                    {"message": "Adding example model to project.", "request_id": request_id},
                )
                model_id = ProjectAPIInterface.add_model(db_session, data)
                mq.post_success(
                    "create_example_project_progress",
                    {"message": "Adding dummy dataset to project.", "request_id": request_id},
                )
                dataset_id = ProjectAPIInterface.add_dummy_dataset(
                    db_session,
                    data,
                )

                mq.post_success(
                    "create_example_project_progress",
                    {"message": "Adding optimization to project.", "request_id": request_id},
                )
                optimization_data = ExamplesAPIInterface.get_optimization_data(
                    db_session=db_session,
                    project_id=project_id,
                    dataset_id=dataset_id,
                    optimization_name=f"{model} quantization",
                    precision=Precisions.INT8.value,
                    optimization=OptimizationTypes.QUANTIZATION.value,
                )
                optimization_id = OptimizationAPIInterface.add_quantization_optimization(
                    db_session,
                    optimization_data,
                )

                mq.post_success(
                    "create_example_project_progress",
                    {"message": "Adding benchmark to project.", "request_id": request_id},
                )
                benchmark_data: BenchmarkAddParamsInterface = (
                    BenchmarkAPIInterface.parse_benchmark_data(
                        {
                            "name": f"{model} benchmark",
                            "project_id": project_id,
                            "model_id": model_id,
                            "dataset_id": dataset_id,
                            "mode": "performance",
                            "batch_size": 1,
                            "iterations": -1,
                            "warmup_iterations": 5,
                            "command_line": "",
                        },
                    )
                )
                benchmark_id = Benchmark.add(
                    db_session=db_session,
                    project_id=benchmark_data.project_id,
                    name=benchmark_data.name,
                    model_id=benchmark_data.model_id,
                    dataset_id=benchmark_data.dataset_id,
                    mode=benchmark_data.mode,
                    batch_size=benchmark_data.batch_size,
                    iterations=benchmark_data.iterations,
                    number_of_instance=benchmark_data.number_of_instance,
                    cores_per_instance=benchmark_data.cores_per_instance,
                    warmup_iterations=benchmark_data.warmup_iterations,
                    execution_command=benchmark_data.command_line,
                )
        except Exception as e:
            mq.post_failure(
                "create_example_project_finish",
                {"message": str(e), "code": 404, "request_id": request_id},
            )
        mq.post_success(
            "create_example_project_finish",
            {
                "message": "Example project has been added.",
                "request_id": request_id,
                "project_id": project_id,
                "model_id": model_id,
                "dataset_id": dataset_id,
                "optimization_id": optimization_id,
                "benchmark_id": benchmark_id,
            },
        )

    @staticmethod
    def get_optimization_data(
        db_session: session.Session,
        project_id: int,
        dataset_id: int,
        optimization_name: str,
        precision: str,
        optimization: str,
    ) -> OptimizationAddParamsInterface:
        """Get data to add optimization."""
        quantization_id = OptimizationType.get_optimization_type_id(
            db_session,
            optimization,
        )
        int8_precision_id = Precision.get_precision_id(
            db_session,
            precision,
        )

        optimization_data: OptimizationAddParamsInterface = (
            OptimizationAPIInterface.parse_optimization_data(
                {
                    "project_id": project_id,
                    "name": optimization_name,
                    "precision_id": int8_precision_id,
                    "optimization_type_id": quantization_id,
                    "dataset_id": dataset_id,
                },
            )
        )
        return optimization_data


class DiagnosisAPIInterface:
    """Interface for queries connected with diagnosis of models."""

    @staticmethod
    def get_op_list(data: dict) -> List[dict]:
        """Get OP list for model."""
        try:
            project_id: int = int(data.get("project_id", None))
            model_id: int = int(data.get("model_id", None))
        except ValueError:
            raise ClientErrorException("Incorrect parameter values.")
        except TypeError:
            raise ClientErrorException("Could not find all required parameters.")

        with Session.begin() as db_session:
            optimization_details = Optimization.get_optimization_by_project_and_model(
                db_session,
                project_id,
                model_id,
            )

        dataset_details = DatasetAPIInterface.get_dataset_details(
            {"id": optimization_details["dataset"]["id"]},
        )
        project_id = optimization_details["project_id"]
        project_details = ProjectAPIInterface.get_project_details({"id": project_id})
        optimization: OptimizationInterface = OptimizationFactory.get_optimization(
            optimization_data=optimization_details,
            project_data=project_details,
            dataset_data=dataset_details,
        )

        diagnosis = DiagnosisFactory.get_diagnosis(optimization)

        return diagnosis.get_op_list()

    @staticmethod
    def get_op_details(data: dict) -> dict:
        """Get OP details for specific OP in model."""
        try:
            project_id: int = int(data.get("project_id", None))
            model_id: int = int(data.get("model_id", None))
            op_name: str = str(data.get("op_name", None))
        except ValueError:
            raise ClientErrorException("Incorrect parameter values.")
        except TypeError:
            raise ClientErrorException("Could not find all required parameters.")

        with Session.begin() as db_session:
            optimization_details = Optimization.get_optimization_by_project_and_model(
                db_session,
                project_id,
                model_id,
            )

        dataset_details = DatasetAPIInterface.get_dataset_details(
            {"id": optimization_details["dataset"]["id"]},
        )
        project_id = optimization_details["project_id"]
        project_details = ProjectAPIInterface.get_project_details({"id": project_id})
        optimization: OptimizationInterface = OptimizationFactory.get_optimization(
            optimization_data=optimization_details,
            project_data=project_details,
            dataset_data=dataset_details,
        )

        diagnosis = DiagnosisFactory.get_diagnosis(optimization)

        op_details: Optional[OpDetails] = diagnosis.get_op_details(op_name)
        if op_details is None:
            return {}
        return op_details.serialize()

    @staticmethod
    def histogram(data: dict) -> list:
        """Get histogram of specific tensor in model."""
        try:
            project_id: int = int(data.get("project_id", None))
            model_id: int = int(data.get("model_id", None))
            op_name: str = str(data.get("op_name", None))
            histogram_type: str = str(data.get("type", None))
        except ValueError:
            raise ClientErrorException("Incorrect parameter values.")
        except TypeError:
            raise ClientErrorException("Could not find all required parameters.")

        histogram_type_map = {
            "weights": "weight",
            "activation": "activation",
        }

        parsed_histogram_type: Optional[str] = histogram_type_map.get(histogram_type, None)
        if parsed_histogram_type is None:
            raise ClientErrorException(
                f"Histogram type not supported. "
                f"Use one of following: {histogram_type_map.keys()}",
            )

        with Session.begin() as db_session:
            optimization_details = Optimization.get_optimization_by_project_and_model(
                db_session,
                project_id,
                model_id,
            )

        dataset_details = DatasetAPIInterface.get_dataset_details(
            {"id": optimization_details["dataset"]["id"]},
        )
        project_id = optimization_details["project_id"]
        project_details = ProjectAPIInterface.get_project_details({"id": project_id})
        optimization: OptimizationInterface = OptimizationFactory.get_optimization(
            optimization_data=optimization_details,
            project_data=project_details,
            dataset_data=dataset_details,
        )

        diagnosis = DiagnosisFactory.get_diagnosis(optimization)

        histogram_data = diagnosis.get_histogram_data(op_name, parsed_histogram_type)
        return histogram_data

    @staticmethod
    def generate_optimization(data: dict) -> int:
        """Parse input data and get optimization details."""
        optimization_data = DiagnosisAPIInterface.parse_optimization_data(data)

        original_optimization = OptimizationAPIInterface.get_optimization_details(
            {"id": optimization_data.optimization_id},
        )

        new_optimization_data: OptimizationAddParamsInterface = (
            OptimizationAPIInterface.parse_optimization_data(
                {
                    "project_id": optimization_data.project_id,
                    "name": optimization_data.optimization_name,
                    "precision_id": original_optimization["precision"]["id"],
                    "optimization_type_id": original_optimization["optimization_type"]["id"],
                    "dataset_id": original_optimization["dataset"]["id"],
                    "batch_size": original_optimization["batch_size"],
                    "sampling_size": original_optimization["sampling_size"],
                    "diagnosis_config": {
                        "op_wise": optimization_data.op_wise,
                        "model_wise": optimization_data.model_wise,
                    },
                },
            )
        )

        with Session.begin() as db_session:
            optimization_id = OptimizationAPIInterface.add_quantization_optimization(
                db_session=db_session,
                optimization_data=new_optimization_data,
            )

        return optimization_id

    @staticmethod
    def model_wise_params(data: dict) -> dict:
        """Get model wise parameters for specified optimization."""
        try:
            optimization_id: int = int(data.get("optimization_id", None))
        except ValueError:
            raise ClientErrorException("Incorrect optimization id.")
        except TypeError:
            raise ClientErrorException("Could not find optimization id.")
        optimization = OptimizationAPIInterface.get_optimization_details({"id": optimization_id})

        try:
            framework = optimization["optimized_model"]["framework"]["name"]
        except KeyError:
            raise ClientErrorException("Could not find framework name for specified optimization.")
        return load_model_wise_params(framework)

    @staticmethod
    def parse_optimization_data(data: dict) -> DiagnosisOptimizationParamsInterface:
        """Parse optimization parameters from diagnosis tab."""
        optimization_params = DiagnosisOptimizationParamsInterface()
        try:
            optimization_params.project_id = int(data.get("project_id", None))
            optimization_params.optimization_id = int(data.get("optimization_id", None))
            optimization_params.optimization_name = str(data.get("optimization_name", None))
            optimization_params.op_wise = DiagnosisAPIInterface.parse_op_wise_config(
                data.get("op_wise", {}),
            )
            optimization_params.model_wise = DiagnosisAPIInterface.parse_model_wise_config(
                data.get("model_wise", {}),
            )
        except ValueError:
            raise ClientErrorException("Could not parse value")
        except TypeError:
            raise ClientErrorException("Could not find required parameter.")

        if not optimization_params.op_wise and not optimization_params.model_wise:
            raise ClientErrorException("At least one of OP and Model wise setting is required.")

        return optimization_params

    @staticmethod
    def parse_op_wise_config(op_wise_params: dict) -> dict:
        """Parse OP wise configuration."""
        parsed_op_wise_config: dict = {}
        for op_name, param_types in op_wise_params.items():
            parsed_op_params = DiagnosisAPIInterface.parse_wise_parameters(param_types)
            parsed_op_wise_config.update({op_name: parsed_op_params})
        return parsed_op_wise_config

    @staticmethod
    def parse_wise_parameters(params_per_type: dict) -> dict:
        """Parse model or OP wise parameters."""
        parsed_params: dict = {}
        for param_type, params in params_per_type.items():
            parsed_params_per_type = parsed_params.get(param_type, {})
            for param_name, param_value in params.items():
                if param_type == "pattern" and param_name == "precision":
                    parsed_params = DiagnosisAPIInterface.set_op_wise_pattern_precision(
                        parsed_params,
                        param_value,
                    )
                parsed_params_per_type.update({param_name: [param_value]})
            if param_type == "pattern" or not bool(parsed_params_per_type):
                log.debug(f"Skipping adding parameters for {param_type}")
                continue
            parsed_params.update({param_type: parsed_params_per_type})
        return parsed_params

    @staticmethod
    def set_op_wise_pattern_precision(op_wise_params: dict, precision: str) -> dict:
        """Set precision from op wise pattern setting."""
        weight_config = op_wise_params.get("weight", None)
        if weight_config is None:
            weight_config = {}
        weight_config.update({"dtype": [precision]})

        activation_config = op_wise_params.get("activation", None)
        if activation_config is None:
            activation_config = {}
        activation_config.update({"dtype": [precision]})

        op_wise_params.update(
            {
                "weight": weight_config,
                "activation": activation_config,
            },
        )
        return op_wise_params

    @staticmethod
    def parse_model_wise_config(data: dict) -> dict:
        """Parse Model wise configuration."""
        return DiagnosisAPIInterface.parse_wise_parameters(data)


def set_database_version() -> None:
    """Set version_num in alembic_version table."""
    alembic_config_path = os.path.join(
        os.path.dirname(__file__),
        "alembic.ini",
    )

    alembic_scripts_location = os.path.join(
        os.path.dirname(alembic_config_path),
        "alembic",
    )
    alembic_cfg = AlembicConfig(alembic_config_path)
    alembic_cfg.set_main_option("sqlalchemy.url", db_manager.database_entrypoint)
    alembic_cfg.set_main_option("script_location", alembic_scripts_location)
    command.ensure_version(alembic_cfg)

    script = ScriptDirectory.from_config(alembic_cfg)
    revision = script.get_revision("head")
    latest_revision = revision.revision
    with db_manager.engine.connect() as conn:
        conn.execute(
            f"INSERT INTO alembic_version(version_num) VALUES ('{latest_revision}') "
            f"ON CONFLICT(version_num) DO UPDATE SET version_num='{latest_revision}';",
        )


def initialize_associations() -> None:
    """Initialize association tables in database."""
    initialize_precision_optimization_types_association()


def initialize_precision_optimization_types_association() -> None:
    """Initialize precision and optimization types association table."""
    with Session.begin() as db_session:
        # Check if there is already data in table:
        association_table_select = precision_optimization_type_association.select()
        association_table_data = db_session.execute(association_table_select)
        if len(association_table_data.all()) > 0:
            log.debug("Precision - Optimization Type association table already exists.")
            return

        optimization_types_db = OptimizationType.list(db_session)
        precisions_db = Precision.list(db_session)
        for optimization_type, precisions in precision_optimization_types.items():
            optimization_type_id = search_in_list_of_dict_for_unique_value(
                optimization_types_db["optimization_types"],
                "name",
                optimization_type.value,
            )["id"]

            for precision in precisions:
                precision_id = search_in_list_of_dict_for_unique_value(
                    precisions_db["precisions"],
                    "name",
                    precision.value,
                )["id"]
                query = precision_optimization_type_association.insert().values(
                    precision_id=precision_id,
                    optimization_type_id=optimization_type_id,
                )
                db_session.execute(query)


def search_in_list_of_dict_for_unique_value(
    list_of_dicts: List[Dict[str, Any]],
    parameter: str,
    value: Any,
) -> Dict[str, Any]:
    """Search for dictionaries with specific unique parameter value."""
    search_results: List[dict] = search_in_list_of_dict(list_of_dicts, parameter, value)
    if len(search_results) > 1:
        raise InternalException("Search result is not unique.")
    return search_results[0]


def search_in_list_of_dict(
    list_of_dicts: List[Dict[str, Any]],
    parameter: str,
    value: Any,
) -> List[Dict[str, Any]]:
    """Search for dictionaries with specific parameter value."""
    matching_entries = []
    for entry in list_of_dicts:
        attribute_value = entry.get(parameter)
        if attribute_value == value:
            matching_entries.append(entry)
    return matching_entries
