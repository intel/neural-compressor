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
"""INC Bench Dataset API interface."""
import os
from shutil import copy as copy_file
from typing import Any, Dict, List, Optional, OrderedDict

from alembic import command
from alembic.config import Config as AlembicConfig
from alembic.script import ScriptDirectory
from sqlalchemy.orm import sessionmaker

from neural_compressor.ux.components.configuration_wizard.configuration_parser import (
    ConfigurationParser,
)
from neural_compressor.ux.components.db_manager.db_manager import DBManager
from neural_compressor.ux.components.db_manager.db_models.dataset import Dataset
from neural_compressor.ux.components.db_manager.db_models.optimization_type import OptimizationType
from neural_compressor.ux.components.db_manager.db_models.precision import (
    Precision,
    precision_optimization_type_association,
)
from neural_compressor.ux.components.db_manager.db_models.project import Project
from neural_compressor.ux.components.db_manager.params_interfaces import DatasetAddParamsInterface
from neural_compressor.ux.components.names_mapper.names_mapper import MappingDirection, NamesMapper
from neural_compressor.ux.utils.consts import WORKSPACE_LOCATION, precision_optimization_types
from neural_compressor.ux.utils.exceptions import ClientErrorException, InternalException
from neural_compressor.ux.utils.logger import log
from neural_compressor.ux.utils.utils import get_predefined_config_path, normalize_string
from neural_compressor.ux.utils.workload.config import Config
from neural_compressor.ux.utils.workload.dataloader import Dataloader as DataloaderConfig
from neural_compressor.ux.utils.workload.dataloader import Dataset as DatasetConfig
from neural_compressor.ux.utils.workload.dataloader import Transform as TransformConfig

db_manager = DBManager()
Session = sessionmaker(bind=db_manager.engine)


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
            os.pardir,
            os.pardir,
            os.pardir,
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
