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
"""INC Bench Diagnosis API interface."""
from typing import List, Optional

from sqlalchemy.orm import sessionmaker

from neural_compressor.ux.components.db_manager.db_manager import DBManager
from neural_compressor.ux.components.db_manager.db_models.optimization import Optimization
from neural_compressor.ux.components.db_manager.db_operations.dataset_api_interface import (
    DatasetAPIInterface,
)
from neural_compressor.ux.components.db_manager.db_operations.optimization_api_interface import (
    OptimizationAPIInterface,
)
from neural_compressor.ux.components.db_manager.db_operations.project_api_interface import (
    ProjectAPIInterface,
)
from neural_compressor.ux.components.db_manager.params_interfaces import (
    DiagnosisOptimizationParamsInterface,
    OptimizationAddParamsInterface,
)
from neural_compressor.ux.components.diagnosis.factory import DiagnosisFactory
from neural_compressor.ux.components.diagnosis.op_details import OpDetails
from neural_compressor.ux.components.optimization.factory import OptimizationFactory
from neural_compressor.ux.components.optimization.optimization import (
    Optimization as OptimizationInterface,
)
from neural_compressor.ux.utils.exceptions import ClientErrorException
from neural_compressor.ux.utils.logger import log
from neural_compressor.ux.utils.utils import load_model_wise_params

db_manager = DBManager()
Session = sessionmaker(bind=db_manager.engine)


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
