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
"""INC Bench Optimization API interface."""
import os
import shutil
from typing import List, Optional, Union

from sqlalchemy.orm import session, sessionmaker

from neural_compressor.ux.components.configuration_wizard.configuration_parser import (
    ConfigurationParser,
)
from neural_compressor.ux.components.configuration_wizard.pruning_config_parser import (
    PruningConfigParser,
)
from neural_compressor.ux.components.db_manager.db_manager import DBManager
from neural_compressor.ux.components.db_manager.db_models.optimization import Optimization
from neural_compressor.ux.components.db_manager.db_models.optimization_type import OptimizationType
from neural_compressor.ux.components.db_manager.db_models.precision import Precision
from neural_compressor.ux.components.db_manager.db_models.pruning_details import PruningDetails
from neural_compressor.ux.components.db_manager.db_models.tuning_details import TuningDetails
from neural_compressor.ux.components.db_manager.db_models.tuning_history import TuningHistory
from neural_compressor.ux.components.db_manager.db_operations.project_api_interface import (
    ProjectAPIInterface,
)
from neural_compressor.ux.components.db_manager.params_interfaces import (
    OptimizationAddParamsInterface,
    OptimizationEditParamsInterface,
    TuningHistoryInterface,
    TuningHistoryItemInterface,
)
from neural_compressor.ux.components.jobs_management import jobs_control_queue, parse_job_id
from neural_compressor.ux.components.optimization.tune.tuning import (
    TuningDetails as TuningDetailsInterface,
)
from neural_compressor.ux.utils.consts import (
    WORKSPACE_LOCATION,
    ExecutionStatus,
    OptimizationTypes,
    Precisions,
)
from neural_compressor.ux.utils.exceptions import ClientErrorException, InternalException
from neural_compressor.ux.utils.logger import log
from neural_compressor.ux.utils.utils import (
    get_default_pruning_config_path,
    load_pruning_details_config,
    normalize_string,
)
from neural_compressor.ux.utils.workload.config import Config
from neural_compressor.ux.utils.workload.pruning import Pruning as PruningConfig

db_manager = DBManager()
Session = sessionmaker(bind=db_manager.engine)


class OptimizationAPIInterface:
    """Interface for queries connected with optimizations."""

    @staticmethod
    def delete_optimization(data: dict) -> dict:
        """Delete optimization from database and clean workspace."""
        try:
            optimization_id: int = int(data.get("id", None))
            optimization_name: str = str(data.get("name", None))
            job_id = parse_job_id("optimization", optimization_id)
            jobs_control_queue.abort_job(job_id, blocking=True)
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

        project_data = ProjectAPIInterface.get_project_details(
            {
                "id": parsed_optimization_data.project_id,
            },
        )
        supports_pruning = project_data.get("input_model", {}).get("supports_pruning", False)

        with Session.begin() as db_session:
            quantization_id = OptimizationType.get_optimization_type_id(
                db_session=db_session,
                optimization_name=OptimizationTypes.QUANTIZATION.value,
            )
            pruning_id = OptimizationType.get_optimization_type_id(
                db_session=db_session,
                optimization_name=OptimizationTypes.PRUNING.value,
            )

            add_optimization_method = OptimizationAPIInterface.add_standard_optimization
            if parsed_optimization_data.optimization_type_id == quantization_id:
                add_optimization_method = OptimizationAPIInterface.add_quantization_optimization
            if parsed_optimization_data.optimization_type_id == pruning_id:
                if not supports_pruning:
                    raise ClientErrorException("Pruning is only supported for TensorFlow models.")
                add_optimization_method = OptimizationAPIInterface.add_pruning_optimization

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
    def add_pruning_optimization(
        db_session: session.Session,
        optimization_data: OptimizationAddParamsInterface,
    ) -> int:
        """Add quantization optimization to database."""
        pruning_config_path = get_default_pruning_config_path()
        pruning_config = Config()
        pruning_config.load(pruning_config_path)

        if not isinstance(pruning_config.pruning, PruningConfig):
            raise InternalException("Could not load predefined pruning configurations.")
        optimization_data.pruning_details = pruning_config.pruning

        pruning_details_id = PruningDetails.add(
            db_session=db_session,
            pruning_details=optimization_data.pruning_details,
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
            pruning_details_id=pruning_details_id,
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
    def edit_optimization(data: dict) -> dict:
        """Edit existing optimization."""
        parser = ConfigurationParser()
        parsed_input_data = parser.parse(data)
        parsed_optimization_data: OptimizationEditParamsInterface = (
            OptimizationAPIInterface.parse_optimization_edit_data(
                parsed_input_data,
            )
        )
        response: dict = {"id": parsed_optimization_data.id}
        with Session.begin() as db_session:
            optimization = Optimization.details(
                db_session,
                parsed_optimization_data.id,
            )

            if optimization.get("status", None) is not None:
                raise ClientErrorException("Can not modify optimization that has been run.")

            if "precision_id" in data:
                optimization_types_for_precision = (
                    OptimizationType.get_optimization_type_for_precision(
                        db_session=db_session,
                        precision_id=parsed_optimization_data.precision_id,
                    )
                )

                if len(optimization_types_for_precision) != 1:
                    raise InternalException(
                        "Found multiple optimization types for given precision.",
                    )

                Optimization.update_precision(
                    db_session=db_session,
                    optimization_id=parsed_optimization_data.id,
                    precision_id=parsed_optimization_data.precision_id,
                    optimization_type_id=optimization_types_for_precision[0]["id"],
                )
                int8_precision_id = Precision.get_precision_id(
                    db_session,
                    Precisions.INT8.value,
                )
                previous_precision_id = optimization.get("precision", {}).get("id", None)
                if (
                    previous_precision_id != int8_precision_id
                    and parsed_optimization_data.precision_id == int8_precision_id
                ):
                    new_details = TuningDetailsInterface()
                    new_details_id = TuningDetails.add(
                        db_session=db_session,
                        strategy=new_details.strategy,
                        accuracy_criterion_type=new_details.accuracy_criterion.type,
                        accuracy_criterion_threshold=new_details.accuracy_criterion.threshold,
                        objective=new_details.objective,
                        exit_policy=new_details.exit_policy,
                        random_seed=new_details.random_seed,
                    )
                    Optimization.update_tuning_details(
                        db_session=db_session,
                        optimization_id=parsed_optimization_data.id,
                        tuning_details_id=new_details_id,
                    )
                if (
                    previous_precision_id == int8_precision_id
                    and parsed_optimization_data.precision_id != int8_precision_id
                ):
                    TuningDetails.delete_tuning_details(
                        db_session=db_session,
                        tuning_details_id=optimization.get("tuning_details", {}).get("id", None),
                    )
                    Optimization.update_tuning_details(
                        db_session=db_session,
                        optimization_id=parsed_optimization_data.id,
                        tuning_details_id=None,
                    )

            if "dataset_id" in data:
                Optimization.update_dataset(
                    db_session=db_session,
                    optimization_id=parsed_optimization_data.id,
                    dataset_id=parsed_optimization_data.dataset_id,
                )

            if "tuning_details" in data:
                tuning_details_response = OptimizationAPIInterface.edit_tuning_details(
                    db_session=db_session,
                    optimization=optimization,
                    parsed_optimization_data=parsed_optimization_data,
                )
                response.update(tuning_details_response)

            if "pruning_details" in data:
                pruning_details_response = OptimizationAPIInterface.edit_pruning_details(
                    db_session=db_session,
                    optimization=optimization,
                    pruning_data=data["pruning_details"],
                )
                response.update(pruning_details_response)

        return response

    @staticmethod
    def get_pruning_details(data: dict) -> dict:
        """Gat pruning details in a form of tree."""
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
        pruning_details = optimization_details.get("pruning_details", None)
        if pruning_details is None:
            raise ClientErrorException("Optimization does not have pruning details.")

        parser = PruningConfigParser()
        pruning_details_id = pruning_details["id"]
        del pruning_details["id"]
        pruning_tree = parser.generate_tree(pruning_details)
        return {
            "pruning_details": {
                "id": pruning_details_id,
                "config_tree": pruning_tree,
            },
        }

    @staticmethod
    def edit_tuning_details(
        db_session: session.Session,
        optimization: dict,
        parsed_optimization_data: OptimizationEditParamsInterface,
    ) -> dict:
        """Edit tuning details."""
        tuning_details_id = optimization.get("tuning_details", {}).get("id", None)
        if tuning_details_id is None:
            raise ClientErrorException("Could not find tuning_details for optimization.")
        tuning_details = TuningDetails.update(
            db_session=db_session,
            tuning_details_id=tuning_details_id,
            tuning_details_data=parsed_optimization_data.tuning_details,
        )

        tuning_details_response = {"tuning_details": tuning_details}

        tuning_details_response.update(
            Optimization.update_batch_size(
                db_session=db_session,
                optimization_id=parsed_optimization_data.id,
                batch_size=parsed_optimization_data.batch_size,
            ),
        )

        tuning_details_response.update(
            Optimization.update_sampling_size(
                db_session=db_session,
                optimization_id=parsed_optimization_data.id,
                sampling_size=parsed_optimization_data.sampling_size,
            ),
        )
        return tuning_details_response

    @staticmethod
    def edit_pruning_details(
        db_session: session.Session,
        optimization: dict,
        pruning_data: dict,
    ) -> dict:
        """Edit pruning details."""
        pruning_details_id = optimization.get("pruning_details", {}).get("id", None)
        if pruning_details_id is None:
            raise ClientErrorException("Could not find pruning_details for optimization.")

        pruning_details = optimization.get("pruning_details", None)

        if pruning_details is None:
            raise InternalException("Could not get pruning details from optimization.")

        pruning_config = PruningConfig(pruning_details)
        if pruning_config.train is not None:
            pruning_config.train.epoch = int(pruning_data["epoch"])
        if pruning_config.approach is not None:
            if pruning_config.approach.weight_compression is not None:
                pruning_config.approach.weight_compression.target_sparsity = float(
                    pruning_data["target_sparsity"],
                )

            if pruning_config.approach.weight_compression_pytorch is not None:
                pruning_config.approach.weight_compression_pytorch.target_sparsity = float(
                    pruning_data["target_sparsity"],
                )

        pruning_details_response = PruningDetails.update(
            db_session=db_session,
            pruning_details_id=pruning_details_id,
            pruning_details_data=pruning_config,
        )
        return pruning_details_response

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
    def parse_optimization_edit_data(data: dict) -> OptimizationEditParamsInterface:
        """Parse data for editing optimization."""
        optimization_data = OptimizationEditParamsInterface()
        try:
            optimization_data.id = int(data.get("id", None))
            optimization_data.precision_id = int(data.get("precision_id", -1))
            optimization_data.dataset_id = int(data.get("dataset_id", -1))
            optimization_data.batch_size = int(
                data.get("tuning_details", {}).get("batch_size", -1),
            )
            optimization_data.sampling_size = int(
                data.get("tuning_details", {}).get("sampling_size", -1),
            )
            objective = data.get("tuning_details", {}).get("multi_objectives", None)
            if objective is not None:
                data["tuning_details"].update({"objective": objective})

            strategy = data.get("tuning_details", {}).get("strategy", None)
            if strategy is not None:
                data["tuning_details"].update({"tuning_strategy": strategy})
            optimization_data.tuning_details = TuningDetailsInterface(
                data.get("tuning_details", {}),
            )
        except ValueError:
            raise ClientErrorException("Could not parse value")
        except TypeError:
            raise ClientErrorException("Could not find required parameter.")

        return optimization_data

    @staticmethod
    def load_pruning_details_config(data: dict) -> List[dict]:
        """Load pruning details config."""
        return load_pruning_details_config()

    @staticmethod
    def clean_status(status_to_clean: ExecutionStatus) -> dict:
        """Clean specified optimization status."""
        with Session.begin() as db_session:
            response = Optimization.clean_status(
                db_session=db_session,
                status_to_clean=status_to_clean,
            )
        return response
