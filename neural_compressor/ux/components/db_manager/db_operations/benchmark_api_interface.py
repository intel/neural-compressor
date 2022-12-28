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
"""INC Bench Benchmark API interface."""
import os
import shutil
from sqlite3 import Connection
from typing import List, Optional, Union

from sqlalchemy import event
from sqlalchemy.orm import Mapper, sessionmaker

from neural_compressor.ux.components.benchmark import Benchmarks
from neural_compressor.ux.components.configuration_wizard.configuration_parser import (
    ConfigurationParser,
)
from neural_compressor.ux.components.db_manager.db_manager import DBManager
from neural_compressor.ux.components.db_manager.db_models.benchmark import Benchmark
from neural_compressor.ux.components.db_manager.db_models.benchmark_result import BenchmarkResult
from neural_compressor.ux.components.db_manager.db_models.optimization import Optimization
from neural_compressor.ux.components.db_manager.db_operations.project_api_interface import (
    ProjectAPIInterface,
)
from neural_compressor.ux.components.db_manager.params_interfaces import (
    BenchmarkAddParamsInterface,
    BenchmarkEditParamsInterface,
)
from neural_compressor.ux.components.jobs_management import jobs_control_queue, parse_job_id
from neural_compressor.ux.utils.consts import WORKSPACE_LOCATION, ExecutionStatus
from neural_compressor.ux.utils.exceptions import ClientErrorException
from neural_compressor.ux.utils.logger import log
from neural_compressor.ux.utils.utils import normalize_string

db_manager = DBManager()
Session = sessionmaker(bind=db_manager.engine)


class BenchmarkAPIInterface:
    """Interface for queries connected with benchmark."""

    @staticmethod
    def delete_benchmark(data: dict) -> dict:
        """Delete benchmark from database and clean workspace."""
        try:
            benchmark_id: int = int(data.get("id", None))
            benchmark_name: str = str(data.get("name", None))
            job_id = parse_job_id("benchmark", benchmark_id)
            jobs_control_queue.abort_job(job_id, blocking=True)
        except ValueError:
            raise ClientErrorException("Could not parse value.")
        except TypeError:
            raise ClientErrorException("Missing project id or project name.")
        with Session.begin() as db_session:
            benchmark_details = Benchmark.details(db_session, benchmark_id)
            project_id = benchmark_details["project_id"]
            project_details = ProjectAPIInterface.get_project_details({"id": project_id})
            Optimization.unpin_benchmark(
                db_connection=db_session,
                benchmark_id=benchmark_id,
            )
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
    def edit_benchmark(data: dict) -> dict:
        """Edit existing benchmark."""
        parser = ConfigurationParser()
        parsed_input_data = parser.parse(data)
        parsed_benchmark_data: BenchmarkEditParamsInterface = (
            BenchmarkAPIInterface.parse_benchmark_edit_data(
                parsed_input_data,
            )
        )
        response: dict = {"id": parsed_benchmark_data.id}
        with Session.begin() as db_session:
            benchmark = Benchmark.details(
                db_session,
                parsed_benchmark_data.id,
            )

            if benchmark.get("status", None) is not None:
                raise ClientErrorException("Can not modify benchmark  that has been run.")

            if parsed_benchmark_data.dataset_id is not None:
                response.update(
                    Benchmark.update_dataset(
                        db_session=db_session,
                        benchmark_id=parsed_benchmark_data.id,
                        dataset_id=parsed_benchmark_data.dataset_id,
                    ),
                )

            if parsed_benchmark_data.batch_size is not None:
                response.update(
                    Benchmark.update_batch_size(
                        db_session=db_session,
                        benchmark_id=parsed_benchmark_data.id,
                        batch_size=parsed_benchmark_data.batch_size,
                    ),
                )

            if parsed_benchmark_data.mode is not None:
                response.update(
                    Benchmark.update_mode(
                        db_session=db_session,
                        benchmark_id=parsed_benchmark_data.id,
                        mode=parsed_benchmark_data.mode,
                    ),
                )

            if parsed_benchmark_data.cores_per_instance is not None:
                response.update(
                    Benchmark.update_cores_per_instance(
                        db_session=db_session,
                        benchmark_id=parsed_benchmark_data.id,
                        cores_per_instance=parsed_benchmark_data.cores_per_instance,
                    ),
                )

            if parsed_benchmark_data.number_of_instance is not None:
                response.update(
                    Benchmark.update_number_of_instance(
                        db_session=db_session,
                        benchmark_id=parsed_benchmark_data.id,
                        number_of_instance=parsed_benchmark_data.number_of_instance,
                    ),
                )

        return response

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
    def parse_benchmark_edit_data(data: dict) -> BenchmarkEditParamsInterface:
        """Parse data for editing benchmark."""
        benchmark_data = BenchmarkEditParamsInterface()
        try:
            benchmark_data.id = int(data.get("id", None))
            if "dataset_id" in data:
                benchmark_data.dataset_id = int(data.get("dataset_id", None))
            if "batch_size" in data:
                benchmark_data.batch_size = int(data.get("batch_size", None))
            if "mode" in data:
                benchmark_data.mode = str(data.get("mode", None))
            if "cores_per_instance" in data:
                benchmark_data.cores_per_instance = int(data.get("cores_per_instance", None))
            if "number_of_instance" in data:
                benchmark_data.number_of_instance = int(data.get("number_of_instance", None))
        except ValueError:
            raise ClientErrorException("Could not parse value")
        except TypeError:
            raise ClientErrorException("Could not find required parameter.")

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


@event.listens_for(Benchmark, "before_delete")
def before_delete_benchmark_entry(
    mapper: Mapper,
    connection: Connection,
    benchmark: Benchmark,
) -> None:
    """Clean up benchmark data before remove."""
    Optimization.unpin_benchmark(
        db_connection=connection,
        benchmark_id=benchmark.id,
    )
