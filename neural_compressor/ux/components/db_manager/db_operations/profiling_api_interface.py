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
"""INC Bench Profiling API interface."""
import os
import shutil
from typing import List, Optional, Union

from sqlalchemy.orm import sessionmaker

from neural_compressor.ux.components.configuration_wizard.configuration_parser import (
    ConfigurationParser,
)
from neural_compressor.ux.components.db_manager.db_manager import DBManager
from neural_compressor.ux.components.db_manager.db_models.profiling import Profiling
from neural_compressor.ux.components.db_manager.db_models.profiling_result import ProfilingResult
from neural_compressor.ux.components.db_manager.db_operations.project_api_interface import (
    ProjectAPIInterface,
)
from neural_compressor.ux.components.db_manager.params_interfaces import (
    ProfilingAddParamsInterface,
    ProfilingEditParamsInterface,
    ProfilingResultAddParamsInterface,
)
from neural_compressor.ux.components.jobs_management import jobs_control_queue, parse_job_id
from neural_compressor.ux.utils.consts import WORKSPACE_LOCATION, ExecutionStatus
from neural_compressor.ux.utils.exceptions import ClientErrorException
from neural_compressor.ux.utils.logger import log
from neural_compressor.ux.utils.utils import normalize_string

db_manager = DBManager()
Session = sessionmaker(bind=db_manager.engine)


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
            job_id = parse_job_id("profiling", profiling_id)
            jobs_control_queue.abort_job(job_id, blocking=True)
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
    def edit_profiling(data: dict) -> dict:
        """Edit existing profiling."""
        parser = ConfigurationParser()
        parsed_input_data = parser.parse(data)
        parsed_profiling_data: ProfilingEditParamsInterface = (
            ProfilingAPIInterface.parse_profiling_edit_data(
                parsed_input_data,
            )
        )
        response: dict = {"id": parsed_profiling_data.id}
        with Session.begin() as db_session:
            profiling = Profiling.details(
                db_session,
                parsed_profiling_data.id,
            )

            if profiling.get("status", None) is not None:
                raise ClientErrorException("Can not modify profiling  that has been run.")

            if parsed_profiling_data.dataset_id is not None:
                response.update(
                    Profiling.update_dataset(
                        db_session=db_session,
                        profiling_id=parsed_profiling_data.id,
                        dataset_id=parsed_profiling_data.dataset_id,
                    ),
                )

            if parsed_profiling_data.num_threads is not None:
                response.update(
                    Profiling.update_num_threads(
                        db_session=db_session,
                        profiling_id=parsed_profiling_data.id,
                        num_threads=parsed_profiling_data.num_threads,
                    ),
                )
        return response

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
    def parse_profiling_edit_data(data: dict) -> ProfilingEditParamsInterface:
        """Parse data for editing profiling."""
        profiling_data = ProfilingEditParamsInterface()
        try:
            profiling_data.id = int(data.get("id", None))
            if "dataset_id" in data:
                profiling_data.dataset_id = int(data.get("dataset_id", None))
            if "num_threads" in data:
                profiling_data.num_threads = int(data.get("num_threads", None))
        except ValueError:
            raise ClientErrorException("Could not parse value")
        except TypeError:
            raise ClientErrorException("Could not find required parameter.")

        return profiling_data

    @staticmethod
    def clean_status(status_to_clean: ExecutionStatus) -> dict:
        """Clean specified optimization status."""
        with Session.begin() as db_session:
            response = Profiling.clean_status(
                db_session=db_session,
                status_to_clean=status_to_clean,
            )
        return response
