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
"""INC Bench Examples API interface."""

from sqlalchemy.orm import session, sessionmaker

from neural_compressor.ux.components.db_manager.db_manager import DBManager
from neural_compressor.ux.components.db_manager.db_models.benchmark import Benchmark
from neural_compressor.ux.components.db_manager.db_models.optimization_type import OptimizationType
from neural_compressor.ux.components.db_manager.db_models.precision import Precision
from neural_compressor.ux.components.db_manager.db_models.project import Project
from neural_compressor.ux.components.db_manager.db_operations.benchmark_api_interface import (
    BenchmarkAPIInterface,
)
from neural_compressor.ux.components.db_manager.db_operations.optimization_api_interface import (
    OptimizationAPIInterface,
)
from neural_compressor.ux.components.db_manager.db_operations.project_api_interface import (
    ProjectAPIInterface,
)
from neural_compressor.ux.components.db_manager.params_interfaces import (
    BenchmarkAddParamsInterface,
    OptimizationAddParamsInterface,
)
from neural_compressor.ux.components.model_zoo.download_model import download_model
from neural_compressor.ux.utils.consts import OptimizationTypes, Precisions
from neural_compressor.ux.utils.exceptions import ClientErrorException, InternalException
from neural_compressor.ux.utils.utils import load_model_config
from neural_compressor.ux.web.communication import MessageQueue

db_manager = DBManager()
Session = sessionmaker(bind=db_manager.engine)


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
