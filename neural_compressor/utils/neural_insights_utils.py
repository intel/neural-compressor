# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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
"""Neural Insights utils functions."""
import os
from typing import Any, Optional

from neural_compressor.utils import logger


def register_neural_insights_workload(
    workload_location: str,
    model: Any,
    workload_mode: str,
    workload_name: str,
) -> Optional[str]:
    """Register workload to Neural Insights.

    Args:
        workload_location: path to workload directory
        model: Neural Compressor's model instance to be registered
        workload_mode: workload mode
        workload_name: Name of the workload

    Returns:
        String with Neural Insight workload UUID if registered else None
    """
    try:
        import os

        from neural_insights import NeuralInsights
        from neural_insights.utils.consts import WORKDIR_LOCATION, WorkloadModes

        try:
            mode = WorkloadModes(workload_mode)
        except ValueError:
            raise Exception(f"Workload mode '{workload_mode}' is not supported.")

        model_path, model_summary_file = get_model_path(model, workload_location)

        neural_insights = NeuralInsights(workdir_location=WORKDIR_LOCATION)
        ni_workload_uuid = neural_insights.add_workload(
            workload_location=workload_location,
            workload_mode=mode,
            model_path=model_path,
            workload_name=workload_name,
            model_summary_file=model_summary_file,
        )
        logger.info(f"Registered {workload_mode} workload to Neural Insights.")
        return ni_workload_uuid
    except ImportError:
        logger.info("Neural Insights not found.")
    except Exception as err:
        logger.warning(f"Could not register workload to Neural Insights: {err}.")
    return None


def update_neural_insights_workload(workload_uuid: str, status: str) -> None:
    """Update status of specific workload.

    Args:
        workload_uuid: string with Neural Insight workload UUID if registered else None
        status: workload status to be set

    Returns:
        None
    """
    try:
        from neural_insights import NeuralInsights
        from neural_insights.utils.consts import WORKDIR_LOCATION

        neural_insights = NeuralInsights(workdir_location=WORKDIR_LOCATION)
        neural_insights.update_workload_status(workload_uuid, status)
    except ImportError:
        logger.info("Neural Insights not found.")
    except Exception as err:
        logger.warning(f"Could not update workload status: {err}.")


def update_neural_insights_workload_accuracy_data(
    workload_uuid: str,
    baseline_accuracy: float,
    optimized_accuracy: float,
) -> None:
    """Update accuracy data of specific workload.

    Args:
        workload_uuid: string with Neural Insight workload UUID if registered else None
        baseline_accuracy: accuracy of input model
        optimized_accuracy: accuracy of optimized model

    Returns:
        None
    """
    try:
        from neural_insights import NeuralInsights
        from neural_insights.utils.consts import WORKDIR_LOCATION

        neural_insights = NeuralInsights(workdir_location=WORKDIR_LOCATION)
        neural_insights.update_workload_accuracy_data(
            workload_uuid,
            baseline_accuracy,
            optimized_accuracy,
        )
    except ImportError:
        logger.info("Neural Insights not found.")
    except Exception as err:
        logger.warning(f"Could not update workload accuracy data: {err}.")


def get_model_path(model: Any, workload_location: str) -> Any:
    """Get model path."""
    from neural_insights.utils.exceptions import ClientErrorException
    from neural_insights.utils.utils import check_module

    model_path = None
    model_summary_file = None
    onnx_installed = False
    pytorch_installed = False

    try:
        check_module("onnx")
        onnx_installed = True
    except ClientErrorException:
        pass

    try:
        check_module("torch")
        pytorch_installed = True
    except ClientErrorException:
        pass

    if isinstance(model.model_path, str):
        return os.path.abspath(model.model_path), model_summary_file
    if onnx_installed:
        import onnx

        from neural_compressor.model.onnx_model import ONNXModel

        if isinstance(model, ONNXModel):
            model_path: str = os.path.join(workload_location, "input_model.onnx")
            os.makedirs(workload_location, exist_ok=True)
            onnx.save(model.model, model_path)
            return model_path, model_summary_file
    if pytorch_installed:
        import torch
        from torchinfo import summary

        from neural_compressor.model.torch_model import PyTorchModel

        if isinstance(model, PyTorchModel):
            model_path: str = os.path.join(workload_location, "input_model.pt")
            os.makedirs(workload_location, exist_ok=True)
            torch.save(model.model.state_dict(), model_path)

            model_stats = summary(model.model, depth=5, verbose=0)
            summary_str = str(model_stats)
            model_summary_file = os.path.join(workload_location, "model_summary.txt")
            with open(model_summary_file, "w", encoding="utf-8") as summary_file:
                summary_file.write(summary_str)
            return model_path, model_summary_file
    assert isinstance(model_path, str), "Model path not detected"
