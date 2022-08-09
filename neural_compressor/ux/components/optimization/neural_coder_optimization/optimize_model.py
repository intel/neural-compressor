# -*- coding: utf-8 -*-
# Copyright (c) 2021 Intel Corporation
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
"""Package graph_optimizer contains all packages required to optimize model graph."""

import shutil
import time
from typing import Any

from neural_coder import enable


# TO-DO: change it to some NeuralCoderOptimization class that inherits from Optimization
# Then it should be also added to OptimizationFactory
def optimize_pt_script(
    optimization: Any,
) -> float:
    """Optimization of PyTorch scripted models."""
    time_start = time.time()

    copy_model_path = optimization.workdir + "/copy_model.py"
    pytorch_script_file_path = optimization.input_graph
    file = open(copy_model_path, "w")
    file.close()
    shutil.copy(pytorch_script_file_path, copy_model_path)

    if optimization.output_precision == "int8 static quantization":
        enable(
            code=copy_model_path,
            features=["pytorch_inc_static_quant"],
            overwrite=False,
            save_patch_path=optimization.workdir + "/",
        )
    else:
        enable(
            code=copy_model_path,
            features=["pytorch_inc_dynamic_quant"],
            overwrite=False,
            save_patch_path=optimization.workdir + "/",
        )

    time_end = time.time()

    optimization_time = time_end - time_start

    return optimization_time
