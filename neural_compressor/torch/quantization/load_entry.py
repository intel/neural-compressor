# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

from neural_compressor.common.utils import FP8_QUANT, GPTQ, RTN  # unified namespace


def load(model, output_dir="./saved_results"):
    qmodel_file_path = os.path.join(os.path.abspath(os.path.expanduser(output_dir)), "quantized_model.pt")
    qconfig_file_path = os.path.join(os.path.abspath(os.path.expanduser(output_dir)), "qconfig.json")
    with open(qconfig_file_path, "r") as f:
        model_qconfig = json.load(f)
    if model_qconfig["algorithm"] == FP8_QUANT:
        from neural_compressor.torch.algorithms.habana_fp8 import load

        return load(model, output_dir)
