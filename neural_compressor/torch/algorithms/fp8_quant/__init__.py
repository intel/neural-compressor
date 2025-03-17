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

from neural_compressor.torch.algorithms.fp8_quant.common import (
    update_mode,
    save_calib_result,
    restore_patched_module,
    with_patched_module,
)
from neural_compressor.torch.algorithms.fp8_quant.prepare_quant.prepare_model import finish_measurements, prep_model
from neural_compressor.torch.algorithms.fp8_quant.quantizer import FP8Quantizer
from neural_compressor.torch.algorithms.fp8_quant.patched_module_base import (
    PatchedModuleBase,
    register_patched_module,
)

from neural_compressor.torch.algorithms.fp8_quant.scaling_method_base import (
    ScalingMethodBase,
    register_scaling_methods,
)
from neural_compressor.torch.algorithms.fp8_quant.observer import (
    ObserverBase,
    register_observer,
)
from neural_compressor.torch.algorithms.fp8_quant.model_configs import (
    ModuleConfig,
    ModuleInfo,
    ModuleType,
    ModuleExtraConfig
)
from neural_compressor.torch.algorithms.fp8_quant.save_load import save, load
