# Copyright (c) 2026 Intel Corporation
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

from neural_compressor.jax.algorithms.dynamic import dynamic_quantize
from neural_compressor.jax.algorithms.static import static_quantize
from .quantize import quantize_model
from .config import DynamicQuantConfig, StaticQuantConfig, get_default_dynamic_config, get_default_static_config
