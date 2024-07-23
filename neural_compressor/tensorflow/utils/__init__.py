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
"""The utils for Tensorflow."""

from neural_compressor.tensorflow.utils.data import BaseDataLoader, DummyDataset, DummyDatasetV2
from neural_compressor.tensorflow.utils.model import (
    Model,
    TFConfig,
)
from neural_compressor.tensorflow.utils.constants import (
    SPR_BASE_VERSIONS,
    TENSORFLOW_DEFAULT_CONFIG,
    DEFAULT_SQ_ALPHA_ARGS,
    UNIFY_OP_TYPE_MAPPING,
)
from neural_compressor.tensorflow.utils.model_wrappers import (
    get_tf_model_type,
    BaseModel,
    KerasModel,
    TensorflowLLMModel,
    TensorflowBaseModel,
    TensorflowSavedModelModel,
)
from neural_compressor.tensorflow.utils.utility import (
    disable_random,
    algos_mapping,
    version1_lt_version2,
    version1_gt_version2,
    version1_eq_version2,
    version1_gte_version2,
    version1_lte_version2,
    register_algo,
    deep_get,
    itex_installed,
    dump_elapsed_time,
    combine_histogram,
    get_all_fp32_data,
    get_tensor_histogram,
    singleton,
    CpuInfo,
    Statistics,
    CaptureOutputToFile,
    valid_keras_format,
    TFSlimNetsFactory,
)
