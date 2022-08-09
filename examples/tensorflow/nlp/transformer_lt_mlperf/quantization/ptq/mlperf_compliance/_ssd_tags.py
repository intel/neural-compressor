# Copyright 2018 MLBenchmark Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keys which only appear in SSD.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Pretrained classifer model
BACKBONE = "backbone"

FEATURE_SIZES = "feature_sizes"
STEPS = "steps"
SCALES = "scales"
ASPECT_RATIOS = "aspect_ratios"
NUM_DEFAULTS_PER_CELL = "num_defaults_per_cell"
LOC_CONF_OUT_CHANNELS = "loc_conf_out_channels"
NUM_DEFAULTS = "num_default_boxes"

# Overlap threshold for NMS
NMS_THRESHOLD = "nms_threshold"
NMS_MAX_DETECTIONS = "nms_max_detections"

# data pipeline
NUM_CROPPING_ITERATIONS = "num_cropping_iterations"
RANDOM_FLIP_PROBABILITY = "random_flip_probability"
DATA_NORMALIZATION_MEAN = "data_normalization_mean"
DATA_NORMALIZATION_STD = "data_normalization_std"
