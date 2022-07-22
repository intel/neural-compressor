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
"""Keys which only appear in MASKRCNN.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Anchor overlap threshop
FG_IOU_THRESHOLD = "foreground_iou_threshold"
BG_IOU_THRESHOLD = "background_iou_threshold"

# Top ROIs to be selected before and after NMS
RPN_PRE_NMS_TOP_N_TRAIN = "rpn_pre_nms_top_n_train"
RPN_PRE_NMS_TOP_N_TEST = "rpn_pre_nms_top_n_test"
RPN_POST_NMS_TOP_N_TRAIN = "rpn_post_nms_top_n_train"
RPN_POST_NMS_TOP_N_TEST = "rpn_post_nms_top_n_test"

#Global batch size during training
GLOBAL_BATCH_SIZE = "global_batch_size"

# Batch size during eval
BATCH_SIZE_TEST = "batch_size_test"


# Pretrained classifer model
BACKBONE = "backbone"

# Anchor aspect ratio
ASPECT_RATIOS = "aspect_ratios"

# Overlap threshold for NMS
NMS_THRESHOLD = "nms_threshold"

# data pipeline
MIN_IMAGE_SIZE = "min_image_size"
MAX_IMAGE_SIZE = "max_image_size"
RANDOM_FLIP_PROBABILITY = "random_flip_probability"
INPUT_NORMALIZATION_STD = "input_normalization_std"
