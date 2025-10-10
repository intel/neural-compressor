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
"""Keys which only appear in GNMT RNN Translation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Loss smoothing factor
MODEL_HP_LOSS_SMOOTHING = "model_hp_loss_smoothing"

# Number of layers in encoder and in decoder
MODEL_HP_NUM_LAYERS = "model_hp_num_layers"

# RNN hidden size
MODEL_HP_HIDDEN_SIZE = "model_hp_hidden_size"

# Dropout
MODEL_HP_DROPOUT = "model_hp_dropout"

# Beam size for beam search
EVAL_HP_BEAM_SIZE = "eval_hp_beam_size"

# Maximum sequence length for training
TRAIN_HP_MAX_SEQ_LEN = "train_hp_max_sequence_length"

# Maximum sequence length for evaluation
EVAL_HP_MAX_SEQ_LEN = "eval_hp_max_sequence_length"

# Length normalization constant for beam search
EVAL_HP_LEN_NORM_CONST = "eval_hp_length_normalization_constant"

# Length normalization factor for beam search
EVAL_HP_LEN_NORM_FACTOR = "eval_hp_length_normalization_factor"

# Coverage penalty factor for beam search
EVAL_HP_COV_PENALTY_FACTOR = "eval_hp_coverage_penalty_factor"
