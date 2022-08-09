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
"""Keys which only appear in NCF Recommendation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# The minimum number of ratings for a user to be included.
PREPROC_HP_MIN_RATINGS = "preproc_hp_min_ratings"

# The number of false negatives to use during evaluation.
PREPROC_HP_NUM_EVAL = "preproc_hp_num_eval"

# Are evaluation negatives sampled with replacement?
PREPROC_HP_SAMPLE_EVAL_REPLACEMENT = "preproc_hp_sample_eval_replacement"


# The number of false negatives per postive generated during training.
INPUT_HP_NUM_NEG = "input_hp_num_neg"

# Are training negatives sampled with replacement?
INPUT_HP_SAMPLE_TRAIN_REPLACEMENT = "input_hp_sample_train_replacement"

# This tag should be emitted each time the submission begins construction of the
# false negatives for a trainging epoch.
INPUT_STEP_TRAIN_NEG_GEN = "input_step_train_neg_gen"

# This tag should be emitted when the evaluation negatives are selected. This
# should occur only once.
INPUT_STEP_EVAL_NEG_GEN = "input_step_eval_neg_gen"

# The number of users in the evaluation set. This should be the same as the
# number of users in the training set.
EVAL_HP_NUM_USERS = "eval_hp_num_users"

# The number of false negatives per positive which actually appear during
# evaluation. This should match PREPROC_HP_NUM_EVAL.
EVAL_HP_NUM_NEG = "eval_hp_num_neg"


# The dimensionality of the matrix factorization portion of the model.
MODEL_HP_MF_DIM = "model_hp_mf_dim"

# The sizes of the fully connected layers in the dense section of the model.
MODEL_HP_MLP_LAYER_SIZES = "model_hp_mlp_layer_sizes"

