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
"""Master list of MLPerf tags to be logged for benchmark submissions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mlperf_compliance._gnmt_tags import *
from mlperf_compliance._ncf_tags import *
from mlperf_compliance._resnet_tags import *
from mlperf_compliance._ssd_tags import *
from mlperf_compliance._transformer_tags import *
from mlperf_compliance._maskrcnn_tags import *

# ==============================================================================
# == Benchmarks ================================================================
# ==============================================================================

# rnn_translator
GNMT = "gnmt"

# reinforcement/
MINIGO = "minigo"

# recommendation/
NCF = "ncf"

# image_classification/
RESNET = "resnet"

# single_stage_detector/
SSD = "ssd"

# object_detection/
MASKRCNN = "maskrcnn"

# translation/
TRANSFORMER = "transformer"

# ==============================================================================
# == Tags ======================================================================
# ==============================================================================
"""
Tags may be used by all models, a subset of models, or only one model. A
specification for which models require which tags can be found below the tag
definitions.
"""

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# All models: Tags which should appear in absolutely every MLPerf model.
# //////////////////////////////////////////////////////////////////////////////

# This tag signals to start the timer. Emission of this tag need not be (and
# generally will not be) the first part of a submission script. Rather, this
# tag must be emitted prior to performing any work which the MLPerf rules
# state must be timed. This tag is generally emitted directly before the first
# step which invokes random number generation or the first step which must be
# performed on the system under test. (Whichever comes first.) If clarification
# is needed, please file an issue under:
#   https://github.com/mlperf/policies
RUN_START = "run_start"

# This tag signals that a submission has reached the relevant stopping criteria,
# and has completed all tasks which are performed in the reference. The wall
# time for a submission will be computed as the difference between the time
# when this tag is emitted and the time whe the RUN_START is emitted.
RUN_STOP = "run_stop"

# This tag should be emitted immediately before ending a run, and should be the
# last tag emitted. This tag should indicate the completion of untimed post
# processing work such as system specific cleanup.
RUN_FINAL = "run_final"


# Emit this tag in the place(s) where random seeds are set.
RUN_SET_RANDOM_SEED = "run_set_random_seed"

# Emit this tag when training data has been purged from volatile caches prior
# to run start.
RUN_CLEAR_CACHES = "run_clear_caches"

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# Common Values: Constants which are expected to be reported across many models.
#                These values are included for convenience.
# //////////////////////////////////////////////////////////////////////////////
BCE = "binary_cross_entropy"
CCE = "categorical_cross_entropy"

SGD = "stochastic_gradient_descent"

# Some conventions distinguish between "vanilla" SGD and SGD with momentum
# (where vanilla SGD would be the specific case of momentum=0)
SGD_WITH_MOMENTUM = "stochastic_gradient_descent_with_momentum"

ADAM = "adam"
LAZY_ADAM = "lazy_adam"

TRUNCATED_NORMAL = "truncated_normal"

RELU = "relu"


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# Preprocessing: Tags for generic preprocessing steps
# //////////////////////////////////////////////////////////////////////////////

# The number of training examples in a single epoch
PREPROC_NUM_TRAIN_EXAMPLES = "preproc_num_train_examples"

# The number of evaluation examples in a single epoch
PREPROC_NUM_EVAL_EXAMPLES = "preproc_num_eval_examples"

# This tag is used to declare what part of code tokenizes the training data.
PREPROC_TOKENIZE_TRAINING = "preproc_tokenize_training"

# This tag is used to declare what part of code tokenizes the evaluation data.
PREPROC_TOKENIZE_EVAL = "preproc_tokenize_eval"

# The vocabulary size used for tokenization.
PREPROC_VOCAB_SIZE = "preproc_vocab_size"


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# Input: Tags for the timed portion of the data input pipeline
# //////////////////////////////////////////////////////////////////////////////

# The number of examples in the training portion of the data pipeline. Generally
# this should match PREPROC_NUM_TRAIN_EXAMPLES. If it does not (for instance
# if certain examples are dropped in compliance with MLPerf rules), the
# call which declares this tag is a good place for a comment stating why the
# disparity is expected.
INPUT_SIZE = "input_size"

# The size of a training minibatch size. If this value is variable, please emit
# "-1" and then log an implementation specific characterization of the batch
# size which is a reasonable analog to the reference. (For instance log that
# all but the last batch has size 64, and the last batch is a partial batch)
INPUT_BATCH_SIZE = "input_batch_size"

# This tag indicates where the location of the code which defines the order in
# which training examples are traversed. It is not necessary to describe the
# method in the tag emission (though comments are always welcome). Rather, this
# should simply provide a good starting point to an interested party.
INPUT_ORDER = "input_order"

# The shard size (in items) when shuffling in the input pipeline.
INPUT_SHARD = "input_shard"

# The number of samples iver which BN stats are computed for normalization during training
INPUT_BN_SPAN = "input_bn_span"


# --------------------------------------
# -- Data Augmentation and Alteration --
# --------------------------------------

# ResNet random cropping
INPUT_CENTRAL_CROP = "input_central_crop"

INPUT_CROP_USES_BBOXES = "input_crop_uses_bboxes"

INPUT_DISTORTED_CROP_MIN_OBJ_COV = "input_distorted_crop_min_object_covered"
INPUT_DISTORTED_CROP_RATIO_RANGE = "input_distorted_crop_aspect_ratio_range"
INPUT_DISTORTED_CROP_AREA_RANGE = "input_distorted_crop_area_range"
INPUT_DISTORTED_CROP_MAX_ATTEMPTS = "input_distorted_crop_max_attempts"

INPUT_MEAN_SUBTRACTION = "input_mean_subtraction"

# Random flip of an image for data augmentation
INPUT_RANDOM_FLIP = "input_random_flip"

INPUT_RESIZE = "input_resize"
INPUT_RESIZE_ASPECT_PRESERVING = "input_resize_aspect_preserving"


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# Opt: Tags for declaring optimizer specific information. Submissions should
#      declare and log explicit values rather than relying on defaults.
# //////////////////////////////////////////////////////////////////////////////

# The name of the optimizer used. (SGD, Adam, etc.)
OPT_NAME = "opt_name"

OPT_LR = "opt_learning_rate"
OPT_MOMENTUM = "opt_momentum"

OPT_WEIGHT_DECAY = "opt_weight_decay"

# beta1, beta2, and epsilon are optimizer hyperparameters associated with the
# Adam optimizer and its variants (e.g. LazyAdam).
OPT_HP_ADAM_BETA1 = "opt_hp_Adam_beta1"
OPT_HP_ADAM_BETA2 = "opt_hp_Adam_beta2"
OPT_HP_ADAM_EPSILON = "opt_hp_Adam_epsilon"

# The number of warm-up steps (SGD).
OPT_LR_WARMUP_STEPS = "opt_learning_rate_warmup_steps"


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#  Train: Tags for control flow during model training.
# //////////////////////////////////////////////////////////////////////////////

# This tag is emitted when a model first enters its training loop. This is not
# necessarily when it begins to apply gradients; rather, it should be placed at
# a location which logically partitions the submission code.
TRAIN_LOOP = "train_loop"

# The current epoch as said epoch begins training.
TRAIN_EPOCH = "train_epoch"

# This tag is used to indicate approximately where checkpoints are written. Some
# frameworks abstract away checkpoint saving; in such cases simply choose a
# logical place in the code which signals that the framework has been instructed
# to save checkpoints, along with an explanatory comment.
TRAIN_CHECKPOINT = "train_checkpoint"


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#  Eval: Tags for control flow during model evaluation.
# //////////////////////////////////////////////////////////////////////////////

# This tag should be emitted whenever the submission begins an evaluation pass
# for a given set of weights.
EVAL_START = "eval_start"

# The number of examples on which evaluation is performed.
EVAL_SIZE = "eval_size"

# The target quality at which the model may stop training.
EVAL_TARGET = "eval_target"

# The observed accuracy of the model at a given epoch.
EVAL_ACCURACY = "eval_accuracy"

# This tag should be emitted whenever the submission ends an evaluation pass
# for a given set of weights.
EVAL_STOP = "eval_stop"


# The observed accuracy of the model at a given iteration. This is only for
# models which evaluate at certain iterations instead of epochs.
EVAL_ITERATION_ACCURACY = "eval_iteration_accuracy"

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#  Model: Tags for logging topology specific information.
# //////////////////////////////////////////////////////////////////////////////

# The loss function (cross entropy, squared error, etc.) used by the model. For
# more exotic loss functions such as those encountered in object detection
# models, additional benchmark specific subcomponents should also be logged.
MODEL_HP_LOSS_FN = "model_hp_loss_fn"

MODEL_HP_INITIAL_SHAPE = "model_hp_initial_shape"
MODEL_HP_FINAL_SHAPE = "model_hp_final_shape"

MODEL_L2_REGULARIZATION = "model_l2_regularization"
MODEL_EXCLUDE_BN_FROM_L2 = "model_exclude_bn_from_l2"

MODEL_HP_RELU = "model_hp_relu"
MODEL_HP_CONV2D_FIXED_PADDING = "model_hp_conv2d_fixed_padding"
MODEL_HP_BATCH_NORM = "model_hp_batch_norm"
MODEL_HP_DENSE = "model_hp_dense"


# ==============================================================================
# == Stdout tags ===============================================================
# ==============================================================================

# These tags are always logged to stdout. The rest will be logged to a file if
# one is available.
STDOUT_TAG_SET = {
    RUN_START,
    RUN_STOP,
    RUN_FINAL,

    TRAIN_LOOP,
    TRAIN_EPOCH,

    EVAL_START,
    EVAL_SIZE,
    EVAL_TARGET,
    EVAL_ACCURACY,
    EVAL_STOP,
}


# ==============================================================================
# == Benchmark tag sets ========================================================
# ==============================================================================
ALL_USED_TAGS = set()

GNMT_TAGS = (
    RUN_START,
    RUN_STOP,
    RUN_FINAL,
    RUN_SET_RANDOM_SEED,
    RUN_CLEAR_CACHES,

    PREPROC_VOCAB_SIZE,
    PREPROC_TOKENIZE_TRAINING,
    PREPROC_TOKENIZE_EVAL,
    PREPROC_NUM_TRAIN_EXAMPLES,
    PREPROC_NUM_EVAL_EXAMPLES,

    INPUT_SIZE,
    INPUT_BATCH_SIZE,
    INPUT_ORDER,
    INPUT_SHARD,

    OPT_NAME,
    OPT_LR,
    OPT_LR_WARMUP_STEPS,
    OPT_HP_ADAM_BETA1,
    OPT_HP_ADAM_BETA2,
    OPT_HP_ADAM_EPSILON,

    TRAIN_LOOP,
    TRAIN_EPOCH,
    TRAIN_CHECKPOINT,
    TRAIN_HP_MAX_SEQ_LEN,

    EVAL_START,
    EVAL_SIZE,
    EVAL_TARGET,
    EVAL_ACCURACY,
    EVAL_STOP,
    EVAL_HP_BEAM_SIZE,
    EVAL_HP_MAX_SEQ_LEN,
    EVAL_HP_LEN_NORM_CONST,
    EVAL_HP_LEN_NORM_FACTOR,
    EVAL_HP_COV_PENALTY_FACTOR,

    MODEL_HP_LOSS_FN,
    MODEL_HP_LOSS_SMOOTHING,
    MODEL_HP_NUM_LAYERS,
    MODEL_HP_HIDDEN_SIZE,
    MODEL_HP_DROPOUT
)

MASKRCNN_TAGS = (
    RUN_START,
    RUN_STOP,
    RUN_FINAL,
    RUN_CLEAR_CACHES,

    INPUT_SIZE,
    INPUT_BATCH_SIZE,
    GLOBAL_BATCH_SIZE,
    INPUT_ORDER,
    INPUT_SHARD,

    BACKBONE,
    NMS_THRESHOLD,
    NMS_MAX_DETECTIONS,

    OPT_NAME,
    OPT_LR,
    OPT_LR_WARMUP_STEPS,
    OPT_MOMENTUM,
    OPT_WEIGHT_DECAY,

    TRAIN_LOOP,
    TRAIN_EPOCH,

    EVAL_START,
    EVAL_SIZE,
    EVAL_TARGET,
    EVAL_ACCURACY,
    EVAL_STOP,
    INPUT_MEAN_SUBTRACTION,
    INPUT_NORMALIZATION_STD,
    INPUT_RESIZE,
    INPUT_RESIZE_ASPECT_PRESERVING,
    MIN_IMAGE_SIZE,
    MAX_IMAGE_SIZE,
    RUN_SET_RANDOM_SEED,
    INPUT_RANDOM_FLIP,
    RANDOM_FLIP_PROBABILITY,
    FG_IOU_THRESHOLD,
    BG_IOU_THRESHOLD,
    RPN_PRE_NMS_TOP_N_TRAIN,
    RPN_PRE_NMS_TOP_N_TEST,
    RPN_POST_NMS_TOP_N_TRAIN,
    RPN_POST_NMS_TOP_N_TEST,
    BATCH_SIZE_TEST,
    ASPECT_RATIOS,
)

MINIGO_TAGS = (
    RUN_START,
    RUN_STOP,
    RUN_FINAL,
    RUN_SET_RANDOM_SEED,
    RUN_CLEAR_CACHES,

    INPUT_SHARD,

    TRAIN_LOOP,
    TRAIN_EPOCH,

    EVAL_START,
    EVAL_SIZE,
    EVAL_TARGET,
    EVAL_ACCURACY,
    EVAL_STOP,
)

NCF_TAGS = (
    RUN_START,
    RUN_STOP,
    RUN_FINAL,
    RUN_CLEAR_CACHES,

    PREPROC_HP_MIN_RATINGS,
    PREPROC_HP_NUM_EVAL,
    PREPROC_HP_SAMPLE_EVAL_REPLACEMENT,

    INPUT_SIZE,
    INPUT_BATCH_SIZE,
    INPUT_ORDER,
    INPUT_SHARD,
    INPUT_HP_NUM_NEG,
    INPUT_HP_SAMPLE_TRAIN_REPLACEMENT,
    INPUT_STEP_TRAIN_NEG_GEN,
    INPUT_STEP_EVAL_NEG_GEN,

    OPT_NAME,
    OPT_LR,
    OPT_HP_ADAM_BETA1,
    OPT_HP_ADAM_BETA2,
    OPT_HP_ADAM_EPSILON,

    TRAIN_LOOP,
    TRAIN_EPOCH,

    EVAL_START,
    EVAL_SIZE,
    EVAL_TARGET,
    EVAL_ACCURACY,
    EVAL_STOP,
    EVAL_HP_NUM_USERS,
    EVAL_HP_NUM_NEG,

    MODEL_HP_LOSS_FN,
    MODEL_HP_MF_DIM,
    MODEL_HP_MLP_LAYER_SIZES,
)

RESNET_TAGS = (
    RUN_START,
    RUN_STOP,
    RUN_FINAL,
    RUN_SET_RANDOM_SEED,
    RUN_CLEAR_CACHES,

    PREPROC_NUM_TRAIN_EXAMPLES,
    PREPROC_NUM_EVAL_EXAMPLES,

    INPUT_SIZE,
    INPUT_BATCH_SIZE,
    INPUT_ORDER,
    INPUT_SHARD,
    INPUT_CENTRAL_CROP,
    INPUT_CROP_USES_BBOXES,
    INPUT_DISTORTED_CROP_MIN_OBJ_COV,
    INPUT_DISTORTED_CROP_RATIO_RANGE,
    INPUT_DISTORTED_CROP_AREA_RANGE,
    INPUT_DISTORTED_CROP_MAX_ATTEMPTS,
    INPUT_MEAN_SUBTRACTION,
    INPUT_RANDOM_FLIP,
    INPUT_RESIZE,
    INPUT_RESIZE_ASPECT_PRESERVING,
    INPUT_BN_SPAN,

    OPT_NAME,
    OPT_LR,
    OPT_LR_WARMUP_STEPS,
    OPT_MOMENTUM,

    TRAIN_LOOP,
    TRAIN_EPOCH,

    EVAL_START,
    EVAL_SIZE,
    EVAL_TARGET,
    EVAL_ACCURACY,
    EVAL_STOP,
    EVAL_EPOCH_OFFSET,

    MODEL_HP_LOSS_FN,
    MODEL_L2_REGULARIZATION,
    MODEL_EXCLUDE_BN_FROM_L2,

    MODEL_HP_INITIAL_SHAPE,
    MODEL_HP_FINAL_SHAPE,
    MODEL_HP_INITIAL_MAX_POOL,
    MODEL_HP_BEGIN_BLOCK,
    MODEL_HP_END_BLOCK,
    MODEL_HP_BLOCK_TYPE,
    MODEL_HP_PROJECTION_SHORTCUT,
    MODEL_HP_SHORTCUT_ADD,
    MODEL_HP_RELU,
    MODEL_HP_CONV2D_FIXED_PADDING,
    MODEL_HP_BATCH_NORM,
    MODEL_HP_DENSE,
    MODEL_HP_RESNET_TOPOLOGY,
)

SSD_TAGS = (
    RUN_START,
    RUN_STOP,
    RUN_FINAL,
    RUN_CLEAR_CACHES,

    INPUT_SIZE,
    INPUT_BATCH_SIZE,
    INPUT_ORDER,
    INPUT_SHARD,
    INPUT_BN_SPAN,

    BACKBONE,
    FEATURE_SIZES,
    STEPS,
    SCALES,
    ASPECT_RATIOS,
    NUM_DEFAULTS_PER_CELL,
    LOC_CONF_OUT_CHANNELS,
    NUM_DEFAULTS,
    NMS_THRESHOLD,
    NMS_MAX_DETECTIONS,

    NUM_CROPPING_ITERATIONS,
    RANDOM_FLIP_PROBABILITY,
    DATA_NORMALIZATION_MEAN,
    DATA_NORMALIZATION_STD,

    OPT_NAME,
    OPT_LR,
    OPT_MOMENTUM,
    OPT_WEIGHT_DECAY,
    OPT_LR_WARMUP_STEPS,

    TRAIN_LOOP,
    TRAIN_EPOCH,

    EVAL_START,
    EVAL_SIZE,
    EVAL_TARGET,
    EVAL_ACCURACY,
    EVAL_STOP,
    EVAL_ITERATION_ACCURACY,
)

TRANSFORMER_TAGS = (
    RUN_START,
    RUN_STOP,
    RUN_FINAL,
    RUN_SET_RANDOM_SEED,
    RUN_CLEAR_CACHES,

    PREPROC_NUM_TRAIN_EXAMPLES,
    PREPROC_NUM_EVAL_EXAMPLES,
    PREPROC_TOKENIZE_TRAINING,
    PREPROC_TOKENIZE_EVAL,
    PREPROC_VOCAB_SIZE,

    INPUT_BATCH_SIZE,
    INPUT_MAX_LENGTH,
    INPUT_ORDER,
    INPUT_SHARD,

    OPT_NAME,
    OPT_LR,
    OPT_LR_WARMUP_STEPS,
    OPT_HP_ADAM_BETA1,
    OPT_HP_ADAM_BETA2,
    OPT_HP_ADAM_EPSILON,

    TRAIN_LOOP,
    TRAIN_EPOCH,

    EVAL_START,
    EVAL_TARGET,
    EVAL_ACCURACY,
    EVAL_STOP,

    MODEL_HP_INITIALIZER_GAIN,
    MODEL_HP_VOCAB_SIZE,
    MODEL_HP_NUM_HIDDEN_LAYERS,
    MODEL_HP_EMBEDDING_SHARED_WEIGHTS,
    MODEL_HP_ATTENTION_DENSE,
    MODEL_HP_ATTENTION_DROPOUT,
    MODEL_HP_FFN_OUTPUT_DENSE,
    MODEL_HP_FFN_FILTER_DENSE,
    MODEL_HP_RELU_DROPOUT,
    MODEL_HP_LAYER_POSTPROCESS_DROPOUT,
    MODEL_HP_NORM,
    MODEL_HP_SEQ_BEAM_SEARCH,
)

ALL_USED_TAGS.update(GNMT_TAGS)
ALL_USED_TAGS.update(MASKRCNN_TAGS)
ALL_USED_TAGS.update(NCF_TAGS)
ALL_USED_TAGS.update(RESNET_TAGS)
ALL_USED_TAGS.update(SSD_TAGS)
ALL_USED_TAGS.update(TRANSFORMER_TAGS)
