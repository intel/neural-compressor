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

"""Keys which only appear in transformer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

INPUT_MAX_LENGTH = "input_max_length"

MODEL_HP_INITIALIZER_GAIN = "model_hp_initializer_gain"
MODEL_HP_VOCAB_SIZE = "model_hp_vocab_size"
MODEL_HP_NUM_HIDDEN_LAYERS = "model_hp_hidden_layers"
MODEL_HP_EMBEDDING_SHARED_WEIGHTS = "model_hp_embedding_shared_weights"
MODEL_HP_ATTENTION_DENSE = "model_hp_attention_dense"
MODEL_HP_ATTENTION_DROPOUT = "model_hp_attention_dropout"
MODEL_HP_FFN_OUTPUT_DENSE = "model_hp_ffn_output_dense"
MODEL_HP_FFN_FILTER_DENSE = "model_hp_ffn_filter_dense"
MODEL_HP_RELU_DROPOUT = "model_hp_relu_dropout"
MODEL_HP_LAYER_POSTPROCESS_DROPOUT = "model_hp_layer_postprocess_dropout"
MODEL_HP_NORM = "model_hp_norm"
MODEL_HP_SEQ_BEAM_SEARCH = "model_hp_sequence_beam_search"
