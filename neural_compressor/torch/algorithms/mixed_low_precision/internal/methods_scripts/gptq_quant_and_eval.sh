#!/bin/bash

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

# An example script of how quantize a model to 4 bit using GPTQ, and evaluate using Optimum Habana


# eval tasks
if  [ -z "${TASKS}" ]; then
    TASKS='piqa winogrande'
fi

if  [ -z "${QUANT_MODEL_OUTPUT}" ]; then
    QUANT_MODEL_OUTPUT=llama-2-7b-4bit
fi

if  [ -z "${HF_MODEL}" ]; then
    HF_MODEL=meta-llama/Llama-2-7b-hf
fi
LOG_DIR=/tmp/${QUANT_MODEL_OUTPUT}/

CWD=$PWD
PT_HPU_LAZY_MODE=2 python $NEURAL_COMPRESSOR_PATH/neural_compressor/torch/algorithms/mixed_low_precision/internal/quantization_methods/quantize_gptq.py --quantized_model_dir ${QUANT_MODEL_OUTPUT} --pretrained_model ${HF_MODEL}

cd $OPTIMUM_HABANA_PATH/examples/text-generation/
mkdir -p $LOG_DIR
python run_lm_eval.py --model_name_or_path $CWD/$QUANT_MODEL_OUTPUT --batch_size=4 -o $LOG_DIR/my_quantized_int4.log --gptq --bf16 --tasks $TASKS --use_hpu_graphs

cd $CWD
