#!/bin/bash
# set -x

# Download the bert_base model on the task MRPC
git clone https://huggingface.co/bert-base-cased-finetuned-mrpc

# Export the model
python bert_base_cased_export.py \
   --input_dir=bert-base-cased-finetuned-mrpc/ \
   --output_model=bert_base_cased_mrpc.onnx

