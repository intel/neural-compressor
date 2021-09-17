Step-by-Step
============

This document list steps of reproducing Intel Optimized PyTorch bert-base-cased/uncased models tuning results via LPOT with quantization aware training.

Our example comes from [Huggingface/transformers](https://github.com/huggingface/transformers)


# Prerequisite

### 1. Installation

PyTorch 1.8 is needed for pytorch_fx backend and huggingface/transformers.

  ```shell
  cd examples/pytorch/fx/huggingface_models/text-classification/qat/
  pip install -r requirements.txt
  ```


### 2. Prepare fine-tuned model

  ```shell
  python run_glue.py \
    --model_name_or_path bert-base-cased \
    --task_name mrpc \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --output_dir bert_model
  ```

# Run

### 1. Enable bert-base-cased/uncased example with the auto quantization aware training strategy of LPOT.

  The changes made are as follows:
  1. add conf_qat.yaml:  
    This file contains the configuration of quantization.  
  2. edit run_glue_tune.py:  
    - For quantization, We used lpot in it.  
    - For training, we enbaled early stop strategy.  

### 2. To get the tuned model and its accuracy: 

    bash run_tuning.sh --input_model=./bert_model  --output_model=./saved_results

### 3. To get the benchmark of tuned model, includes Batch_size and Throughput: 

    bash run_benchmark.sh --input_model=./bert_model --config=./saved_results --mode=benchmark --int8=true/false
