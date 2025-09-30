#!/bin/bash
set -x

function main {

  init_params "$@"
  run_pruning

}

# init params
function init_params {
  dataset_name="NeelNanda/pile-10k"
  model_name_or_path="facebook/opt-125m"
  output_dir="./test-clm"
  per_device_train_batch_size=8
  block_size=128
  gradient_accumulation_steps=4
  num_train_epochs=3
  target_sparsity=0.8
  pruning_type="snip_momentum"
  pruning_scope="local"
  pruning_pattern="4x1"
  pruning_frequency=1000
  for var in "$@"
  do
    case $var in
      --dataset_name=*)
          dataset_name=$(echo $var |cut -f2 -d=)
      ;;
      --model_name_or_path=*)
          model_name_or_path=$(echo $var |cut -f2 -d=)
      ;;
       --output_dir=*)
           output_dir=$(echo $var |cut -f2 -d=)
       ;;
       --per_device_train_batch_size=*)
           per_device_train_batch_size=$(echo $var |cut -f2 -d=)
       ;;
       --block_size=*)
           block_size=$(echo $var |cut -f2 -d=)
       ;;
       --gradient_accumulation_steps=*)
           gradient_accumulation_steps=$(echo $var |cut -f2 -d=)
       ;;
       --num_train_epochs=*)
          num_train_epochs=$(echo $var |cut -f2 -d=)
      ;;
       --target_sparsity=*)
           target_sparsity=$(echo $var |cut -f2 -d=)
       ;;
       --pruning_type=*)
           pruning_type=$(echo $var |cut -f2 -d=)
       ;;
       --pruning_scope=*)
           pruning_scope=$(echo $var |cut -f2 -d=)
       ;;
       --pruning_pattern=*)
           pruning_pattern=$(echo $var |cut -f2 -d=)
       ;;
       --pruning_frequency=*)
           pruning_frequency=$(echo $var |cut -f2 -d=)
       ;;
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

}

# run_tuning
function run_pruning {
  accelerate launch --deepspeed_config_file config/ds_config.json --mixed_precision fp16 \
      run_clm_no_trainer_deepspeed.py \
      --dataset_name $dataset_name \
      --model_name_or_path $model_name_or_path \
      --block_size $block_size \
      --per_device_train_batch_size $per_device_train_batch_size \
      --gradient_accumulation_steps $gradient_accumulation_steps \
      --output_dir $output_dir \
      --do_prune \
      --num_train_epochs $num_train_epochs \
      --target_sparsity $target_sparsity \
      --pruning_type $pruning_type \
      --pruning_scope $pruning_scope \
      --pruning_pattern $pruning_pattern \
      --pruning_frequency $pruning_frequency

}

main "$@"

