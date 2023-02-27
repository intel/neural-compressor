#!/bin/bash
set -x

function main {

    init_params "$@"
    run_tuning

}

# init params
function init_params {
    batch_size=8
    task_name="mrpc"
    output_model="saved_results"
    input_model="bert-base-cased"
    for var in "$@"
    do
      case $var in
        --topology=*)
            topology=$(echo $var |cut -f2 -d=)
        ;;
        --dataset_location=*)
            dataset_location=$(echo $var |cut -f2 -d=)
        ;;
        --input_model=*)
            input_model=$(echo $var |cut -f2 -d=)
        ;;
        --output_model=*)
            tuned_checkpoint=$(echo $var |cut -f2 -d=)
        ;;
        *)
            echo "Error: No such parameter: ${var}"
            exit 1
        ;;
      esac
    done

}

# run_tuning
function run_tuning {

    python run_glue.py \
        --model_name_or_path ${input_model} \
        --task_name ${task_name} \
        --do_train \
        --do_eval \
        --max_seq_length 128 \
        --per_device_eval_batch_size ${batch_size} \
        --per_device_train_batch_size ${batch_size} \
        --learning_rate 2e-5 \
        --num_train_epochs 3 \
        --output_dir ${output_model} --overwrite_output_dir \
        --eval_steps 300 \
        --save_steps 300 \
        --greater_is_better True \
        --load_best_model_at_end True \
        --evaluation_strategy steps \
        --save_strategy steps \
        --metric_for_best_model f1 \
        --save_total_limit 1 \
        --tune
}

main "$@"
