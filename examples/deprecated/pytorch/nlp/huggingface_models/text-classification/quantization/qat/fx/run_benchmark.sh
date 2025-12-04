#!/bin/bash
set -x


function main {

    init_params "$@"
    run_benchmark

}

# init params
function init_params {
    iters=100
    batch_size=8
    task_name="mrpc"
    topology="bert-base-cased"
    tuned_checkpoint="saved_results"
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
        --mode=*)
            mode=$(echo $var |cut -f2 -d=)
        ;;
        --batch_size=*)
            batch_size=$(echo $var |cut -f2 -d=)
        ;;
        --iters=*)
            iters=$(echo ${var} |cut -f2 -d=)
        ;;
        --int8=*)
            int8=$(echo ${var} |cut -f2 -d=)
        ;;
        --config=*)
            tuned_checkpoint=$(echo $var |cut -f2 -d=)
        ;;
        *)
            echo "Error: No such parameter: ${var}"
            exit 1
        ;;
        esac
    done
}

# run_benchmark
function run_benchmark {

    mode_cmd=""
    if [[ ${mode} == "performance" ]]; then
        mode_cmd="--performance --iters "${iters}
    fi

    extra_cmd='--model_name_or_path '${input_model}
    if [[ ${int8} == "true" ]]; then
        extra_cmd='--model_name_or_path '${tuned_checkpoint}
        extra_cmd=$extra_cmd" --int8"
    fi

    python run_glue.py \
        --task_name ${task_name} \
        --do_train \
        --do_eval \
        --max_seq_length 128 \
        --per_device_eval_batch_size ${batch_size} \
        --per_device_train_batch_size ${batch_size} \
        --learning_rate 2e-5 \
        --num_train_epochs 3 \
        --metric_for_best_model f1 \
        --output_dir ./output_log --overwrite_output_dir \
        ${extra_cmd} \
        ${mode_cmd}
}

main "$@"
