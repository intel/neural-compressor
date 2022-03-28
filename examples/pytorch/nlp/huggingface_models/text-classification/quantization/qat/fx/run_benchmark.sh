#!/bin/bash
set -x


function main {

    init_params "$@"
    run_benchmark

}

# init params
function init_params {
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
    if [[ ${mode} == "benchmark" ]]; then
        mode_cmd="--benchmark "
    fi

    if [[ ${int8} == "true" ]]; then
        mode_cmd=$mode_cmd"--int8"
    fi

    if [ ${iters} ]; then
        samples=`expr $iters \* $batch_size`
        mode_cmd=$mode_cmd" --max_eval_samples="${samples}" --max_train_samples="${samples}
    fi

    python run_glue_tune.py \
        --model_name_or_path ${input_model} \
        --task_name ${task_name} \
        --do_train \
        --do_eval \
        --max_seq_length 128 \
        --per_device_eval_batch_size ${batch_size} \
        --per_device_train_batch_size ${batch_size} \
        --learning_rate 2e-5 \
        --num_train_epochs 3 \
        --metric_for_best_model f1 \
        --output_dir ${tuned_checkpoint} --overwrite_output_dir \
        ${mode_cmd}
}

main "$@"
