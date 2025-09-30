#!/bin/bash
set -x

function main {

    init_params "$@"
    run_benchmark

}

# init params
function init_params {
    batch_size=128
    iters=100
    for var in "$@"
    do
        case $var in
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
                iters=$(echo $var |cut -f2 -d=)
            ;;
        esac
    done

}


# run_tuning
function run_benchmark {
    if [[ ${mode} == "accuracy" ]]; then
        python main.py \
            --input_model ${input_model} \
            --dataset_location ${dataset_location} \
            --mode ${mode} \
            --batch_size ${batch_size} \
            --benchmark
    elif [[ ${mode} == "performance" ]]; then
        incbench --num_c 4 main.py \
            --input_model ${input_model} \
            --dataset_location ${dataset_location} \
            --mode ${mode} \
            --batch_size ${batch_size} \
            --iteration ${iters} \
            --benchmark
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi
}

main "$@"
