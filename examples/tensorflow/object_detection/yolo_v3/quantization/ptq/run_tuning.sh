#!/bin/bash
set -x

function main {

    init_params "$@"
    run_benchmark

}

# init params
function init_params {
    for var in "$@"
    do
        case $var in
            --config=*)
                config=$(echo $var |cut -f2 -d=)
            ;;
            --input_model=*)
                input_model=$(echo $var |cut -f2 -d=)
            ;;
            --output_model=*)
                output_model=$(echo $var |cut -f2 -d=)
            ;;
        esac
    done

}


# run tuning
function run_benchmark {
    python infer_detections.py \
        --input_graph ${input_model} \
        --config ${config} \
        --output_graph ${output_model}
}

main "$@"
