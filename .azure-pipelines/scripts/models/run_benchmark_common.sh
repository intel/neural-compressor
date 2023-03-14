#!/bin/bash
set -eo pipefail
source /neural-compressor/.azure-pipelines/scripts/change_color.sh

# get parameters
PATTERN='[-a-zA-Z0-9_]*='
SCRIPTS_PATH="/neural-compressor/.azure-pipelines/scripts/models"

for i in "$@"; do
    case $i in
        --framework=*)
            framework=`echo $i | sed "s/${PATTERN}//"`;;
        --model=*)
            model=`echo $i | sed "s/${PATTERN}//"`;;
        --input_model=*)
            input_model=`echo $i | sed "s/${PATTERN}//"`;;
        --benchmark_cmd=*)
            benchmark_cmd=`echo $i | sed "s/${PATTERN}//"`;;
        --log_dir=*)
            log_dir=`echo $i | sed "s/${PATTERN}//"`;;
        --new_benchmark=*)
            new_benchmark=`echo $i | sed "s/${PATTERN}//"`;;
        --precision=*)
            precision=`echo $i | sed "s/${PATTERN}//"`;;
        --stage=*)
            stage=`echo $i | sed "s/${PATTERN}//"`;;
        --USE_TUNE_ACC=*)
            USE_TUNE_ACC=`echo $i | sed "s/${PATTERN}//"`;;
        --PERF_STABLE_CHECK=*)
            PERF_STABLE_CHECK=`echo $i | sed "s/${PATTERN}//"`;;
        --BUILD_BUILDID=*)
            BUILD_BUILDID=`echo $i | sed "s/${PATTERN}//"`;;
        *)
            echo "Parameter $i not recognized."; exit 1;;
    esac
done

$BOLD_YELLOW && echo "-------- run_benchmark_common --------" && $RESET

main() {
    # run accuracy
    echo "USE_TUNE_ACC=${USE_TUNE_ACC}, PERF_STABLE_CHECK=${PERF_STABLE_CHECK}"
    # USE_TUNE_ACC==true means using accuracy results from tuning log
    if [ ${USE_TUNE_ACC} == "false" ]; then
        run_accuracy
    fi

    # run performance
    if [ ${PERF_STABLE_CHECK} == "false" ]; then
        run_performance
    else
        max_loop=3
        gap=(0.05 0.05 0.1)
        for ((iter = 0; iter < ${max_loop}; iter++)); do
            run_performance
            {
                check_perf_gap ${gap[${iter}]}
                exit_code=$?
            } || true

            if [ ${exit_code} -ne 0 ]; then
                $BOLD_RED && echo "FAILED with performance gap!!" && $RESET
            else
                $BOLD_GREEN && echo "SUCCEED!!" && $RESET
                break
            fi
        done
        exit ${exit_code}
    fi
}

function check_perf_gap() {
    python -u ${SCRIPTS_PATH}/collect_log_model.py \
        --framework=${framework} \
        --fwk_ver=${fwk_ver} \
        --model=${model} \
        --logs_dir="${log_dir}" \
        --output_dir="${log_dir}" \
        --build_id=${BUILD_BUILDID} \
        --stage=${stage} \
        --gap=$1
}

function run_performance() {
    cmd="${benchmark_cmd} --input_model=${input_model}"
    if [ "${new_benchmark}" == "true" ]; then
        $BOLD_YELLOW && echo "run with internal benchmark..." && $RESET
        export NUM_OF_INSTANCE=2
        export CORES_PER_INSTANCE=4
        eval ${cmd} 2>&1 | tee ${log_dir}/${framework}-${model}-performance-${precision}.log
    else
        $BOLD_YELLOW && echo "run with external multiInstance benchmark..." && $RESET
        multiInstance
    fi
}

function run_accuracy() {
    $BOLD_YELLOW && echo "run tuning accuracy in precision ${precision}" && $RESET
    eval "${benchmark_cmd} --input_model=${input_model} --mode=accuracy" 2>&1 | tee ${log_dir}/${framework}-${model}-accuracy-${precision}.log
}

function multiInstance() {
    ncores_per_socket=${ncores_per_socket:=$(lscpu | grep 'Core(s) per socket' | cut -d: -f2 | xargs echo -n)}
    $BOLD_YELLOW && echo "Executing multi instance benchmark" && $RESET
    ncores_per_instance=4
    $BOLD_YELLOW && echo "ncores_per_socket=${ncores_per_socket}, ncores_per_instance=${ncores_per_instance}" && $RESET

    logFile="${log_dir}/${framework}-${model}-performance-${precision}"
    benchmark_pids=()

    core_list=$(python ${SCRIPTS_PATH}/new_benchmark.py --cores_per_instance=${ncores_per_instance} --num_of_instance=$(expr $ncores_per_socket / $ncores_per_instance))
    core_list=($(echo $core_list | tr ';' ' '))

    for ((j = 0; $j < $(expr $ncores_per_socket / $ncores_per_instance); j = $(($j + 1)))); do
        $BOLD_GREEN && echo "OMP_NUM_THREADS=${ncores_per_instance} numactl --localalloc --physcpubind=${core_list[${j}]} ${cmd} 2>&1 | tee ${logFile}-${ncores_per_socket}-${ncores_per_instance}-${j}.log &" && $RESET
        OMP_NUM_THREADS=${ncores_per_instance} numactl --localalloc --physcpubind=${core_list[${j}]} ${cmd} 2>&1 | tee ${logFile}-${ncores_per_socket}-${ncores_per_instance}-${j}.log &
        benchmark_pids+=($!)
    done

    status="SUCCESS"
    for pid in "${benchmark_pids[@]}"; do
        wait $pid
        exit_code=$?
        $BOLD_YELLOW && echo "Detected exit code: ${exit_code}" && $RESET
        if [ ${exit_code} == 0 ]; then
            $BOLD_GREEN && echo "Process ${pid} succeeded" && $RESET
        else
            $BOLD_RED && echo "Process ${pid} failed" && $RESET
            status="FAILURE"
        fi
    done

    $BOLD_YELLOW && echo "Benchmark process status: ${status}" && $RESET
    if [ ${status} == "FAILURE" ]; then
        $BOLD_RED && echo "Benchmark process returned non-zero exit code." && $RESET
        exit 1
    fi
}

main
