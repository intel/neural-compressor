#!/bin/bash
set -eo pipefail
source /neural-compressor/.azure-pipelines/scripts/change_color.sh

# get parameters
PATTERN='[-a-zA-Z0-9_]*='

for i in "$@"
do
    case $i in
        --framework=*)
            framework=`echo $i | sed "s/${PATTERN}//"`;;
        --model=*)
            model=`echo $i | sed "s/${PATTERN}//"`;;
        --input_model=*)
            input_model=`echo $i | sed "s/${PATTERN}//"`;;
        --benchmark_cmd=*)
            benchmark_cmd=`echo $i | sed "s/${PATTERN}//"`;;
        --tune_acc=*)
            tune_acc=`echo $i | sed "s/${PATTERN}//"`;;
        --log_dir=*)
            log_dir=`echo $i | sed "s/${PATTERN}//"`;;
        --new_benchmark=*)
            new_benchmark=`echo $i | sed "s/${PATTERN}//"`;;
        --precision=*)
            precision=`echo $i | sed "s/${PATTERN}//"`;;
        *)
            echo "Parameter $i not recognized."; exit 1;;
    esac
done

$BOLD_YELLOW && echo "-------- run_benchmark_common --------" && $RESET

# run accuracy
# tune_acc==true means using accuracy results from tuning log
if [ "${tune_acc}" == "false" ]; then
    $BOLD_YELLOW && echo "run tuning accuracy in precision ${precision}" && $RESET
    eval "${benchmark_cmd} --input_model=${input_model} --mode=accuracy" 2>&1 | tee ${log_dir}/${framework}-${model}-accuracy-${precision}.log
fi


function multiInstance() {
    ncores_per_socket=${ncores_per_socket:=$( lscpu | grep 'Core(s) per socket' | cut -d: -f2 | xargs echo -n)}
    $BOLD_YELLOW && echo "Executing multi instance benchmark" && $RESET
    ncores_per_instance=4
    $BOLD_YELLOW && echo "ncores_per_socket=${ncores_per_socket}, ncores_per_instance=${ncores_per_instance}" && $RESET

    logFile="${log_dir}/${framework}-${model}-performance-${precision}"
    benchmark_pids=()

    for((j=0;$j<${ncores_per_socket};j=$(($j + ${ncores_per_instance}))));
    do
        end_core_num=$((j + ncores_per_instance -1))
        if [ ${end_core_num} -ge ${ncores_per_socket} ]; then
            end_core_num=$((ncores_per_socket-1))
        fi
        numactl -m 0 -C "${j}-${end_core_num}" ${cmd} 2>&1 | tee ${logFile}-${ncores_per_socket}-${ncores_per_instance}-${j}.log &
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


# run performance
cmd="${benchmark_cmd} --input_model=${input_model}"

if [ "${new_benchmark}" == "true" ]; then
    $BOLD_YELLOW && echo "run with internal benchmark..." && $RESET
    eval ${cmd} 2>&1 | tee ${log_dir}/${framework}-${model}-performance-${precision}.log
else
    $BOLD_YELLOW && echo "run with external multiInstance benchmark..." && $RESET
    multiInstance
fi
