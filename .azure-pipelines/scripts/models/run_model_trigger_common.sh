#!/bin/bash
set -eo pipefail
source /neural-compressor/.azure-pipelines/scripts/change_color.sh
# get parameters
PATTERN='[-a-zA-Z0-9_]*='

for i in "$@"
do
    case $i in
        --yaml=*)
            yaml=`echo $i | sed "s/${PATTERN}//"`;;
        --framework=*)
            framework=`echo $i | sed "s/${PATTERN}//"`;;
        --fwk_ver=*)
            fwk_ver=`echo $i | sed "s/${PATTERN}//"`;;
        --torch_vision_ver=*)
            torch_vision_ver=`echo $i | sed "s/${PATTERN}//"`;;
        --model=*)
            model=`echo $i | sed "s/${PATTERN}//"`;;
        --model_src_dir=*)
            model_src_dir=`echo $i | sed "s/${PATTERN}//"`;;
        --dataset_location=*)
            dataset_location=`echo $i | sed "s/${PATTERN}//"`;;
        --input_model=*)
            input_model=`echo $i | sed "s/${PATTERN}//"`;;
        --batch_size=*)
            batch_size=`echo $i | sed "s/${PATTERN}//"`;;
        --strategy=*)
            strategy=`echo $i | sed "s/${PATTERN}//"`;;
        --new_benchmark=*)
            new_benchmark=`echo $i | sed "s/${PATTERN}//"`;;
        --inc_new_api=*)
            inc_new_api=`echo $i | sed "s/${PATTERN}//"`;;
        --tuning_cmd=*)
            tuning_cmd=`echo $i | sed "s/${PATTERN}//"`;;
        --benchmark_cmd=*)
            benchmark_cmd=`echo $i | sed "s/${PATTERN}//"`;;
        --mode=*)
            mode=`echo $i | sed "s/${PATTERN}//"`;;
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

log_dir="/neural-compressor/.azure-pipelines/scripts/models"
SCRIPTS_PATH="/neural-compressor/.azure-pipelines/scripts/models"
if [[ "${inc_new_api}" == "3x"* ]]; then
    WORK_SOURCE_DIR="/neural-compressor/examples/3.x_api/${framework}"
else
    WORK_SOURCE_DIR="/neural-compressor/examples/${framework}"
fi
$BOLD_YELLOW && echo "processing ${framework}-${fwk_ver}-${model}" && $RESET

if [ "${mode}" == "env_setup" ]; then
    /bin/bash env_setup.sh \
        --yaml=${yaml} \
        --framework=${framework} \
        --fwk_ver=${fwk_ver} \
        --torch_vision_ver=${torch_vision_ver} \
        --model=${model} \
        --model_src_dir=${model_src_dir} \
        --dataset_location=${dataset_location} \
        --batch_size=${batch_size} \
        --strategy=${strategy} \
        --new_benchmark=${new_benchmark} \
        --inc_new_api="${inc_new_api}"
elif [ "${mode}" == "tuning" ]; then
    if [ "${framework}" == "onnxrt" ]; then
        output_model=${log_dir}/${model}/${framework}-${model}-tune.onnx
    elif [ "${framework}" == "mxnet" ]; then
        output_model=${log_dir}/${model}/resnet50_v1
    elif [ "${framework}" == "tensorflow" ]; then
        output_model=${log_dir}/${model}/${framework}-${model}-tune.pb
    fi
    [[ ${output_model} ]] && tuning_cmd="${tuning_cmd} --output_model=${output_model}"

    cd ${WORK_SOURCE_DIR}/${model_src_dir}
    $BOLD_YELLOW && echo "workspace ${WORK_SOURCE_DIR}/${model_src_dir}" && $RESET
    $BOLD_YELLOW && echo "tuning_cmd is === ${tuning_cmd}" && $RESET
    $BOLD_YELLOW && echo "======== run tuning ========" && $RESET
    /bin/bash ${SCRIPTS_PATH}/run_tuning_common.sh \
        --tuning_cmd="${tuning_cmd}" \
        --strategy=${strategy} \
        2>&1 | tee -a ${log_dir}/${model}/${framework}-${model}-tune.log
    $BOLD_YELLOW && echo "====== check tuning status. ======" && $RESET
    if [[ "${inc_new_api}" == "3x" ]]; then
        control_phrase="Quantization end."
    else
        control_phrase="model which meet accuracy goal."
    fi
    if [ $(grep "${control_phrase}" ${log_dir}/${model}/${framework}-${model}-tune.log | wc -l) == 0 ];then
        $BOLD_RED && echo "====== Quantization FAILED!! ======" && $RESET; exit 1
    fi
    if [ $(grep "${control_phrase}" ${log_dir}/${model}/${framework}-${model}-tune.log | grep "Not found" | wc -l) == 1 ];then
        $BOLD_RED && echo "====== Quantization FAILED!! ======" && $RESET; exit 1
    fi
    $BOLD_GREEN && echo "====== Quantization SUCCEED!! ======" && $RESET
elif [ "${mode}" == "fp32_benchmark" ]; then
    cd ${WORK_SOURCE_DIR}/${model_src_dir}
    $BOLD_YELLOW && echo "workspace ${WORK_SOURCE_DIR}/${model_src_dir}" && $RESET
    $BOLD_YELLOW && echo "benchmark_cmd is ${benchmark_cmd}" && $RESET
    $BOLD_YELLOW && echo "====== run benchmark fp32 =======" && $RESET
    /bin/bash ${SCRIPTS_PATH}/run_benchmark_common.sh \
        --framework=${framework} \
        --model=${model} \
        --input_model=${input_model} \
        --benchmark_cmd="${benchmark_cmd}" \
        --log_dir="${log_dir}/${model}" \
        --new_benchmark=${new_benchmark} \
        --precision="fp32" \
        --stage=${mode} \
        --USE_TUNE_ACC=${USE_TUNE_ACC} \
        --PERF_STABLE_CHECK=${PERF_STABLE_CHECK} \
        --BUILD_BUILDID=${BUILD_BUILDID}
elif [ "${mode}" == "int8_benchmark" ]; then
    cd ${WORK_SOURCE_DIR}/${model_src_dir}
    $BOLD_YELLOW && echo "workspace ${WORK_SOURCE_DIR}/${model_src_dir}" && $RESET
    $BOLD_YELLOW && echo "benchmark_cmd is ${benchmark_cmd}" && $RESET
    $BOLD_YELLOW && echo "====== run benchmark int8 =======" && $RESET
    if [[ "${framework}" == "onnxrt" ]]; then
        model_name="${log_dir}/${model}/${framework}-${model}-tune.onnx"
    elif [[ "${framework}" == "mxnet" ]]; then
        model_name="${log_dir}/${model}"
    elif [[ "${framework}" == "tensorflow" ]]; then
        model_name="${log_dir}/${model}/${framework}-${model}-tune.pb"
    elif [[ "${framework}" == "pytorch" ]]; then
        model_name=${input_model}
        benchmark_cmd="${benchmark_cmd} --int8=true"
    fi
    /bin/bash ${SCRIPTS_PATH}/run_benchmark_common.sh \
        --framework=${framework} \
        --model=${model} \
        --input_model="${model_name}" \
        --benchmark_cmd="${benchmark_cmd}" \
        --log_dir="${log_dir}/${model}" \
        --new_benchmark=${new_benchmark} \
        --precision="int8" \
        --stage=${mode} \
        --USE_TUNE_ACC=${USE_TUNE_ACC} \
        --PERF_STABLE_CHECK=${PERF_STABLE_CHECK} \
        --BUILD_BUILDID=${BUILD_BUILDID}
elif [ "${mode}" == "collect_log" ]; then
    cd ${WORK_SOURCE_DIR}/${model_src_dir}
    $BOLD_YELLOW && echo "workspace ${WORK_SOURCE_DIR}/${model_src_dir}" && $RESET
    $BOLD_YELLOW && echo "====== collect logs of model ${model} =======" && $RESET
    python -u ${SCRIPTS_PATH}/collect_log_model.py \
        --framework=${framework} \
        --fwk_ver=${fwk_ver} \
        --model=${model} \
        --logs_dir="${log_dir}/${model}" \
        --output_dir="${log_dir}/${model}" \
        --build_id=${BUILD_BUILDID} \
        --stage=${mode} \
        --inc_new_api="${inc_new_api}"
    $BOLD_YELLOW && echo "====== Finish collect logs =======" && $RESET
fi
