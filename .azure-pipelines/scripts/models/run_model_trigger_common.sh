#!/bin/bash

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
        --tuning_cmd=*)
            tuning_cmd=`echo $i | sed "s/${PATTERN}//"`;;
        --benchmark_cmd=*)
            benchmark_cmd=`echo $i | sed "s/${PATTERN}//"`;;
        --tune_acc=*)
            tune_acc=`echo $i | sed "s/${PATTERN}//"`;;
        --mode=*)
            mode=`echo $i | sed "s/${PATTERN}//"`;;
        *)
            echo "Parameter $i not recognized."; exit 1;;
    esac
done

log_dir="/neural-compressor/.azure-pipelines/scripts/models"
WORK_SOURCE_DIR="/neural-compressor/examples/${framework}"
SCRIPTS_PATH="/neural-compressor/.azure-pipelines/scripts/models"
echo "processing ${framework}-${fwk_ver}-${model}"
echo "tuning_cmd is ${tuning_cmd}"
echo "benchmark_cmd is ${benchmark_cmd}"

if [ "${mode}" == "env_setup" ]; then
    echo "======= creat log_dir ========="
    if [ -d "${log_dir}/${model}" ]; then
        echo "${log_dir}/${model} already exists, don't need to mkdir."
    else
        echo "no log dir ${log_dir}/${model}, create."
        cd ${log_dir}
        mkdir ${model}
    fi

    echo "====== install requirements ======"
    /bin/bash /neural-compressor/.azure-pipelines/scripts/install_nc.sh

    cd ${WORK_SOURCE_DIR}/${model_src_dir}
    pip install ruamel_yaml
    pip install psutil
    pip install protobuf==3.20.1
    if [[ "${framework}" == "tensorflow" ]]; then
        pip install intel-tensorflow==${fwk_ver}
    elif [[ "${framework}" == "pytorch" ]]; then
        pip install torch==${fwk_ver} -f https://download.pytorch.org/whl/torch_stable.html
        pip install torchvision==${torch_vision_ver} -f https://download.pytorch.org/whl/torch_stable.html
    elif [[ "${framework}" == "onnxrt" ]]; then
        pip install onnx==1.11.0
        pip install onnxruntime==${fwk_ver}
    elif [[ "${framework}" == "mxnet" ]]; then
        if [[ "${fwk_ver}" == "1.7.0" ]]; then
            pip install mxnet==${fwk_ver}.post2
        elif [[ "${fwk_ver}" == "1.6.0" ]]; then
            pip install mxnet-mkl==${mxnet_version}
        else
            pip install mxnet==${fwk_ver}
        fi
    fi

    if [ -f "requirements.txt" ]; then
        sed -i '/neural-compressor/d' requirements.txt
        if [ "${framework}" == "onnxrt" ]; then
            sed -i '/^onnx>=/d;/^onnx==/d;/^onnxruntime>=/d;/^onnxruntime==/d' requirements.txt
        fi
        if [ "${framework}" == "tensorflow" ]; then
            sed -i '/tensorflow==/d;/tensorflow$/d' requirements.txt
            sed -i '/^intel-tensorflow/d' requirements.txt
        fi
        if [ "${framework}" == "mxnet" ]; then
            sed -i '/mxnet==/d;/mxnet$/d;/mxnet-mkl==/d;/mxnet-mkl$/d' requirements.txt
        fi
        if [ "${framework}" == "pytorch" ]; then
            sed -i '/torch==/d;/torch$/d;/torchvision==/d;/torchvision$/d' requirements.txt
        fi
        n=0
        until [ "$n" -ge 5 ]; do
            python -m pip install -r requirements.txt && break
            n=$((n + 1))
            sleep 5
        done
        pip list
    else
        echo "Not found requirements.txt file."
    fi

    echo "======== update yaml config ========"
    echo -e "\nPrint origin yaml..."
    cat ${yaml}
    python ${SCRIPTS_PATH}/update_yaml_config.py \
        --yaml=${yaml} \
        --framework=${framework} \
        --dataset_location=${dataset_location} \
        --batch_size=${batch_size} \
        --strategy=${strategy} \
        --new_benchmark=${new_benchmark} \
        --multi_instance='true'
    echo -e "\nPrint updated yaml... "
    cat ${yaml}
elif [ "${mode}" == "tuning" ]; then
    cd ${WORK_SOURCE_DIR}/${model_src_dir}
    echo "======== run tuning ========"
    /bin/bash ${SCRIPTS_PATH}/run_tuning_common.sh \
        --framework=${framework} \
        --model=${model} \
        --tuning_cmd="${tuning_cmd}" \
        --log_dir="${log_dir}/${model}" \
        --input_model=${input_model} \
        --strategy=${strategy} \
        2>&1 | tee -a ${log_dir}/${model}/${framework}-${model}-tune.log
elif [ "${mode}" == "int8_benchmark" ]; then
    cd ${WORK_SOURCE_DIR}/${model_src_dir}
    echo "====== run benchmark fp32 ======="
    /bin/bash ${SCRIPTS_PATH}/run_benchmark_common.sh \
        --framework=${framework} \
        --model=${model} \
        --input_model=${input_model} \
        --benchmark_cmd="${benchmark_cmd}" \
        --tune_acc=${tune_acc} \
        --log_dir="${log_dir}/${model}" \
        --new_benchmark=${new_benchmark} \
        --precision="fp32"
elif [ "${mode}" == "fp32_benchmark" ]; then
    cd ${WORK_SOURCE_DIR}/${model_src_dir}
    echo "====== run benchmark int8 ======="
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
        --tune_acc=${tune_acc} \
        --log_dir="${log_dir}/${model}" \
        --new_benchmark=${new_benchmark} \
        --precision="int8"
elif [ "${mode}" == "collect_log" ]; then
    cd ${WORK_SOURCE_DIR}/${model_src_dir}
    echo "====== collect logs of model ${model} ======="
    python -u ${SCRIPTS_PATH}/collect_log_model.py \
        --framework=${framework} \
        --fwk_ver=${fwk_ver} \
        --model=${model} \
        --logs_dir="${log_dir}/${model}" \
        --output_dir="${log_dir}/${model}" \
        --build_id=${BUILD_BUILDID}
    echo "====== Finish model test ======="
fi
