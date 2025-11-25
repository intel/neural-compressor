#!/bin/bash
set -eo pipefail
source /neural-compressor/.azure-pipelines/scripts/change_color.sh

# get parameters
PATTERN='[-a-zA-Z0-9_]*='

for i in "$@"; do
    case $i in
    --yaml=*)
        yaml=$(echo $i | sed "s/${PATTERN}//")
        ;;
    --framework=*)
        framework=$(echo $i | sed "s/${PATTERN}//")
        ;;
    --fwk_ver=*)
        fwk_ver=$(echo $i | sed "s/${PATTERN}//")
        ;;
    --torch_vision_ver=*)
        torch_vision_ver=$(echo $i | sed "s/${PATTERN}//")
        ;;
    --model=*)
        model=$(echo $i | sed "s/${PATTERN}//")
        ;;
    --model_src_dir=*)
        model_src_dir=$(echo $i | sed "s/${PATTERN}//")
        ;;
    --dataset_location=*)
        dataset_location=$(echo $i | sed "s/${PATTERN}//")
        ;;
    --batch_size=*)
        batch_size=$(echo $i | sed "s/${PATTERN}//")
        ;;
    --strategy=*)
        strategy=$(echo $i | sed "s/${PATTERN}//")
        ;;
    --new_benchmark=*)
        new_benchmark=$(echo $i | sed "s/${PATTERN}//")
        ;;
    --inc_new_api=*)
        inc_new_api=$(echo $i | sed "s/${PATTERN}//")
        ;;
    *)
        echo "Parameter $i not recognized."
        exit 1
        ;;
    esac
done

SCRIPTS_PATH="/neural-compressor/.azure-pipelines/scripts/models"
log_dir="/neural-compressor/.azure-pipelines/scripts/models"
WORK_SOURCE_DIR="/neural-compressor/examples/${framework}"

$BOLD_YELLOW && echo "processing ${framework}-${fwk_ver}-${model}" && $RESET

$BOLD_YELLOW && echo "======= creat log_dir =========" && $RESET
if [ -d "${log_dir}/${model}" ]; then
    $BOLD_GREEN && echo "${log_dir}/${model} already exists, don't need to mkdir." && $RESET
else
    $BOLD_GREEN && echo "no log dir ${log_dir}/${model}, create." && $RESET
    cd ${log_dir}
    mkdir ${model}
fi

$BOLD_YELLOW && echo "====== install requirements ======" && $RESET
cd /neural-compressor
bash /neural-compressor/.azure-pipelines/scripts/install_nc.sh ${inc_new_api}

cd ${WORK_SOURCE_DIR}/${model_src_dir}

if [[ "${fwk_ver}" != "latest" ]]; then
    pip install ruamel.yaml==0.17.40
    pip install psutil
    pip install protobuf==4.23.4
    if [[ "${framework}" == "pytorch" ]]; then
        pip install torch==${fwk_ver} --index-url https://download.pytorch.org/whl/cpu
        pip install torchvision==${torch_vision_ver} --index-url https://download.pytorch.org/whl/cpu
    elif [[ "${framework}" == "onnxrt" ]]; then
        pip install onnx==1.15.0
        pip install onnxruntime==${fwk_ver}
    fi
fi

if [ -f "requirements.txt" ]; then
    sed -i '/neural-compressor/d' requirements.txt
    if [ "${framework}" == "onnxrt" ]; then
        sed -i '/^onnx>=/d;/^onnx==/d;/^onnxruntime>=/d;/^onnxruntime==/d' requirements.txt
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
    $BOLD_RED && echo "Not found requirements.txt file." && $RESET
fi
