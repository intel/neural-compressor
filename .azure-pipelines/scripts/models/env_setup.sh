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
/bin/bash /neural-compressor/.azure-pipelines/scripts/install_nc.sh

mkdir -p ${WORK_SOURCE_DIR}
cd ${WORK_SOURCE_DIR}
if [[ "${inc_new_api}" == "false" ]]; then
    echo "copy old api examples to workspace..."
    git clone -b old_api_examples https://github.com/intel/neural-compressor.git old-lpot-models
    cd old-lpot-models
    git branch
    cd -
    rm -rf ${model_src_dir}
    mkdir -p ${model_src_dir}
    cp -r old-lpot-models/examples/${framework}/${model_src_dir} ${WORK_SOURCE_DIR}/${model_src_dir}/../
fi

cd ${model_src_dir}
pip install ruamel.yaml==0.17.40
pip install psutil
pip install protobuf==4.23.4
if [[ "${framework}" == "tensorflow" ]]; then
    pip install intel-tensorflow==${fwk_ver}
elif [[ "${framework}" == "pytorch" ]]; then
    pip install torch==${fwk_ver} -f https://download.pytorch.org/whl/torch_stable.html
    pip install torchvision==${torch_vision_ver} -f https://download.pytorch.org/whl/torch_stable.html
elif [[ "${framework}" == "onnxrt" ]]; then
    pip install onnx==1.14.1
    pip install onnxruntime==${fwk_ver}
elif [[ "${framework}" == "mxnet" ]]; then
    pip install numpy==1.23.5
    echo "re-install pycocotools resolve the issue with numpy..."
    pip uninstall pycocotools -y
    pip install --no-cache-dir pycocotools
    pip install mxnet==${fwk_ver}
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
    $BOLD_RED && echo "Not found requirements.txt file." && $RESET
fi

if [[ "${inc_new_api}" == "false" ]]; then
    $BOLD_YELLOW && echo "======== update yaml config ========" && $RESET
    $BOLD_YELLOW && echo -e "\nPrint origin yaml..." && $RESET
    cat ${yaml}
    python ${SCRIPTS_PATH}/update_yaml_config.py \
        --yaml=${yaml} \
        --framework=${framework} \
        --dataset_location=${dataset_location} \
        --batch_size=${batch_size} \
        --strategy=${strategy} \
        --new_benchmark=${new_benchmark} \
        --multi_instance='true'
    $BOLD_YELLOW && echo -e "\nPrint updated yaml... " && $RESET
    cat ${yaml}
fi
