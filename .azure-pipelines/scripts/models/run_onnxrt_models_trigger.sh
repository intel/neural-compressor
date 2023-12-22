#!/bin/bash
set -eo pipefail
# get parameters
PATTERN='[-a-zA-Z0-9_]*='

for i in "$@"
do
    case $i in
        --model=*)
            model=`echo $i | sed "s/${PATTERN}//"`;;
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

echo "specify FWs version..."
source /neural-compressor/.azure-pipelines/scripts/fwk_version.sh 'latest'
FRAMEWORK="onnxrt"
FRAMEWORK_VERSION=${onnxruntime_version}

inc_new_api=false
# ======== set up config for onnxrt models ========
if [ "${model}" == "resnet50-v1-12" ]; then
    model_src_dir="image_recognition/onnx_model_zoo/resnet50/quantization/ptq_static"
    dataset_location="/tf_dataset2/datasets/imagenet/ImagenetRaw/ImagenetRaw_small_5000/ILSVRC2012_img_val"
    input_model="/tf_dataset2/models/onnx/resnet50-v1-12/resnet50-v1-12.onnx"
    yaml="resnet50_v1_5.yaml"
    strategy="basic"
    batch_size=1
    new_benchmark=true
    inc_new_api=true
    tuning_cmd="bash run_quant.sh --input_model=${input_model} --dataset_location=${dataset_location}"
    benchmark_cmd="bash run_benchmark.sh --config=${yaml} --mode=performance --dataset_location=${dataset_location}"
fi


/bin/bash run_model_trigger_common.sh \
    --yaml=${yaml} \
    --framework=${FRAMEWORK} \
    --fwk_ver=${FRAMEWORK_VERSION} \
    --model=${model} \
    --model_src_dir=${model_src_dir} \
    --dataset_location=${dataset_location} \
    --input_model=${input_model} \
    --batch_size=${batch_size} \
    --strategy=${strategy} \
    --new_benchmark=${new_benchmark} \
    --tuning_cmd="${tuning_cmd}" \
    --benchmark_cmd="${benchmark_cmd}" \
    --inc_new_api="${inc_new_api}" \
    --mode=${mode} \
    --USE_TUNE_ACC=${USE_TUNE_ACC} \
    --PERF_STABLE_CHECK=${PERF_STABLE_CHECK} \
    --BUILD_BUILDID=${BUILD_BUILDID}