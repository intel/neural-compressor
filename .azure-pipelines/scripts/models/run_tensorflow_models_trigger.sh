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
FRAMEWORK="tensorflow"
FRAMEWORK_VERSION=${tensorflow_version}

inc_new_api=false
# ======== set up config for tensorflow models ========
if [ "${model}" == "resnet50v1.5" ]; then
    model_src_dir="image_recognition/tensorflow_models/resnet50_v1_5/quantization/ptq"
    dataset_location="/tf_dataset/dataset/TF_mini_imagenet"
    input_model="/tf_dataset/pre-trained-models/resnet50v1_5/fp32/resnet50_v1.pb"
    new_benchmark=true
    inc_new_api=true
    tuning_cmd="bash run_quant.sh --dataset_location=${dataset_location} --input_model=${input_model}"
    benchmark_cmd="bash run_benchmark.sh --dataset_location=${dataset_location} --batch_size=1 --mode=performance"
elif [ "${model}" == "ssd_resnet50_v1" ];then
    model_src_dir="object_detection/tensorflow_models/ssd_resnet50_v1/quantization/ptq"
    dataset_location="/tf_dataset/tensorflow/mini-coco-100.record"
    input_model="/tf_dataset/pre-train-model-oob/object_detection/ssd_resnet50_v1/frozen_inference_graph.pb"
    new_benchmark=true
    inc_new_api=true
    tuning_cmd="bash run_quant.sh --dataset_location=${dataset_location} --input_model=${input_model}"
    benchmark_cmd="bash run_benchmark.sh --dataset_location=${dataset_location} --batch_size=1 --mode=performance"
elif [ "${model}" == "ssd_mobilenet_v1_ckpt" ];then
    model_src_dir="object_detection/tensorflow_models/ssd_mobilenet_v1/quantization/ptq"
    dataset_location="/tf_dataset/tensorflow/mini-coco-100.record"
    input_model="/tf_dataset/pre-train-model-oob/object_detection/ssd_mobilenet_v1"
    new_benchmark=true
    inc_new_api=true
    tuning_cmd="bash run_quant.sh --dataset_location=${dataset_location} --input_model=${input_model}"
    benchmark_cmd="bash run_benchmark.sh --dataset_location=${dataset_location} --batch_size=1 --mode=performance"
elif [ "${model}" == "inception_v1" ]; then
    model_src_dir="image_recognition/tensorflow_models/quantization/ptq"
    dataset_location="/tf_dataset/dataset/TF_mini_imagenet"
    input_model="/tf_dataset/pre-train-model-slim/pbfile/frozen_pb/frozen_inception_v1.pb"
    yaml="inception_v1.yaml"
    strategy="basic"
    batch_size=1
    new_benchmark=true
    tuning_cmd="bash run_tuning.sh --config=${yaml} --input_model=${input_model}"
    benchmark_cmd="bash run_benchmark.sh --config=${yaml} --mode=performance"
elif [ "${model}" == "darknet19" ]; then
    model_src_dir="oob_models/quantization/ptq"
    dataset_location=""
    input_model="/tf_dataset/tensorflow/tf_oob_models/ov/all_tf_models/PublicInHouse/classification/darknet19/darknet19.pb"
    yaml="config.yaml"
    strategy="basic"
    batch_size=1
    new_benchmark=false
    inc_new_api=true
    tuning_cmd="bash run_quant.sh --topology=${model} --input_model=${input_model}"
    benchmark_cmd="bash run_benchmark.sh --topology=${model} --mode=performance --batch_size=1 --iters=500"
elif [ "${model}" == "densenet-121" ]; then
    model_src_dir="oob_models/quantization/ptq"
    dataset_location=""
    input_model="/tf_dataset/tensorflow/tf_oob_models/ov/all_tf_models/classification/densenet/121/tf/densenet-121.pb"
    yaml="config.yaml"
    strategy="basic"
    batch_size=1
    new_benchmark=false
    inc_new_api=true
    tuning_cmd="bash run_quant.sh --topology=${model} --input_model=${input_model}"
    benchmark_cmd="bash run_benchmark.sh --topology=${model} --mode=performance --batch_size=1 --iters=500"
elif [ "${model}" == "resnet-101" ]; then
    model_src_dir="oob_models/quantization/ptq"
    dataset_location=""
    input_model="/tf_dataset/tensorflow/tf_oob_models/ov/all_tf_models/classification/resnet/v1/101/tf/resnet-101.pb"
    yaml="config.yaml"
    strategy="basic"
    batch_size=1
    new_benchmark=false
    inc_new_api=true
    tuning_cmd="bash run_quant.sh --topology=${model} --input_model=${input_model}"
    benchmark_cmd="bash run_benchmark.sh --topology=${model} --mode=performance --batch_size=1 --iters=500"
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
