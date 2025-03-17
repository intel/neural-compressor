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

dataset_location=""
input_model=""
yaml=""
strategy=""
batch_size=""
new_benchmark=true
inc_new_api=true
benchmark_cmd=""
# ======== set up config for pytorch models ========
if [ "${model}" == "resnet18" ]; then
    model_src_dir="image_recognition/torchvision_models/quantization/ptq/cpu/eager"
    dataset_location="/tf_dataset2/datasets/mini-imageraw"
    input_model=""
    yaml="conf.yaml"
    strategy="bayesian"
    batch_size=1
    new_benchmark=false
    inc_new_api=false
    tuning_cmd="bash run_tuning.sh --topology=resnet18 --dataset_location=${dataset_location} --input_model=${input_model}"
    benchmark_cmd="bash run_benchmark.sh --topology=resnet18 --dataset_location=${dataset_location} --mode=benchmark --batch_size=${batch_size} --iters=500"
elif [ "${model}" == "resnet18_fx" ]; then
    model_src_dir="image_recognition/torchvision_models/quantization/ptq/cpu/fx/"
    dataset_location="/tf_dataset2/datasets/mini-imageraw"
    input_model="resnet18"
    yaml=""
    strategy="basic"
    batch_size=1
    new_benchmark=true
    inc_new_api=true
    tuning_cmd="bash run_quant.sh --topology=resnet18 --dataset_location=${dataset_location} --input_model=${input_model}"
    benchmark_cmd="bash run_benchmark.sh --topology=resnet18 --dataset_location=${dataset_location} --mode=performance --batch_size=${batch_size} --iters=500"
elif [ "${model}" == "opt_125m_woq_gptq_int4" ]; then
    model_src_dir="nlp/huggingface_models/language-modeling/quantization/weight_only"
    inc_new_api=3x_pt
    tuning_cmd="bash run_quant.sh --topology=opt_125m_woq_gptq_int4"
elif [ "${model}" == "opt_125m_woq_gptq_nf4_dq_bnb" ]; then
    model_src_dir="nlp/huggingface_models/language-modeling/quantization/weight_only"
    inc_new_api=3x_pt
    tuning_cmd="bash run_quant.sh --topology=opt_125m_woq_gptq_nf4_dq_bnb"
elif [ "${model}" == "opt_125m_woq_gptq_int4_dq_ggml" ]; then
    model_src_dir="nlp/huggingface_models/language-modeling/quantization/weight_only"
    inc_new_api=3x_pt
    tuning_cmd="bash run_quant.sh --topology=opt_125m_woq_gptq_int4_dq_ggml"
fi

echo "Specify FWs version..."

FRAMEWORK="pytorch"
source /neural-compressor/.azure-pipelines/scripts/fwk_version.sh 'latest'
if [[ "${inc_new_api}" == "3x"* ]]; then
    FRAMEWORK_VERSION="latest"
    export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
else
    FRAMEWORK_VERSION=${pytorch_version}
    TORCH_VISION_VERSION=${torchvision_version}
fi


/bin/bash run_model_trigger_common.sh \
    --yaml=${yaml} \
    --framework=${FRAMEWORK} \
    --fwk_ver=${FRAMEWORK_VERSION} \
    --torch_vision_ver=${TORCH_VISION_VERSION} \
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