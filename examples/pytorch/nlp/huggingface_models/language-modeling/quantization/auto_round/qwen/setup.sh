#!/bin/bash
set -e
usage() {
    echo "Usage: $0 --device=[gpu|xpu] --format=[AR|LLMC] --task=[task_list] --bench_tool=[lm_eval|aisbench]"
    echo "  --device    target device for quantization (gpu or xpu)"
    echo "  --format    quantization format (AR for auto_round, LLMC for llm_compressor)"
    echo "  --task      comma-separated list of evaluation tasks (e.g. gsm8k,hellaswag)"
    echo "  --bench_tool benchmarking tool to use (lm_eval or aisbench)"
}

DEVICE="gpu"
FORMAT="LLMC"
TASKS="hellaswag,piqa,mmlu,gsm8k"
BENCH_TOOL="lm_eval"

while [[ $# -gt 0 ]]; do
    case $1 in
        --device=*)
            DEVICE="${1#*=}"
            shift
            ;;
        --format=*)
            FORMAT="${1#*=}"
            shift
            ;;
        --task=*)
            TASKS="${1#*=}"
            shift
            ;;
        --bench_tool=*)
            BENCH_TOOL="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ "$DEVICE" == "xpu" ]]; then
    uv pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/xpu
    uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/xpu
    uv pip install setuptools --upgrade
    uv pip install packaging --upgrade
    uv pip install -U "huggingface_hub[cli]"
    # install vllm and plugins
    git clone -b v0.19.1 --single-branch  https://github.com/vllm-project/vllm.git
    cd vllm
    uv pip install --upgrade pip
    uv pip install -v -r requirements/xpu.txt
    uv pip uninstall -y triton triton-xpu
    uv pip install triton-xpu==3.6.0 --extra-index-url https://download.pytorch.org/whl/xpu
    VLLM_TARGET_DEVICE=xpu uv pip install --no-build-isolation --no-deps . -v
    cd ..
elif [[ "$DEVICE" == "gpu" ]]; then
    uv pip install -r requirements.txt
    uv pip install setuptools --upgrade
    uv pip install packaging --upgrade
    uv pip install -U "huggingface_hub[cli]"
    if [[ "$FORMAT" == "LLMC" ]]; then
        uv pip install vllm==0.19.1
        git clone https://github.com/yiliu30/vllm-qdq-plugin.git
        uv pip install vllm-qdq-plugin/ -v
        cd ..
    else
        # use default setting 
        git clone -b fused-moe-ar  --single-branch --quiet https://github.com/yiliu30/vllm-fork.git && cd vllm-fork
        VLLM_USE_PRECOMPILED=1 uv pip install . -v
        cd ..
    fi
else
    echo "Unsupported device: $DEVICE. Supported devices are gpu and xpu."
    usage
    exit 1
fi

# Uninstall flash_attn to avoid conflicts
uv pip uninstall flash_attn -y

