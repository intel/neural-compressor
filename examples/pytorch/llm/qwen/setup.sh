#!/bin/bash
set -e
usage() {
    echo "Usage: $0 --device=[gpu|xpu] --task=[task_list] --bench_tool=[lm_eval|aisbench]"
    echo "  --device    target device for quantization (gpu or xpu)"
    echo "  --task      comma-separated list of evaluation tasks (e.g. gsm8k,hellaswag)"
    echo "  --bench_tool benchmarking tool to use (lm_eval or aisbench)"
}

DEVICE="${DEVICE:-gpu}"
TASKS="${TASKS:-hellaswag,piqa,mmlu,gsm8k,ruler}"
BENCH_TOOL="${BENCH_TOOL:-lm_eval}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --device=*)
            DEVICE="${1#*=}"
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
    # support quant only on xpu for now
    uv pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/xpu
    uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/xpu
elif [[ "$DEVICE" == "gpu" ]]; then
    uv pip install -r requirements.txt
    uv pip install setuptools --upgrade
    uv pip install packaging --upgrade
    uv pip install -U "huggingface_hub[cli]"
    uv pip install vllm==0.25.1

    uv pip install ray
    git clone https://github.com/yiliu30/vllm-qdq-plugin.git
    uv pip install vllm-qdq-plugin/ -v
    if [[ "$BENCH_TOOL" == "lm_eval" ]]; then
        uv pip install lm-eval==0.4.12
        uv pip install lm-eval[api]
        uv pip install lm-eval["ruler"]
        if [[ "$TASKS" == *"longbench"* ]]; then
            uv pip install "long-bench-eval @ git+https://github.com/yiliu30/long-bench-eval"
        fi
    elif [[ "$BENCH_TOOL" == "aisbench" ]]; then
        echo "Installing aisbench..."
    fi
    # Uninstall flash_attn to avoid conflicts
    uv pip uninstall flash_attn
else
    echo "Unsupported device: $DEVICE. Supported devices are gpu and xpu."
    usage
    exit 1
fi