#!/bin/bash
set -e
usage() {
    echo "Usage: $0 --device=[gpu|xpu] --format=[AR|LLMC] --task=[task_list] --bench_tool=[lm_eval|aisbench]"
    echo "  --device    target device for quantization (gpu or xpu)"
    echo "  --format    quantization format (AR for auto_round, LLMC for llm_compressor)"
    echo "  --task      comma-separated list of evaluation tasks (e.g. gsm8k,hellaswag)"
    echo "  --bench_tool benchmarking tool to use (lm_eval or aisbench)"
}

detect_cuda_version() {
    local cuda_version=""
    local candidate_version=""

    if command -v nvidia-smi >/dev/null 2>&1; then
        candidate_version=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([^ ]*\).*/\1/p' | head -n 1)
        if [[ "$candidate_version" =~ ^[0-9.]+$ ]]; then
            cuda_version="$candidate_version"
        fi
    fi

    if [[ -z "$cuda_version" ]] && command -v nvcc >/dev/null 2>&1; then
        candidate_version=$(nvcc --version | awk '/release/ {print $6}' | sed 's/^V//; s/,//')
        if [[ "$candidate_version" =~ ^[0-9.]+$ ]]; then
            cuda_version="$candidate_version"
        fi
    fi

    if [[ -z "$cuda_version" ]]; then
        echo "Unable to detect CUDA version from nvidia-smi or nvcc." >&2
        exit 1
    fi

    echo "$cuda_version"
}

DEVICE="${DEVICE:-gpu}"
FORMAT="${FORMAT:-LLMC}"
TASKS="${TASKS:-hellaswag,piqa,mmlu,gsm8k}"
BENCH_TOOL="${BENCH_TOOL:-lm_eval}"

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
    # support quant only on xpu for now
    uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/xpu
    uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/xpu
elif [[ "$DEVICE" == "gpu" ]]; then
    uv pip install -r requirements.txt
    uv pip install setuptools --upgrade
    uv pip install packaging --upgrade
    uv pip install -U "huggingface_hub[cli]"
    if [[ "$FORMAT" == "LLMC" ]]; then
        uv pip install ray
        # use official vllm after PR merge, https://github.com/vllm-project/vllm/pull/42916
        git clone -b fp8-attn-rebase --single-branch --quiet https://github.com/yiliu30/vllm-fork.git && cd vllm-fork
        git fetch --tags
        VLLM_USE_PRECOMPILED=1 uv pip install --prerelease=allow . -v
        cd ..
        git clone https://github.com/yiliu30/vllm-qdq-plugin.git
        uv pip install vllm-qdq-plugin/ -v
    else
        # use default setting for AR format, required by fused-moe-ar
        uv pip install torch==2.9.0
        git clone -b fused-moe-ar  --single-branch --quiet https://github.com/yiliu30/vllm-fork.git && cd vllm-fork
        git fetch --tags
        VLLM_USE_PRECOMPILED=1 uv pip install --prerelease=allow . -v
        cd ..
    fi
    if [[ "$BENCH_TOOL" == "lm_eval" ]]; then
        uv pip install lm-eval==0.4.10
        uv pip install lm-eval[api]
        if [[ "$TASKS" == *"longbench"* ]]; then
            uv pip install "long-bench-eval @ git+https://github.com/yiliu30/long-bench-eval"
        fi
        if [[ "$TASKS" == *"ruler"* ]]; then
            uv pip install lm_eval["ruler"]
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