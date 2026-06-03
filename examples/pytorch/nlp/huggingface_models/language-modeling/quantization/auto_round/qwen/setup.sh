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
TASKS="${TASKS:-hellaswag,piqa,mmlu,gsm8k,ruler}"
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
        CUDA_VERSION=$(detect_cuda_version)
        echo "Detected system CUDA version: $CUDA_VERSION"
        if [[ "$CUDA_VERSION" == "12."* ]]; then
            uv pip install vllm==0.22.0 --extra-index-url https://wheels.vllm.ai/0.22.0/cu129 --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match
            uv pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match
        elif [[ "$CUDA_VERSION" == "13."* ]]; then
            uv pip install vllm==0.22.0
        else
            echo "Unsupported CUDA version: $CUDA_VERSION. Supported versions are 12.x and 13.x."
            exit 1
        fi

        uv pip install ray
        git clone https://github.com/yiliu30/vllm-qdq-plugin.git
        uv pip install vllm-qdq-plugin/ -v
    else
        # use default setting for AR format, required by fused-moe-ar
        uv pip install torch==2.9.0
        git clone -b fused-moe-ar  --single-branch --quiet https://github.com/yiliu30/vllm-fork.git && cd vllm-fork
        VLLM_USE_PRECOMPILED=1 uv pip install --prerelease=allow . -v
        cd ..
    fi
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