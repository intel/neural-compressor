#!/usr/bin/env bash
# Bootstrap, not a validated clean-install lock. Never install requirements-eval.txt.
set -euo pipefail
usage() {
    echo "Usage: bash $0 [--check|--install] ENV_DIR OMNI_SOURCE"
    echo "Default: --check (no downloads or changes). --install creates a Python 3.12 venv."
    echo "Requires an explicitly selected CUDA_HOME, CUDA-13-capable driver, git, ffmpeg and ffprobe."
}
if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
mode=--check
if [[ ${1:-} == --check || ${1:-} == --install ]]; then mode=$1; shift; fi
if [[ $# != 2 ]]; then usage >&2; exit 2; fi
fail() { echo "ERROR: $*" >&2; exit 1; }
trap 'echo "Setup failed; no alternate versions were substituted. See docs/environment.md and pinned upstream build instructions. Partial installs may remain in ENV_DIR; no clean-install validation is claimed." >&2' ERR
command -v git >/dev/null || fail "git is required"
env_dir=$(realpath -m -- "$1")
omni_source=$(realpath -m -- "$2")
commit=7465c0f52bcb46a55f62c1b8613846cda30d0083
[[ "$env_dir" != "$omni_source" && "$env_dir" != / ]] || fail "Use separate environment and source directories"

if [[ ! -e "$omni_source" ]]; then
    [[ "$mode" == --install ]] || fail "Omni checkout missing; use --install to clone it"
    git clone https://github.com/vllm-project/vllm-omni.git "$omni_source"
    git -C "$omni_source" checkout --detach "$commit"
fi
[[ $(git -C "$omni_source" rev-parse --show-toplevel) == "$omni_source" ]] || fail "OMNI_SOURCE must be the checkout root"
[[ $(git -C "$omni_source" rev-parse HEAD) == "$commit" ]] || fail "Omni commit mismatch; existing checkouts are never reset"
[[ -z $(git -C "$omni_source" status --porcelain --untracked-files=no) ]] || fail "Omni has tracked changes; existing checkouts are never cleaned"
[[ -f "$omni_source/requirements/cuda.txt" ]] || fail "Pinned CUDA requirements missing"

[[ -n ${CUDA_HOME:-} && -x "$CUDA_HOME/bin/nvcc" ]] || fail "Set CUDA_HOME to the intended toolkit with bin/nvcc (historical compiler: 12.8.93, Torch CUDA: 13.0)"
# Changes affect only this script, not the parent shell or host configuration.
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
unset PYTHONPATH PYTHONHOME
for tool in ffmpeg ffprobe; do command -v "$tool" >/dev/null || fail "$tool is required on PATH; install host tools manually"; done

if [[ ! -e "$env_dir" ]]; then
    [[ "$mode" == --install ]] || fail "Environment missing; use --install to create it"
    if command -v python3.12 >/dev/null; then
        python3.12 -m venv "$env_dir"
    elif command -v uv >/dev/null; then
        uv venv --python 3.12 "$env_dir"
    else
        fail "Python 3.12 or uv is required; no system Python will be changed"
    fi
fi
[[ -f "$env_dir/pyvenv.cfg" && -x "$env_dir/bin/python" ]] || fail "ENV_DIR must be a dedicated Python 3.12 venv"
py="$env_dir/bin/python"
export PATH="$env_dir/bin:$PATH"
export PYTHONNOUSERSITE=1
"$py" -I - "$env_dir" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
assert sys.version_info[:2] == (3, 12), 'Python 3.12 is required'
assert Path(sys.prefix).resolve() == root, 'Interpreter is outside ENV_DIR'
assert sys.prefix != sys.base_prefix, 'Refusing a global interpreter'
cfg = (root / 'pyvenv.cfg').read_text().lower().replace(' ', '')
assert 'include-system-site-packages=false' in cfg, 'System-site-packages overlays are not supported'
PY

if [[ "$mode" == --install ]]; then
    echo "Best-effort bootstrap, NOT an exact reproduction of the historical overlay."
    echo "Using upstream Transformers >=5.16.1,<5.20.0, not observed 5.20.2 (an upstream constraint mismatch)."
    constraints=$(mktemp)
    trap 'rm -f -- "$constraints"' EXIT
    cat > "$constraints" <<'PINS'
torch==2.14.0
torchvision==0.29.0
vllm==0.33.0
diffusers==0.40.0
flashinfer-python==0.7.2.post1
nvidia-cutlass-dsl==4.8.0
transformers>=5.16.1,<5.20.0
vbench==0.1.5
dreamsim==0.2.1
pyiqa==0.1.16
gdown==4.7.3
PINS
    # PIP_CONSTRAINT also protects dependency installs invoked by build backends.
    export PIP_CONSTRAINT="$constraints"
    "$py" -m ensurepip --upgrade
    "$py" -m pip install --upgrade pip setuptools wheel
    "$py" -m pip install --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.14.0 torchvision==0.29.0
    "$py" -m pip install \
        https://wheels.vllm.ai/4ba787ec9959e566d3b587a796dff07673324da5/vllm-0.33.0-cp38-abi3-manylinux_2_28_x86_64.whl
    # Resolve upstream CUDA/common dependencies under the binary-stack constraints.
    # Unavailable wheels or conflicting pins are hard failures, not permission to upgrade Torch.
    "$py" -m pip install -r "$omni_source/requirements/cuda.txt" \
        diffusers==0.40.0 flashinfer-python==0.7.2.post1 nvidia-cutlass-dsl==4.8.0 \
        'transformers>=5.16.1,<5.20.0' ninja \
        vbench==0.1.5 dreamsim==0.2.1 pyiqa==0.1.16 gdown==4.7.3
    VLLM_OMNI_TARGET_DEVICE=cuda "$py" -m pip install --no-deps -e "$omni_source"
fi

"$py" -m pip check
command -v ninja >/dev/null || fail "ninja must be on the environment PATH"
nvcc --version
"$py" -I - "$omni_source" <<'PY'
import importlib
from importlib.metadata import version
from pathlib import Path
import sys
expected = {
    'torch': '2.14.0', 'torchvision': '0.29.0', 'vllm': '0.33.0',
    'diffusers': '0.40.0', 'flashinfer-python': '0.7.2.post1',
    'nvidia-cutlass-dsl': '4.8.0', 'vbench': '0.1.5',
    'dreamsim': '0.2.1', 'pyiqa': '0.1.16', 'gdown': '4.7.3',
}
for name, wanted in expected.items():
    actual = version(name)
    assert actual.split('+')[0] == wanted, f'{name}: expected {wanted}, got {actual}'
    print(f'{name} distribution: {actual}')
for name, wanted in (('torch', '2.14.0'), ('vllm', '0.33.0'),
                     ('diffusers', '0.40.0'), ('flashinfer', '0.7.2.post1'),
                     ('cutlass', '4.8.0')):
    module = importlib.import_module(name)
    actual = str(module.__version__)
    assert actual.split('+')[0] == wanted, f'{name} import mismatch: {actual}'
    print(f'{name} import: {actual} {module.__file__}')
import torch
assert torch.version.cuda == '13.0', f'Torch CUDA mismatch: {torch.version.cuda}'
import transformers
from packaging.version import Version
assert Version('5.16.1') <= Version(transformers.__version__) < Version('5.20.0'), 'Upstream Transformers constraint mismatch (observed 5.20.2 is not accepted)'
for name in ('torchvision', 'vllm_omni', 'vbench', 'vbench2_beta_i2v', 'dreamsim', 'pyiqa', 'gdown'):
    module = importlib.import_module(name)
    print(f'{name} import: {module.__file__}')
    if name == 'vllm_omni':
        assert Path(module.__file__).resolve().is_relative_to(Path(sys.argv[1]) / name), 'Omni imported from another checkout'
PY
echo "Package/source checks passed only; CUDA kernels, checkpoints and generation remain UNVALIDATED."
echo "Activate ENV_DIR, select CUDA_HOME and Omni as in docs/environment.md; run environment preflight and a separate one-video smoke before benchmarking."
echo "Metric checkpoints (CLIP, DINO, MUSIQ, AMT, ViCLIP, DreamSim) may require upstream provisioning/network access."