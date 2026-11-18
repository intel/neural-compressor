# Environment and installation notes

[Overview](../README.md) · [T2V](t2v.md) · [I2V](i2v.md) · [Sanitized package inventory](../requirements-eval.txt)

## What was actually measured

The recorded runs used **Linux, NVIDIA B200, driver 580.178.04**, and a dedicated Python 3.12.3 virtual environment. This was **not a container and not a clean, self-contained installation**. Its `.pth` overlays searched a newer inference environment before a shared base environment, and selected the official Omni checkout through `PYTHONPATH`. Base packages remained visible. A clean install of this stack has not been tested; neither the setup helper nor the inventory below is a tested environment lock.

Runtime import checks on 2026-11-12, after sourcing the historical `activate_main.sh`, confirmed:

| Component | Actual runtime version | Import / resolved source role |
|---|---|---|
| Python | 3.12.3 | Dedicated benchmark venv |
| PyTorch | `2.14.0+cu130` | `torch`; newer inference overlay |
| Torchvision distribution | `0.29.0` | Newer inference overlay |
| vLLM | `0.33.0` | `vllm`; newer inference overlay |
| vLLM-Omni source | `7465c0f52bcb46a55f62c1b8613846cda30d0083` | Selected checkout's `vllm_omni/__init__.py` |
| Diffusers | `0.40.0` | `diffusers`; shared base |
| Transformers distribution | `5.20.2` | Newer inference overlay |
| FlashInfer | `0.7.2.post1` | Distribution `flashinfer-python`, import `flashinfer`; newer overlay |
| CUTLASS DSL | `4.8.0` | Distribution `nvidia-cutlass-dsl`, import `cutlass`; nested `nvidia_cutlass_dsl/dsl_packages/cutlass` |
| VBench distribution | `0.1.5` | `vbench` and `vbench2_beta_i2v`; shared base |
| DreamSim distribution | `0.2.1` | `dreamsim`; shared base |
| PyIQA | `0.1.16` | `pyiqa`; shared base |
| CUDA compiler selected by shell | `12.8.93` | `nvcc` from CUDA toolkit 12.8 |
| CUDA compiled into PyTorch | `13.0` | `torch.version.cuda` |
| FFmpeg / ffprobe | `7.0.2-static` | Executables on `PATH` |

The toolkit compiler and PyTorch's compiled CUDA version are **different facts**. The activation script selected CUDA toolkit 12.8 for FlashInfer JIT compilation, while the Torch/vLLM binaries use CUDA 13.0. Do not describe this as an all-CUDA-12.8 installation. A suitable CUDA 13 driver is still needed. The older system `nvcc` rejected `--compress-mode=size`; ensure the intended compiler wins on `PATH`.

### Source identity beats stale package metadata

`git rev-parse HEAD` of the selected Omni checkout returned the full commit above, with no tracked changes reported. `vllm_omni.__file__` resolved inside that checkout. Its runtime version fell back to `dev` because `_version.py` was absent. Meanwhile, installed distribution metadata reported **`vllm-omni==0.1.dev3005+gad93bf2fc`**, belonging to an older editable source. **That metadata is not the source revision used for these runs.**

Other duplicate distributions were visible too: Torch 2.13.0, vLLM 0.29.0, FlashInfer 0.6.18, CUTLASS DSL 4.6.2, and Transformers 5.14.1 from the shared base. The newer overlay won at import time. Do not select versions by listing only the base venv, or treat a plain package freeze as proof of import identity.

The sanitized [full visible-distribution freeze](../requirements-eval.txt) retains the first selected metadata entry per package, including unrelated packages inherited from the base. Source/editable entries are comments with their observed versions; local URLs and personal paths have been removed. Torch metadata freezes as `2.14.0`, although the imported module reports `2.14.0+cu130`. It is an inventory, **not a minimal evaluation dependency set, ABI lock, or guarantee that all listed versions resolve publicly**. Its AutoRound 0.16.0 metadata is not the historical export revision; [export provenance](quantization.md) is separate.

## Installation: use upstream instructions, then validate

Consult the pinned checkout rather than installing arbitrary current releases:

- [Official quickstart](https://github.com/vllm-project/vllm-omni/blob/7465c0f52bcb46a55f62c1b8613846cda30d0083/docs/getting_started/quickstart.md)
- [CUDA installation/source wheel instructions](https://github.com/vllm-project/vllm-omni/blob/7465c0f52bcb46a55f62c1b8613846cda30d0083/docs/getting_started/installation/gpu/cuda.inc.md)
- [CUDA requirements](https://github.com/vllm-project/vllm-omni/blob/7465c0f52bcb46a55f62c1b8613846cda30d0083/requirements/cuda.txt), [common requirements](https://github.com/vllm-project/vllm-omni/blob/7465c0f52bcb46a55f62c1b8613846cda30d0083/requirements/common.txt), and [setup dependency routing](https://github.com/vllm-project/vllm-omni/blob/7465c0f52bcb46a55f62c1b8613846cda30d0083/setup.py)
- [vLLM GPU installation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
- [VBench installation and evaluator checkpoint instructions](https://github.com/Vchitect/VBench)

The pinned Omni source instructions target vLLM **0.33.0**, source commit `4ba787ec9959e566d3b587a796dff07673324da5`, and its CUDA 13.0 wheel. The release-wheel examples elsewhere in that checkout use matching **0.32.0** releases; those are not this measured 0.33.0 stack. `setup.py` selects hardware-specific requirements, and `requirements/cuda.txt` includes the common requirements rather than pinning the entire Torch/vLLM binary stack.

There is a real dependency mismatch to resolve before claiming a clean install: the pinned common requirements specify Transformers `>=5.16.1,<5.20.0`, while the observed overlay contains **5.20.2**. Do not silently relax the upstream constraint and call the result an exact reproduction. A normal dependency install may select a different version; an advanced `--no-deps` install can retain the measured one but bypasses resolver guarantees.

### Practical starting point, not a validated clean-install recipe

Create a Python 3.12 environment and pin the source. These commands illustrate the installation sequence, not a claim that today's public indexes reproduce the overlay:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
git clone https://github.com/vllm-project/vllm-omni.git /path/to/vllm-omni
export OMNI_SOURCE=/path/to/vllm-omni
git -C "$OMNI_SOURCE" checkout 7465c0f52bcb46a55f62c1b8613846cda30d0083

# Use the official CUDA 13.0 index for the Torch binary pair.
python -m pip install --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.14.0 torchvision==0.29.0
# This wheel URL is supplied by the pinned Omni source instructions.
python -m pip install \
  'https://wheels.vllm.ai/4ba787ec9959e566d3b587a796dff07673324da5/vllm-0.33.0-cp38-abi3-manylinux_2_28_x86_64.whl'
VLLM_OMNI_TARGET_DEVICE=cuda python -m pip install -e "$OMNI_SOURCE"
python -m pip install vbench==0.1.5 dreamsim==0.2.1 pyiqa==0.1.16 gdown==4.7.3
python -m pip check
```

The official `https://download.pytorch.org/whl/cu130/` index returned HTTP 200 during inspection; **individual wheel availability and a complete resolver install were not tested**. Custom/nightly builds or exact versions in the recorded freeze may not resolve from public indexes. If a pinned wheel is unavailable, obtain a compatible upstream build or follow upstream source-build instructions; do not silently substitute another CUDA/Torch/vLLM combination. Install the upstream requirements needed by the selected metrics, including CLIP, DINO, MUSIQ, AMT, ViCLIP, and DreamSim checkpoint dependencies. `ffmpeg`, `ffprobe`, and `unzip` are external executables, not provided by a Python freeze.

For an **already provisioned and ABI-compatible** Torch/vLLM environment, an expert may install the Omni source with `VLLM_OMNI_TARGET_DEVICE=cuda python -m pip install --no-deps -e "$OMNI_SOURCE"`. Likewise, a local matching vLLM wheel can be installed with `--no-deps` only after manually satisfying its requirements. This prevents unwanted dependency replacement but does **not** solve compatibility or missing packages. Review upstream requirements and run `pip check`, imports, and a real generation smoke afterward. Do not install the full inventory blindly over a working GPU stack.

The optional setup helper takes `bash scripts/setup_env.sh ENV_DIR OMNI_SOURCE`. Inspect its actions and finish with the checks below. It is a convenience helper, not evidence that a clean installation has passed.

## Activate and preflight

The measured host used a private `activate_main.sh` to activate the benchmark interpreter, select the official checkout, put `ninja`/FFmpeg on `PATH`, and export toolkit/backend settings. That host-specific file is not distributed here. Its portable equivalent, **after installing dependencies**, is:

```bash
source .venv/bin/activate
export OMNI_SOURCE=/path/to/vllm-omni
export PYTHONPATH="$OMNI_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export DIFFUSION_ATTENTION_BACKEND=TRTLLM_ATTN
export PYTHONUNBUFFERED=1
command -v ninja ffmpeg ffprobe nvcc
nvcc --version
git -C "$OMNI_SOURCE" rev-parse HEAD
python scripts/benchmark.py environment --task t2v --root runs/preflight \
  --omni-source "$OMNI_SOURCE"
```

Inspect the environment report's source paths, versions, CUDA build, compiler, and tools before generation. A path from another Omni checkout is a failure even if its package name is correct. Activation matters: using only an absolute Python executable can leave `ninja` off `PATH` and break FlashInfer JIT. Set `CUDA_VISIBLE_DEVICES` explicitly for each worker or evaluation command; a host activation script's default GPU must not select devices accidentally.

CUTLASS DSL needs special care in an overlay. Adding a parent site-packages directory to `sys.path` does not automatically process that directory's nested `.pth` files. In the measured environment, an additional `.pth` exposed `nvidia_cutlass_dsl/dsl_packages`, allowing **`import cutlass`**. Do not test `import nvidia_cutlass_dsl` as a substitute. A regular install should configure its own paths; never copy another machine's absolute `.pth` entries.

Check `torch.__version__`, `torch.version.cuda`, `vllm.__version__`, `diffusers.__version__`, `flashinfer.__version__`, `cutlass.__version__`, and the modules' `__file__` locations. Distribution versions alone are insufficient in an overlay. T2V MXFP8 uses a CuTe-DSL activation quantizer even with a CUTLASS matrix-multiply backend, so a successful `flashinfer` import alone does not establish kernel readiness.

Finish with a **separate smoke root** (`prepare`, then `generate --limit 1`) for each checkpoint/precision. Check the log for actual kernel/backend selection and verify the produced video's dimensions, frames, and fps. Imports do not prove kernel execution; one successful video does not prove a complete benchmark. Full evaluation requires all 165 T2V or 355 I2V jobs and the correct per-metric counts.

## Downloads and publication hygiene

Model weights, evaluator checkpoints, cropped reference images, and generated videos remain external and subject to their own licenses. First evaluation may need network access for model downloads. Provision caches according to upstream VBench/DreamSim instructions rather than assuming a single cache environment variable covers every metric. Never commit credentials, private index URLs, local editable paths, weight files, or generated media. The checked-in inventory contains no host-specific install URLs.

See [evaluator checkpoints and first-run downloads](evaluator-checkpoints.md) for
the metric-to-model mapping, known DINO/CLIP checksums, AMT compatibility scope,
and the portable runner's DreamSim cache location.
