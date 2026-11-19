# Wan2.2 VBench

Generate and score Wan2.2-A14B I2V and T2V videos with BF16, MXFP8, NVFP4,
or NVFP4 E5M3 checkpoints.

## Environment setup

The GPU tests used NVIDIA B200. Use Linux x86-64, Python 3.12, a CUDA
13 compatible driver, and `uv`, `ffmpeg`, and `ffprobe` on `PATH`. If
FlashInfer cannot find `nvcc`, set `CUDA_HOME` to your toolkit directory.

From the neural-compressor repository root:

```bash
uv venv --seed --python 3.12 .venv-vbench
source .venv-vbench/bin/activate

uv pip install --torch-backend=cu130 vllm==0.32.0 vllm-omni==0.32.0 \
  ninja nvidia-cutlass-dsl==4.8.0 \
  dreamsim==0.2.1 pyiqa==0.1.16 timm==1.0.12 \
  boto3 cython decord easydict fairscale fvcore lvis matplotlib openai-clip \
  opencv-python pycocoevalcap scikit-learn scikit-image tensorboard wheel
uv pip install --no-deps vbench==0.1.5
```

Install VBench with `--no-deps` because its Transformers and NumPy pins
conflict with this inference stack. The command above installs its evaluator
dependencies separately. Have `git` and `wget` on `PATH` for the first
evaluation, when VBench downloads its evaluator models.

For **NVFP4 E5M3**, install the Omni plugin in the same environment.

```bash
uv pip install --no-deps -e benchmark/vllm-qdq-plugin
```

For **I2V**, extract the [VBench image archive](https://drive.google.com/file/d/1Y_JnYnyJ3a6QhiranoX0MQVZFcTDPekZ/view)
and set `I2V_IMAGES` to its `crop/16-9` directory.

## Run

```bash
bash benchmark/vbench/run_vbench.sh generate --task=t2v --model_path=org/model --dtype=mxfp8 --output_dir=runs/t2v
bash benchmark/vbench/run_vbench.sh evaluate --task=t2v --output_dir=runs/t2v
```

For I2V, use `--task=i2v` in both commands and an I2V checkpoint.
`--model_path` accepts a local path or Hugging Face model ID. `--dtype` is
optional and checks the checkpoint precision. Add `--gpus=0,1` to generate
on two GPUs, with one model copy per GPU. Scores are written to
`<output_dir>/evaluation/summary.json`.

T2V uses 165 prompts at 1280 × 720. I2V uses 355 prompt/image pairs at
832 × 480. Both generate one video per input with 81 frames at 16 fps
and score five selected VBench dimensions. Inputs come from
[Vchitect/VBench](https://github.com/Vchitect/VBench) under Apache-2.0.
