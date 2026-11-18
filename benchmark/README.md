This folder is for benchmarking code: accuracy evaluation of (quantized) LLMs served by vLLM, agent benchmarks and
vLLM plugins used for evaluation. The per-model wrappers under `examples/pytorch/llm/<model>/run_benchmark.sh` route
tasks to the shared drivers below.

| Directory | Entry point | What it evaluates |
| --- | --- | --- |
| [`lm_eval/`](lm_eval) | `run_lm_eval.sh` | [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) tasks with the vLLM backend |
| [`evalscope/`](evalscope) | `run_evalscope.sh` | [evalscope](https://github.com/modelscope/evalscope) datasets through `vllm serve` (OpenAI-compatible API) |
| [`ifbench/`](ifbench) | `run_ifbench.sh` | [IFBench](https://github.com/allenai/IFBench) instruction following against an OpenAI-compatible endpoint |
| [`agent_benchmarks/`](agent_benchmarks) | `run_swe_verified.sh`, `run_swebenchpro.sh`, `run_mcp_atlas.sh` | agentic coding and tool-use benchmarks |
| [`vllm-qdq-plugin/`](vllm-qdq-plugin) | `pip install -e benchmark/vllm-qdq-plugin` | vLLM plugin for quant-dequant (QDQ) simulation and Sage3 attention |

Both shared drivers take `--key=value` options, accept a local model directory or a Hugging Face model id for
`--model_path`, and infer the tensor parallel size from `CUDA_VISIBLE_DEVICES`. Run a driver with `--help` for all
options.

## lm_eval

`lm_eval/run_lm_eval.sh` runs lm_eval tasks and routes them by name:

- standard tasks (e.g. `hellaswag`, `piqa`, `mmlu`, `gsm8k`): lm_eval `vllm` backend;
- chat-template tasks (`gsm8k_llama`, `mmlu_llama`);
- long-context RULER tasks (`*ruler*`, `niah_*`, `--ruler_max_pos`) and LongBench tasks (`*longbench*`): evaluated
  against a `vllm serve` endpoint.

```bash
CUDA_VISIBLE_DEVICES=0,1 bash lm_eval/run_lm_eval.sh --model_path=<model> --tasks=hellaswag,piqa,mmlu,gsm8k
```

## evalscope

`evalscope/run_evalscope.sh` starts `vllm serve`, evaluates evalscope datasets (e.g. `aime26`, `gpqa_diamond`,
`math_500`, `live_code_bench`, `openai_mrcr`) `--num_runs` times and summarizes the scores with `eval_summary.py`
(the highest and lowest scores are dropped when there are 3 or more runs).

```bash
CUDA_VISIBLE_DEVICES=0,1 bash evalscope/run_evalscope.sh --model_path=<model> --tasks=aime26,gpqa_diamond --num_runs=5
```

### OpenAI MRCR (`openai_mrcr`)

[OpenAI MRCR](https://huggingface.co/datasets/openai/mrcr) measures multi-round coreference resolution over long
conversations: the model must reproduce the n-th instance of a request hidden among 2, 4 or 8 similar ones
("needles"). It runs through evalscope's native `openai_mrcr` benchmark in `evalscope/run_evalscope.sh`; the score is
the `difflib.SequenceMatcher` ratio between answer and reference, and 0 when the required random prefix is missing.

Every model evaluates the same fixed 300 samples, listed in `evalscope/mrcr/openai_mrcr_300.json`:

- dataset `openai/mrcr` pinned to revision `f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d`;
- 100 samples each for 2, 4 and 8 needles, drawn with `random.Random(42 + needles)`;
- 4096 to 112000 `o200k_base` tokens (prompt + answer), so a prompt plus 2048 generated tokens fits in a 131072
  context even for tokenizers that need ~12% more tokens.

Before evaluating, `evalscope/mrcr/prepare_mrcr_subset.py` downloads the referenced shards, verifies the sha256 of
every sample and writes `2needle_train.jsonl`, `4needle_train.jsonl` and `8needle_train.jsonl` to `MRCR_DATA_DIR`
(default `~/.cache/inc_benchmark/openai_mrcr_300`). `--build_manifest` regenerates the manifest.

`run_evalscope.sh` evaluates `openai_mrcr` once by default (it is in `SINGLE_RUN_TASKS`; the MiniMax wrapper repeats
it `--num_runs` times because it samples with thinking) with temperature 0, `max_tokens=2048` and thinking disabled
(`MRCR_GENERATION_CONFIG`) and prints the scores per needle count and per token-length bin. The vLLM
`--max-model-len` is 131072 unless `--max_model_len` is passed or `MRCR_MAX_MODEL_LEN` is set. Requires `evalscope` and `tiktoken`.

MRCR is only meant for models with native long-context support (at least 131072 tokens): Kimi / GLM
(`examples/pytorch/llm/kimi-glm`), DeepSeek-V4 (`examples/pytorch/llm/deepseekv4`) and MiniMax
(`examples/pytorch/llm/minimax`).

## IFBench

`ifbench/run_ifbench.sh` clones upstream IFBench, overlays `evaluate_model.py` and `configs/config.py`, and generates
and scores responses against any OpenAI-compatible endpoint with parameters aligned to the NVIDIA Nemotron 3 Ultra
IFBench recipe. See [ifbench/README.md](ifbench/README.md).

## Agent benchmarks

`agent_benchmarks/` serves the model with `start_vllm_serve.sh` (after `setup_vllm.sh`) and runs agent benchmarks
against its OpenAI-compatible API. Each benchmark has a `setup_*.sh` and a `run_*.sh` script; pinned versions are in
`versions.env`. See [agent_benchmarks/README.md](agent_benchmarks/README.md).

- SWE-Verified / SWE-Verified Mini: `setup_swe_verified.sh`, `run_swe_verified.sh`
- SWE-bench Pro: `setup_swebenchpro.sh`, `run_swebenchpro.sh`
- MCP-Atlas: `setup_mcp_atlas.sh`, `run_mcp_atlas.sh`

## vLLM QDQ plugin

`vllm-qdq-plugin/` is an out-of-tree vLLM plugin that simulates activation quant-dequant (QDQ), so quantized
formats can be evaluated on GPUs without native kernels, plus a Sage3 Triton diffusion attention backend. The
shared drivers enable it with `VLLM_QDQ=1` on GPUs other than SM 10.x. See
[vllm-qdq-plugin/README.md](vllm-qdq-plugin/README.md).
