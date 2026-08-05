# Agent Benchmarks

Scripts for serving models with vLLM and running agent benchmarks against its OpenAI-compatible API.

## vLLM environment setup

Run the setup script inside an existing uv, Conda, or Docker environment that provides Python and supports `uv pip install`. The script does not create or activate an environment.

```bash
# Standard vLLM
bash setup_vllm.sh

# DeepSeek-V4-compatible vLLM
bash setup_vllm.sh --model /path/to/DeepSeek-V4
```

A model name or path containing `deepseek-v4` selects the
pinned DeepSeek-V4 vLLM fork and installs DeepGEMM. All other names select the pinned standard vLLM release now.

## Start the vLLM server

```bash
CUDA_VISIBLE_DEVICES=0 bash start_vllm_serve.sh MODEL [VLLM_OPTIONS...]
```

The server runs in the foreground. Arguments after `MODEL` are passed directly to `vllm serve`. 

Common settings:

- Port: `8888`, unless the user passes `--port`
- Served model name: `gpt-3.5-turbo`
- `--trust-remote-code`
- `--enable-auto-tool-choice` for every model

For example:

`--tool-call-parser` must match the model's tool-call format so vLLM can return structured `tool_calls`. 
A wrong parser may cause `RepeatedFormatError`.
The default is `hermes`; Qwen3.5/Qwen3.6 requires `qwen3_xml`:

```bash
CUDA_VISIBLE_DEVICES=0 bash start_vllm_serve.sh \
    Qwen3/Qwen3.6-27B \
    --tool-call-parser qwen3_xml
```


## SWE-Verified

### SWE-Verified environment setup

```bash
bash setup_swe_verified.sh
```

The setup script:

1. Clones mini-SWE-agent v2.4.6 into `mini-swe-agent/`.
2. Applies `patches/swebench_verified_per_instance_cleanup.patch` so each instance removes its Docker container and image during cleanup.
3. Installs mini-SWE-agent to generate predictions and the official SWE-bench harness to evaluate those predictions locally. It also installs datasets for loading SWE-bench Verified.


### Run SWE-Verified

```bash
bash run_swe_verified.sh \
  --port 8888 \
  --workers 2 \
  --eval-workers 2 \
  --step-limit 250 \
  --tag qwen36_27b_full
```
| Option | Default | Description |
| --- | --- | --- |
| `--host HOST` | `127.0.0.1` | vLLM host |
| `--port PORT` | `8888` | vLLM port |
| `--served-name NAME` | discovered | Model ID exposed by vLLM |
| `--num-tasks N` | all | Run the first N instances |
| `--slice START:END` | all | Run an explicit slice; cannot be combined with `--num-tasks` |
| `--workers N` | `2` | Parallel mini-SWE-agent workers |
| `--eval-workers N` | agent workers | Parallel local harness workers |
| `--step-limit N` | `250` | Maximum model calls per instance |
| `--tag TAG` | UTC timestamp | Output and log label |
| `--skip-eval` | disabled | Generate predictions without local evaluation |

Outputs:

- mini-SWE-agent results: `mini-swe-agent/results/swe_verified_<TAG>/`
- Harness predictions: `mini-swe-agent/results/swe_verified_<TAG>/preds.jsonl`
- Local evaluation report: `<SERVED_MODEL_NAME>.<TAG>.json`
- Log: `logs/swe_verified_<TAG>.log`
