# Agent Benchmark Runner

`run_agent_v2.sh` 是统一的 coding-agent benchmark 评测脚本，支持三个主流基准：

| Benchmark | 描述 | Agent 版本 | 评测方式 |
|-----------|------|-----------|----------|
| **SWE-bench Pro** | Scale AI 构建的 GitHub issue 修复基准（731 tasks） | mini-swe-agent v1.15.0 (commit `d74716a`) | 本地 Docker |
| **SWE-bench Verified** | Princeton 500 个人工验证 issue | mini-swe-agent v2.4.5 (main) | 云端 sb-cli（免费） |
| **MCP-Atlas** | Scale AI 构建的 MCP tool-use 基准（500 tasks） | — | LLM-as-judge 本地 |

---

## 环境配置（BKC）

| 项目 | SWE-bench Pro | SWE-bench Verified |
|------|---------------|-------------------|
| GPU | `CUDA_VISIBLE_DEVICES=2` | `CUDA_VISIBLE_DEVICES=0` |
| vLLM port | `8880` | `8881` |
| Workers | `8` | `4` |
| Step limit | `250` | `250` |
| venv | `.venv-swebp` | `.venv-swe` |

**Model**：`/storage/changwa1/swe/Qwen3-30B-A3B-Instruct-2507`

---

## 快速开始

```bash
BD=/storage/changwa1/swe_pro/neural-compressor/examples/pytorch/nlp/huggingface_models/language-modeling/quantization/auto_round/deepseekv4
cd $BD

# 首次 setup（每台机器只需跑一次）
BENCHMARK_DIR=$BD bash setup_agent.sh swebp
BENCHMARK_DIR=$BD bash setup_agent.sh swe-verified

# SWE-bench Pro smoke（10 instances）
CUDA_VISIBLE_DEVICES=2 bash run_agent_v2.sh \
    --task swebp --instances 10test \
    --port 8880 --model /storage/changwa1/swe/Qwen3-30B-A3B-Instruct-2507 \
    --workers 2 --step-limit 100 --tag smoke_swebp

# SWE-bench Verified smoke（10 instances）
CUDA_VISIBLE_DEVICES=0 \
SWEBENCH_API_KEY=<your_key> \
bash run_agent_v2.sh \
    --task swe-verified --num-tasks 10 \
    --port 8881 --model /storage/changwa1/swe/Qwen3-30B-A3B-Instruct-2507 \
    --workers 2 --step-limit 30 --tag smoke_verified
```

---

## 首次 Setup

`setup_agent.sh` 负责所有环境准备，**`run_agent_v2.sh` 不含 setup 逻辑**。

```bash
BENCHMARK_DIR=$BD bash setup_agent.sh [swebp|swe-verified|mcp-atlas|all]
```

| Task | 内容 |
|------|------|
| `swebp` | clone `SWE-bench_Pro-os` + submodule (v1.15.0)，写 `run_swebp.py` 和 `config/swebp_vllm.yaml`，下载 731 instances，创建 `.venv-swebp` |
| `swe-verified` | clone `mini-swe-agent` main (v2.4.5)，创建 `.venv-swe` |
| `mcp-atlas` | clone `mcp-atlas`，创建 `.venv-mcp`，npm install，pull Docker 镜像，生成 `.env` |

> **注意**：setup 不装 `ninja`（会触发 flashinfer 编译报错）。

---

## 所有参数

### 通用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--task TASK` | `swebp` | `swebp` \| `swe-verified` \| `mcp-atlas` |
| `--port N` | `8888` | vLLM API 监听端口 |
| `--model PATH` | _(空)_ | 模型路径；**传入时自动启动 vLLM** |
| `--max-model-len N` | `262144` | vLLM `--max-model-len` |
| `--served-name NAME` | `gpt-3.5-turbo` | vLLM `--served-model-name` |
| `--tag NAME` | `YYYYMMDD_HHMMSS` | 运行标签，区分输出目录 |
| `--skip-serve` | _(false)_ | 跳过 vLLM 就绪检测（vLLM 已在运行时用） |

### SWE-bench Pro 专用（`--task swebp`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--instances KEY` | `full` | `full`（731）\| `10test`（10）\| `/path/to/file.json` |
| `--slice S:E` | _(全量)_ | 实例切片，如 `0:50` |
| `--workers N` | `2` | 并行 agent workers |
| `--step-limit N` | `250` | 每个实例最大 agent 步数 |
| `--redo` | _(false)_ | 重新运行已完成的实例 |

### SWE-bench Verified 专用（`--task swe-verified`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-tasks N` | _(全量 500)_ | 限制运行数量（取前 N 条） |
| `--slice S:E` | _(全量)_ | 与 `--num-tasks` 二选一 |
| `--workers N` | `2` | 并行 agent workers |
| `--step-limit N` | `250` | 每个实例最大 agent 步数 |

### MCP-Atlas 专用（`--task mcp-atlas`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-tasks N` | _(全量 500)_ | 限制运行数量 |
| `--concurrency N` | `5` | 并行评测 workers |
| `--sandbox-port P` | `1984` | MCP sandbox Docker 端口 |
| `--harness-port P` | `3001` | Agent harness 端口 |

---

## 各 Benchmark 详细说明

### SWE-bench Pro

- **Agent**：mini-swe-agent **v1.15.0**（`SWE-bench_Pro-os/mini-swe-agent`，commit `d74716a`）
- **入口**：`python3 run_swebp.py`（`mini-extra run-batch` API 已废弃）
- **Docker 镜像**：`jefzda/sweap-images:{dockerhub_tag}`
- **评测**：本地 Docker，`swe_bench_pro_eval.py`
- **输出**：`SWE-bench_Pro-os/mini-swe-agent/results/run_<tag>/preds.json`

```bash
# Full run（731 instances）
CUDA_VISIBLE_DEVICES=2 bash run_agent_v2.sh \
    --task swebp --instances full \
    --port 8880 --model /storage/changwa1/swe/Qwen3-30B-A3B-Instruct-2507 \
    --workers 8 --step-limit 250 --tag swebp_full

# 断点续跑
CUDA_VISIBLE_DEVICES=2 bash run_agent_v2.sh \
    --task swebp --instances full --redo \
    --port 8880 --skip-serve --tag swebp_full
```

---

### SWE-bench Verified

- **Agent**：mini-swe-agent **v2.4.5**（`mini-swe-agent/`，独立 clone，main branch）
- **入口**：`mini-extra swebench` + `-c key=value` 多配置
- **评测**：云端 `sb-cli submit swe-bench_verified test`（需 `SWEBENCH_API_KEY`）
- **输出**：`mini-swe-agent/results/swe_verified_<tag>/preds.jsonl`

```bash
# Full run（500 instances）
CUDA_VISIBLE_DEVICES=0 \
SWEBENCH_API_KEY=<your_key> \
bash run_agent_v2.sh \
    --task swe-verified \
    --port 8881 --model /storage/changwa1/swe/Qwen3-30B-A3B-Instruct-2507 \
    --workers 4 --step-limit 250 --tag verified_full
```

**获取 SWEBENCH_API_KEY（一次性）：**

```bash
.venv-swe/bin/sb-cli gen-api-key your@email.com
.venv-swe/bin/sb-cli verify-api-key YOUR_CODE
export SWEBENCH_API_KEY=swb_xxx...
```

---

### MCP-Atlas

- **评测**：LLM-as-judge（使用本地 vLLM 作为评判模型）
- **主要指标**：`Pass@0.75`
- **输出**：`mcp-atlas/outputs/run_<tag>/outputs.csv`

```bash
bash run_agent_v2.sh --task mcp-atlas \
    --port 8880 --model /storage/changwa1/swe/Qwen3-30B-A3B-Instruct-2507 \
    --tag mcp_full
```

---

## 目录结构

```
$BD/
├── run_agent_v2.sh              # 统一运行脚本
├── setup_agent.sh               # 一次性环境初始化脚本
├── README.md
├── .venv-swebp/                 # SWE-bench Pro venv (Python 3.10)
├── .venv-swe/                   # SWE-bench Verified venv (Python 3.10)
├── .venv-mcp/                   # MCP-Atlas venv (Python 3.10)
├── logs/                        # vLLM + agent 运行日志
├── SWE-bench_Pro-os/
│   ├── mini-swe-agent/          # v1.15.0 submodule (commit d74716a)
│   │   ├── run_swebp.py         # 由 setup_agent.sh 生成
│   │   ├── config/swebp_vllm.yaml
│   │   └── data/
│   │       ├── swebench_pro_instances.json   # 731 instances
│   │       └── swebp_10instances.json        # 前 10（smoke）
│   └── run_scripts/
├── mini-swe-agent/              # v2.4.5 独立 clone（swe-verified 专用）
└── mcp-atlas/
    ├── .env
    └── outputs/
```

---

## 当前测试结果（Qwen3-30B-A3B-Instruct-2507，2026-07-07）

| Benchmark | 任务数 | 结果 | 备注 |
|-----------|--------|------|------|
| SWE-bench Pro | 10 | 1/10 pass（10%） | smoke，step_limit=100 |
| SWE-bench Verified | — | 流程验证中 | — |
| MCP-Atlas | — | 未开始 | — |
