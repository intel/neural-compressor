This folder is for benchmarking code.

## OpenAI MRCR (`openai_mrcr`)

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

`run_evalscope.sh` evaluates `openai_mrcr` once with temperature 0, `max_tokens=2048` and thinking disabled
(`MRCR_GENERATION_CONFIG`) and prints the scores per needle count and per token-length bin. The vLLM
`--max-model-len` is 131072 unless `--max_model_len` is passed or `MRCR_MAX_MODEL_LEN` is set. Requires `evalscope` and `tiktoken`.

MRCR is only meant for models with native long-context support (at least 131072 tokens): Kimi / GLM
(`examples/pytorch/llm/kimi-glm`), DeepSeek-V4 (`examples/pytorch/llm/deepseekv4`) and MiniMax
(`examples/pytorch/llm/minimax`).
