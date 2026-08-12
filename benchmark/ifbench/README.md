# IFBench Reproduction

End-to-end reproduction of the [IFBench](https://github.com/allenai/IFBench)
(precise instruction-following) benchmark against any **OpenAI-compatible
endpoint**, using generation + scoring parameters aligned with the
**NVIDIA Nemotron 3 Ultra** IFBench recipe.

The driver clones upstream `allenai/IFBench`, overlays our `evaluate_model.py`
(a one-shot *generate + score* tool that aligns responses to inputs **by key**),
configures the endpoint, and runs the evaluation.

---

## Layout

```
benchmark/ifbench/
├── run_ifbench.sh          # one-shot reproduction driver
├── evaluate_model.py       # end-to-end generate + evaluate (new; not in upstream)
├── README.md               # this file
└── configs/
    ├── config.py           # patched upstream config (Nemotron-aligned defaults)
    └── env.template         # .env template with recommended defaults
```

`run_ifbench.sh` clones IFBench into `$WORKDIR` (default `~/ifbench-run/IFBench`)
and copies `evaluate_model.py` and `configs/config.py` into that clone before
running — the upstream repo is never modified in place, and nothing is committed
here.

---

## Quick start

```bash
cd benchmark/ifbench

# Provide the endpoint under test (or edit the generated .env afterwards):
API_BASE="https://your-endpoint/v1" \
API_KEY="your-key" \
MODEL="your-model-id" \
./run_ifbench.sh

# Smoke test (few prompts, single repeat):
API_BASE=... API_KEY=... MODEL=... ./run_ifbench.sh --limit 5 --num-repeats 1
```

Any extra CLI args after `./run_ifbench.sh` are forwarded verbatim to
`evaluate_model.py` (e.g. `--workers 4`, `--num-repeats 8`, `--no-thinking`).

Results land in `<clone>/eval/`:
- `<model>-results.json` — summary (per-repeat + mean strict/loose accuracy)
- `<model>-eval_results_{strict,loose}-r<i>.jsonl` — per-prompt results
- `<model>-responses-r<i>.jsonl` — generated responses (with `--save-responses`)

---

## What the driver does

1. **Clone** `allenai/IFBench` (shallow) into `$CLONE_DIR` (skipped if present).
2. **Overlay** `evaluate_model.py` (new) and the patched `config.py`.
3. **Environment** — create a venv and install `requirements.txt` plus
   `httpx`, `tqdm`, `pydantic-settings`; download NLTK data (best-effort).
4. **Configure** `.env` from `configs/env.template`; apply `API_BASE` / `API_KEY`
   / `MODEL` overrides from the environment. Fails fast if the endpoint is still
   the placeholder.
5. **Run** `python evaluate_model.py "$@"`.

### Configurable environment variables

| Var | Default | Meaning |
|---|---|---|
| `REPO_URL` | `https://github.com/allenai/IFBench.git` | Upstream repo |
| `REPO_REF` | `main` | Branch/tag to clone |
| `WORKDIR` | `~/ifbench-run` | Clone + venv location |
| `CLONE_DIR` | `$WORKDIR/IFBench` | Clone path |
| `VENV_DIR` | `$WORKDIR/.venv` | Virtualenv path |
| `PYTHON_BIN` | `python3` | Interpreter for the venv |
| `API_BASE`/`API_KEY`/`MODEL` | — | Injected into `.env` |

> On hosts where the system `python3` lacks build tooling, point at a ready
> interpreter, e.g. `PYTHON_BIN=/data/miniforge3/bin/python3 ./run_ifbench.sh`.

---

## How configuration takes effect / precedence

`configs/env.template` is **not read directly** by the code. It is only a source
the driver copies into the clone as `.env`. The Python side
(`config.py` → `BenchmarkSettings`) reads **`.env`**, via
`SettingsConfigDict(env_file=".env", ...)` — resolved relative to the working
directory, which the driver sets to the clone (`cd "$CLONE_DIR"`) before running.

The chain is:

```
configs/env.template
      │  run_ifbench.sh copies it (only if the clone has no .env yet)
      ▼
$CLONE_DIR/.env  ──(+ API_BASE/API_KEY/MODEL overrides injected by the driver)
      │  pydantic-settings loads it (keys are case-insensitive)
      ▼
config.py: BenchmarkSettings  ──▶  evaluate_model.py CLI defaults
```

**Precedence, lowest to highest:**

1. `config.py` `Field(default=...)` — fallback when a key is absent from `.env`.
2. `.env` (i.e. `env.template` values + injected `API_BASE`/`API_KEY`/`MODEL`).
3. Command-line flags forwarded after `run_ifbench.sh` (e.g. `--num-repeats 8`).

So `NUM_REPEATS=2` in `.env` overrides the `config.py` default; passing
`--num-repeats 8` overrides `.env` in turn.

**Notes:**

- The template is copied **only if the clone has no `.env`**. To pick up an
  updated template, delete the clone's `.env` (or edit it directly).
- Only keys declared as fields on `BenchmarkSettings` are consumed; other keys in
  `.env` are ignored by pydantic.

---

## `evaluate_model.py`

A single command that **generates responses from the `.env` model and scores
instruction-following** with the upstream rule-based checkers
(`evaluation_lib`, both *strict* and *loose*). Key properties:

- **Alignment by `key`.** Responses are matched to inputs via `InputExample.key`,
  not by prompt text. This eliminates the prompt-text mismatch/scramble failures
  that the two-file `run_eval.py` flow is prone to (a single stray whitespace or
  a corrupted row causes `KeyError` there).
- **Consensus repeats.** `--num-repeats N` samples N times (repeat *i* uses
  `seed+i`) and reports per-repeat scores plus mean/stdev — matching Nemotron's
  `num_repeats: 8`.
- **Robust generation.** Long per-request timeout (default 3600s) with
  exponential-backoff retries (default 10); `max_tokens<=0` omits the field so
  long thinking output is not truncated; `enable_thinking` is sent as
  `chat_template_kwargs.enable_thinking=true`.

### Useful flags

```
--limit N                 only evaluate the first N prompts (smoke test)
--num-repeats N           consensus repeats (default from .env; Nemotron=8)
--temperature / --top-p   sampling params (default 1.0 / 0.95)
--max-tokens N            <=0 omits max_tokens (server default)
--enable-thinking / --no-thinking
--request-timeout SECS    per-request timeout (default 3600)
--max-retries N           retries on timeout/error (default 10)
--workers N               generation concurrency (default 8)
--save-responses          also dump generated responses per repeat
--output-dir DIR          where to write results (default eval)
```

---

## Parameter choices (Nemotron 3 Ultra alignment)

The upstream `.env.example` ships `TEMPERATURE=0.6`, `MAX_TOKENS=8192`. NVIDIA's
Nemotron 3 Ultra IFBench recipe instead uses:

| Parameter | Upstream default | Nemotron / here |
|---|---|---|
| temperature | 0.6 | **1.0** |
| top_p | (unset) | **0.95** |
| max_tokens | 8192 | **omitted** (server default; avoids truncating thinking) |
| thinking | — | **enabled** |
| num_repeats | 1 | **8** |
| request_timeout | 600 (hardcoded) | **3600** |
| max_retries | — | **10** |

These are the defaults baked into `configs/config.py` and `configs/env.template`.

> **Why timeout + retries matter.** With thinking on and no `max_tokens` cap,
> tail generations can exceed a 600s client timeout; timed-out requests become
> empty responses and are scored as failures, deflating the reported accuracy.
> Raising the timeout to 3600s and retrying (as Nemotron does) removes these
> spurious failures.

---

## Notes & caveats

- **IFBench needs no LLM judge** — scoring is rule-based (strict = exact,
  loose = upper bound over whitespace/formatting variants).
- **Runtime.** Thinking + uncapped tokens makes each request slow (tens of
  seconds to minutes). 300 prompts × 8 repeats is a long run; use `--limit` for
  smoke tests and tune `--workers` to your endpoint's capacity.
- **Reproducibility.** With `SEED` set, the seed sequence (`42,43,…`) is
  deterministic; leaving `SEED` empty samples randomly (not reproducible).
- **`enable_thinking`** only takes effect if the model's chat template reads that
  variable (e.g. Qwen3-style). Otherwise the field is accepted but ignored by the
  template.

---

## References

- Upstream benchmark: https://github.com/allenai/IFBench
- IFBench paper: https://arxiv.org/pdf/2507.02833
- Nemotron 3 Ultra reproduction (IFBench = `nemo_skills.ns_ifbench`):
  `NVIDIA-NeMo/Evaluator` → `examples/nemotron/nemotron-3-ultra/` (`reproducibility.md`,
  `v0.2/README.md`, `local_nemotron-3-ultra-550b-a55b.yaml`)
