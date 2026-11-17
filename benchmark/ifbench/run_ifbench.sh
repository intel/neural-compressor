#!/usr/bin/env bash
#
# run_ifbench.sh — one-shot IFBench reproduction driver.
#
# Clones allenai/IFBench, sets up a Python environment, overlays our
# `evaluate_model.py` (end-to-end generate + score, aligned to the NVIDIA
# Nemotron 3 Ultra IFBench recipe), configures the endpoint via `.env`, and runs
# the evaluation.
#
# Usage:
#   ./run_ifbench.sh                         # interactive-ish; edits .env if missing
#   API_BASE=... API_KEY=... MODEL=... ./run_ifbench.sh
#   ./run_ifbench.sh --limit 10 --num-repeats 1   # smoke test
#
# Any extra CLI args are forwarded verbatim to evaluate_model.py.
#
set -euo pipefail

# ------------------------------------------------------------------ config ----
REPO_URL="${REPO_URL:-https://github.com/allenai/IFBench.git}"
REPO_REF="${REPO_REF:-main}"
# Where to clone the benchmark (kept outside this repo so we don't commit it).
WORKDIR="${WORKDIR:-$HOME/ifbench-run}"
CLONE_DIR="${CLONE_DIR:-$WORKDIR/IFBench}"
# Python interpreter used to create the venv. Override if needed, e.g.
#   PYTHON_BIN=/data/miniforge3/bin/python3 ./run_ifbench.sh
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$WORKDIR/.venv}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS_DIR="$SCRIPT_DIR/configs"

log() { printf '\033[1;32m[ifbench]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[ifbench:error]\033[0m %s\n' "$*" >&2; }

# ------------------------------------------------------------- 1. clone -------
mkdir -p "$WORKDIR"
if [[ -d "$CLONE_DIR/.git" ]]; then
  log "Repo already present at $CLONE_DIR (skipping clone)."
else
  log "Cloning $REPO_URL ($REPO_REF) -> $CLONE_DIR"
  git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$CLONE_DIR"
fi

# ------------------------------------------------- 2. overlay our configs -----
# evaluate_model.py is new; config.py is patched with Nemotron-aligned defaults
# (temperature/top_p/enable_thinking/num_repeats/request_timeout/max_retries).
log "Overlaying evaluate_model.py and patched config.py into the clone."
cp "$SCRIPT_DIR/evaluate_model.py" "$CLONE_DIR/evaluate_model.py"
cp "$CONFIGS_DIR/config.py"        "$CLONE_DIR/config.py"

# ------------------------------------------------- 3. python environment ------
if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating virtualenv at $VENV_DIR ($PYTHON_BIN)"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

log "Installing dependencies."
python -m pip install --upgrade pip >/dev/null
# Upstream benchmark deps + the HTTP/CLI deps evaluate_model.py needs.
python -m pip install -r "$CLONE_DIR/requirements.txt"
python -m pip install "httpx" "tqdm" "pydantic-settings"

# NLTK data used by some instruction checkers (best-effort).
python - <<'PY' || true
import nltk
for pkg in ("punkt", "punkt_tab"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass
PY

# --------------------------------------------------------- 4. configure .env --
ENV_FILE="$CLONE_DIR/.env"
if [[ ! -f "$ENV_FILE" ]]; then
  log "Writing $ENV_FILE from template."
  cp "$CONFIGS_DIR/env.template" "$ENV_FILE"
fi
# Allow overriding endpoint/model from the environment without editing the file.
python - "$ENV_FILE" <<'PY'
import os, re, sys
path = sys.argv[1]
overrides = {k: os.environ[k] for k in ("API_BASE", "API_KEY", "MODEL")
             if os.environ.get(k)}
if overrides:
    lines = open(path).read().splitlines()
    seen = set()
    for i, ln in enumerate(lines):
        m = re.match(r"^(\w+)=", ln)
        if m and m.group(1) in overrides:
            k = m.group(1); lines[i] = f"{k}={overrides[k]}"; seen.add(k)
    for k, v in overrides.items():
        if k not in seen:
            lines.append(f"{k}={v}")
    open(path, "w").write("\n".join(lines) + "\n")
    print(f"[ifbench] Applied env overrides: {', '.join(overrides)}")
PY

# Fail early if the endpoint is still the placeholder and no override was given.
if grep -qE '^API_BASE=https://your-openai-compatible-endpoint' "$ENV_FILE"; then
  err "API_BASE is still the placeholder. Edit $ENV_FILE (or export API_BASE/API_KEY/MODEL) and re-run."
  exit 2
fi

# --------------------------------------------------------------- 5. run -------
log "Running evaluation (extra args: ${*:-<none>})."
cd "$CLONE_DIR"
python evaluate_model.py "$@"

log "Done. Results are under $CLONE_DIR/eval/  (see *-results.json for the summary)."
