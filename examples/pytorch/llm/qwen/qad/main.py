# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Backward-compatible Qwen QAD CLI wrapper delegating to ``train.py``.

Supports both legacy flags (``--recipe``, ``--data``, ``--output``, ``--amax-source``, ``--stop-after``)
and standard v2.0 ``train.py`` ``HfArgumentParser`` flags.
"""

from __future__ import annotations

import sys

from train import run_training

_LEGACY_FLAG_MAP = {
    "--recipe": "--config",
    "--data": "--data_prefix",
    "--output": "--output_dir",
    "--model": "--model_name_or_path",
    "--teacher": "--teacher_model_name_or_path",
    "--amax-source": "--quantizer_state_path",
    "--stop-after": "--stop_after_steps",
    "--save-steps": "--save_steps",
    "--seq-length": "--seq_length",
    "--train-iters": "--max_steps",
    "--calib-batches": "--calib_batches",
    "--synthetic-batches": "--synthetic_batches",
    "--top-k": "--kd_top_k",
    "--debug-grads": "--grad_debug",
    "--resume-from-checkpoint": "--resume_from_checkpoint",
    "--export-dir": "--export_dir",
}


def _translate_legacy_argv(argv: list[str]) -> list[str]:
    translated: list[str] = []
    for token in argv:
        if "=" in token:
            head, tail = token.split("=", 1)
            translated.append(f"{_LEGACY_FLAG_MAP.get(head, head)}={tail}")
        else:
            translated.append(_LEGACY_FLAG_MAP.get(token, token))
    return translated


def main(argv: list[str] | None = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    run_training(_translate_legacy_argv(raw))


if __name__ == "__main__":
    main()
