#!/usr/bin/env python3
"""Patch evalplus 0.3.1's OpenAI decoder for long / reasoning-style outputs.

Changes:
- raise the default `max_new_tokens` from 768 to 4096;
- treat `message.content is None` as an empty string;
- retry empty completions up to two times.

Run with the same Python that has evalplus installed:

    .venv/bin/python scripts/patch_evalplus.py
"""

from __future__ import annotations

import re
from pathlib import Path

import evalplus

ROOT = Path(evalplus.__file__).resolve().parent


def patch_base() -> None:
    path = ROOT / "provider/base.py"
    text = path.read_text()
    updated, count = re.subn(
        r"max_new_tokens: int = \d+,",
        "max_new_tokens: int = 4096,",
        text,
        count=1,
    )
    if count and updated != text:
        path.write_text(updated)
        print(f"patched {path}")
    else:
        print(f"base.py already patched or pattern not found: {path}")


def patch_openai() -> None:
    path = ROOT / "provider/openai.py"
    text = path.read_text()
    if "for _ in range(2):" in text:
        print(f"openai.py already patched: {path}")
        return

    old = """        outputs = []
        for item in ret.choices:
            outputs.append(item.message.content)

        return outputs"""
    new = """        outputs = [item.message.content or "" for item in ret.choices]
        for index, text in enumerate(outputs):
            if text.strip():
                continue
            for _ in range(2):
                retry = openai_request.make_auto_request(
                    self.client,
                    message=message,
                    model=self.name,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    n=1,
                )
                candidate = retry.choices[0].message.content or ""
                if candidate.strip():
                    outputs[index] = candidate
                    break

        return outputs"""
    if old not in text:
        raise SystemExit("openai.py pattern not found; evalplus version may differ")
    path.write_text(text.replace(old, new, 1))
    print(f"patched {path}")


if __name__ == "__main__":
    patch_base()
    patch_openai()
