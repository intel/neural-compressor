# SPDX-License-Identifier: Apache-2.0
"""Route table for per-(layer, denoising-step) attention quant-config routing.

A route file is JSON:

    {
      "default": "mxfp4",
      "rules": [
        {"layers": "20-39", "steps": "0-7", "config": "mixed_fp8qk_fp4pv"},
        {"steps": "0-7",   "config": "sdpa"},
        {"layers": "20-39", "config": "mxfp8_s1"}
      ]
    }

Each rule has an optional ``layers`` predicate, an optional ``steps`` predicate
(inclusive range strings like ``"0-7,9,12-15"``), and a required ``config`` that
must be a sage3 quant-config name (key of ``ATTENTION_CONFIGS``) or the literal
token ``"sdpa"``. A rule matches when all of its present predicates match.
Rules are evaluated in order, first match wins; otherwise ``default`` (or
``None`` if no default) is returned.
"""

import json

from ..sage3.quant_config import ATTENTION_CONFIGS

# Full-precision fallback token (handled by Sage3TritonImpl._forward_sdpa).
SDPA_TOKEN = "sdpa"


def _allowed_configs() -> set[str]:
    """Valid route-config vocabulary: sage3 config names plus the sdpa token."""
    return set(ATTENTION_CONFIGS) | {SDPA_TOKEN}


def _parse_ranges(spec: str) -> set[int]:
    """Parse an inclusive-range string like ``"0-7,9,12-15"`` into a set of ints."""
    result: set[int] = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            try:
                lo, hi = int(lo_s.strip()), int(hi_s.strip())
            except ValueError:
                raise ValueError(f"invalid range segment {part!r} in {spec!r}")
            if hi < lo:
                raise ValueError(f"descending range {part!r} in {spec!r}")
            result.update(range(lo, hi + 1))
        else:
            try:
                result.add(int(part))
            except ValueError:
                raise ValueError(f"invalid integer {part!r} in {spec!r}")
    return result


def _coerce_range_spec(value, field: str, rule_idx: int) -> str:
    if isinstance(value, bool):  # bool is an int subclass; reject explicitly
        raise ValueError(f"route rule {rule_idx}: {field!r} must be a range string")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value
    raise ValueError(
        f"route rule {rule_idx}: {field!r} must be a range string " f"(e.g. '0-7,9'), got {type(value).__name__}"
    )


class _Rule:
    __slots__ = ("layers", "steps", "config")

    def __init__(self, layers: set[int] | None, steps: set[int] | None, config: str):
        self.layers = layers
        self.steps = steps
        self.config = config

    def matches(self, layer_idx: int | None, step_idx: int | None) -> bool:
        if self.layers is not None:
            if layer_idx is None or layer_idx not in self.layers:
                return False
        if self.steps is not None:
            if step_idx is None or step_idx not in self.steps:
                return False
        return True


class RouteTable:
    """Ordered, first-match-wins routing of attention quant configs."""

    def __init__(self, default: str | None, rules: list[_Rule]):
        self.default = default
        self.rules = rules

    @classmethod
    def from_json_file(cls, path: str) -> "RouteTable":
        with open(path) as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"route file {path!r} must contain a JSON object, " f"got {type(data).__name__}")

        allowed = _allowed_configs()

        default = data.get("default")
        if default is not None and default not in allowed:
            raise ValueError(
                f"route file {path!r}: default config {default!r} not in " f"allowed configs {sorted(allowed)}"
            )

        raw_rules = data.get("rules", [])
        if not isinstance(raw_rules, list):
            raise ValueError(f"route file {path!r}: 'rules' must be a list, " f"got {type(raw_rules).__name__}")

        rules: list[_Rule] = []
        for i, raw in enumerate(raw_rules):
            if not isinstance(raw, dict):
                raise ValueError(f"route file {path!r}: rule {i} must be an object, " f"got {type(raw).__name__}")
            config = raw.get("config")
            if config is None:
                raise ValueError(f"route file {path!r}: rule {i} missing 'config'")
            if config not in allowed:
                raise ValueError(
                    f"route file {path!r}: rule {i} config {config!r} not in " f"allowed configs {sorted(allowed)}"
                )
            layers = _parse_ranges(_coerce_range_spec(raw["layers"], "layers", i)) if "layers" in raw else None
            steps = _parse_ranges(_coerce_range_spec(raw["steps"], "steps", i)) if "steps" in raw else None
            rules.append(_Rule(layers, steps, config))

        return cls(default, rules)

    def resolve(self, layer_idx: int | None, step_idx: int | None) -> str | None:
        """Return the routed config name, or None to leave the impl default.

        Never raises. A ``None`` signal (unknown layer or step) simply fails to
        match any predicate that constrains that dimension.
        """
        for rule in self.rules:
            if rule.matches(layer_idx, step_idx):
                return rule.config
        return self.default
