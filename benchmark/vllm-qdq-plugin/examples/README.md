# sage3 attention route example

`sage3_route.json` is an example route file for per-(layer, denoising-step)
attention quant-config routing. Point the plugin at it with:

```bash
SAGE3_ROUTE_FILE=examples/sage3_route.json \
SAGE3_ROUTE_DEBUG=1 \
VLLM_SAGE3_TRITON=1 \
DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN \
  python ...   # your Wan2.2 T2V run
```

## What this example does

For a **40-step, 40-layer** Wan2.2 run:

| Region | Config | Meaning |
| --- | --- | --- |
| steps `0-7` (all layers) | `sdpa` | first 20% of steps run full precision |
| layer `0` (all steps) | `sdpa` | first layer stays full precision |
| everything else | `mxfp4` | cheap quant attention |

## Schema

```json
{
  "default": "mxfp4",
  "rules": [
    {"steps": "0-7", "config": "sdpa"},
    {"layers": "0",  "config": "sdpa"}
  ]
}
```

- Each rule has optional `layers` and/or `steps` (inclusive range strings like
  `"0-7,9,12-15"`) and a required `config`.
- `config` is any key in the sage3 `ATTENTION_CONFIGS` (e.g. `mxfp4`,
  `mxfp8_s1`) or the literal `"sdpa"` (full precision).
- Rules are **first-match-wins**. No matching rule falls back to `default`.
  With no `default`, the impl's own config (`SAGE3_QUANT_FORMAT`) is left alone.
- `SAGE3_ROUTE_DEBUG=1` logs each unique `(layer, step) -> config` once.

## Indices are absolute — rescale for your step count

`steps` and `layers` are absolute indices, not percentages. `"steps": "0-7"`
equals "first 20%" **only at 40 steps**. If you change the step count, rescale
the step rule to keep the same window:

| Steps | First 20% |
| --- | --- |
| 20 | `"0-3"` |
| 40 | `"0-7"` |
| 50 | `"0-9"` |

The layer rule never changes — Wan2.2 always has 40 layers.
