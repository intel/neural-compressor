# Observed quality results

[t2v.json](t2v.json) and [i2v.json](i2v.json) preserve the full numeric precision of the historical BF16, MXFP8, and NVFP4 summary files. They are observed scores from completed local evaluations, not CI-generated scores or CI acceptance thresholds. No new generation or scoring was run to create these snapshots. Scores are fractions; higher is better for these metrics.

The `protocol` applies to all three model entries in each file. `video_count` counts generated videos per model; `metric_counts` counts videos actually scored for each dimension. `source_summary_pattern` identifies the archived summaries without personal paths. Model IDs identify the published model families; immutable Hugging Face checkpoint revisions and artifact hashes are not recorded here.

| Snapshot | Videos per model | Metric coverage |
|---|---:|---|
| T2V | 165 | All five metrics on all 165 videos together, not a mean of the 72- and 93-video group scores |
| I2V | 355 | `i2v_subject`, `subject_consistency`, `motion_smoothness`: 246 each; `i2v_background`, `background_consistency`: 109 each |

T2V uses the historical two-group prompt subset in [the data snapshot](../data/t2v_prompts.json), 1280 × 720, 81 frames, 40 steps, seed 42, CFG 4.0/3.0, and 16 fps. I2V uses 355 official prompt/image pairs, cropped 16:9 references, 832 × 480, 81 frames, 40 steps, seed 0 reset per pair, CFG 3.5/3.5, and 16 fps. I2V metric membership follows the official metadata tags, not all 355 videos for every metric. Both use the official Wan negative prompt, recorded in the JSON protocols.

These are subset, one-video-per-input snapshots, **not standard full-suite VBench runs**. No official aggregate VBench score is reported. The I2V NVFP4 **absolute mean drop is 0.994491942 percentage points** versus BF16: 100 times the equally weighted mean of `(BF16 - NVFP4)` across the five recorded metric scores. This is a descriptive absolute difference, not a relative loss or an official VBench total.

Reruns are not guaranteed to be bitwise identical. Checkpoint revisions, software and kernel versions, hardware, and nondeterminism can affect outputs. The historical evaluator retained weights-only loading with a narrow `typing.OrderedDict` safe-global alias for the AMT checkpoint. Per-video files and machine-local paths are intentionally not duplicated here. These quality snapshots make no throughput, latency, or memory-performance claims; a single showcase video is not a performance benchmark.

See [data attribution](../data/README.md) and [quantization provenance](../docs/quantization.md), including the limits of the recorded MXFP8 recipes.
