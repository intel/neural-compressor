# Image-to-video protocol

[Back to overview](../README.md) · [Environment](environment.md) · [Recorded results](../results/i2v.json)

## Official inputs

[The metadata snapshot](../data/i2v_metadata.json) contains all **1,118** upstream records. Its SHA-256 is `8cf4d34dc11779a336c216a1a71d482927cfa24a1bf736ab5c92425609f129f8`. Preparation selects only the union tagged `i2v_subject` or `i2v_background`: **355 jobs** (246 subject + 109 background). The 763 camera-motion records are not generation jobs in this protocol.

The reference images are external data. Follow the [official VBench I2V instructions](https://github.com/Vchitect/VBench/tree/master/vbench2_beta_i2v), [data attribution](../data/README.md), and any source-image terms. The downloader uses `gdown` to obtain official Google Drive **crop.zip**, file ID `1Y_JnYnyJ3a6QhiranoX0MQVZFcTDPekZ`, and extracts it into the supplied destination. Metadata is already in this repository; no separate metadata download is needed.

```bash
bash scripts/download_i2v.sh /path/to/external/vbench-i2v-data
export I2V_IMAGES=/path/to/external/vbench-i2v-data/crop/16-9
python scripts/benchmark.py prepare --task i2v --root runs/i2v-nvfp4 \
  --i2v-images "$I2V_IMAGES"
```

Use the official cropped **16:9** image set and preserve its filenames. Do not substitute full images or other aspect-ratio crops. `prepare` validates/snapshots selected jobs and reference-image identities; it does not initialize the model. The fixed set remains 355 even for smoke generation.

## Generation

| Setting | Recorded value |
|---|---|
| Resolution / dimension multiple | 832 × 480 / 16 |
| Frames / export rate | 81 / 16 fps |
| Inference steps | 40 |
| Seed | 0 for each input |
| Guidance scales | 3.5 / 3.5 |
| Boundary ratio / flow shift | 0.9 / 3.0 |
| Negative prompt | Official Wan negative prompt, verbatim in the [protocol JSON](../results/i2v.json) |
| Image resize | LANCZOS, official Omni example |
| Attention / execution | `TRTLLM_ATTN` / eager |
| Cache / CPU offload / VAE tiling / VAE slicing | Disabled |

The runner invokes the **unchanged official [image-to-video example](https://github.com/vllm-project/vllm-omni/blob/7465c0f52bcb46a55f62c1b8613846cda30d0083/examples/offline_inference/image_to_video/image_to_video.py)** from the pinned checkout. Its `Omni` construction is cached within each worker so the same engine serves successive inputs; the example file itself is not patched. Do not transfer the T2V API's explicit linear-backend override to this example path or interpret cached engine reuse as diffusion caching.

```bash
export OMNI_SOURCE=/path/to/vllm-omni
export MODEL=/path/to/local/Wan2.2-I2V-A14B-Diffusers-NVFP4-AutoRound
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark.py generate \
  --task i2v --root runs/i2v-nvfp4 --model "$MODEL" --precision nvfp4 \
  --omni-source "$OMNI_SOURCE" --shard-index 0 --num-shards 1
python scripts/benchmark.py verify --task i2v --root runs/i2v-nvfp4
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark.py evaluate \
  --task i2v --root runs/i2v-nvfp4
```

Use one root per model. The generation configuration is persisted at `generate`; fingerprint matching and `ffprobe` checks protect resume. Required outputs have the recorded dimensions, 81 decoded frames, and 16 fps. A `--limit N` run is a smoke test, not a smaller official evaluation; use a separate smoke root and do not score it as a complete run.

### Five-GPU reproduction and portable sharding

The historical run used physical GPUs **0, 1, 2, 5, 7**, five independent workers, **71 inputs each**. All workers completed the NVFP4 phase first, then BF16, then MXFP8. That ordering is historical execution provenance, not a requirement for scoring.

```bash
bash scripts/launch_workers.sh i2v "$MODEL" nvfp4 runs/i2v-nvfp4 \
  "$OMNI_SOURCE" 0,1,2,5,7 "$I2V_IMAGES"
```

The launcher prepares, assigns indexed shards, writes worker logs, waits for all workers, and verifies completion. The portable CLI supports arbitrary independent sharding, not only five GPUs: all workers use the same root and `--num-shards N`, with distinct `--shard-index` values. Each worker owns `jobs[index::N]`. Changing the physical GPU list does not change the selected inputs; timing or bitwise identity across hardware is not promised.

## Evaluation: filter each dimension separately

Generation produces 355 videos per model. Evaluation retains only records carrying the **metric's own official dimension tag**; it does not run every metric on all 355 videos.

| CLI metric | Evaluator model | Videos |
|---|---|---:|
| `i2v_subject` | DINO reference-image/video features | 246 |
| `subject_consistency` | DINO temporal subject features | 246 |
| `background_consistency` | CLIP temporal background features | 109 |
| `motion_smoothness` | AMT frame interpolation | 246 |
| `i2v_background` | DreamSim reference-image/video similarity | 109 |

The `i2v_*` compute functions come from `vbench2_beta_i2v`; the other three come from `vbench`. Evaluation uses `local=True`, `read_frame=False`, and the 16:9 reference set. The evaluator's checkpoint/image caches must be available; the repository does not include them. To rerun selected metrics:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark.py evaluate \
  --task i2v --root runs/i2v-nvfp4 --metrics i2v_subject i2v_background
```

This is a two-metric rerun, not a full five-metric report. GPU selection is through the shell, not an `--gpu` flag.

### Observed scores

| Metric | BF16 | MXFP8 | NVFP4 |
|---|---:|---:|---:|
| `i2v_subject` | 0.977584 | 0.975640 | 0.961735 |
| `subject_consistency` | 0.952603 | 0.951910 | 0.936307 |
| `background_consistency` | 0.969536 | 0.968830 | 0.961778 |
| `motion_smoothness` | 0.984598 | 0.985272 | 0.983227 |
| `i2v_background` | 0.991074 | 0.989965 | 0.982622 |

Rounded from [the recorded JSON](../results/i2v.json). This protocol uses **one video per selected pair**, not the standard five-sample full protocol; camera motion is excluded. It is not a full-suite result. See [quantization provenance](quantization.md) for the distinction between the I2V MXFP8 card recipe and independently verified export history.
