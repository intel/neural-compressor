# Evaluator model caches and first-run downloads

Scoring requires additional pretrained models, separately from Wan weights. Install
VBench 0.1.5 and its metric dependencies in the evaluated environment; do not
replace upstream scoring implementations with approximate substitutes.

| Metric | Evaluator model |
|---|---|
| T2V `imaging_quality` | PyIQA MUSIQ |
| T2V `aesthetic_quality` | LAION aesthetic predictor and CLIP |
| T2V `overall_consistency` | ViCLIP |
| `subject_consistency`, I2V `i2v_subject` | DINO ViT-B/16 |
| I2V `background_consistency` | CLIP ViT-B/32 |
| `motion_smoothness` | AMT-S |
| I2V `i2v_background` | DreamSim |

The upstream initializers download missing files even with `local=True`; local
means using local model repositories/checkpoint paths, **not an offline guarantee**.
First evaluation needs access to upstream GitHub, Hugging Face, PyTorch model
hosting, and the checkpoint URLs used by VBench. A slow/no-progress initialization
can be a checkpoint download or shared-filesystem delay, not GPU computation.

## DINO

VBench expects a DINO source checkout under
`~/.cache/vbench/dino_model/facebookresearch_dino_main/` and a checkpoint named
`dino_vitbase16_pretrain.pth`. Its initializer can clone/download these. Torch Hub
inside DINO may also look under `~/.cache/torch/hub/checkpoints/`, so provisioning
only the VBench path may not prevent a second download.

The checkpoint used in the historical run is available from:

```
https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth
```

SHA-256:
`bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e`.
If provisioning manually, ensure both cache locations resolve to those verified
bytes, e.g. by using a symlink. Do not silently use a different backbone/checkpoint.

## CLIP and AMT

The I2V initializer uses `~/.cache/vbench/clip_model/ViT-B-32.pt` for background
consistency and `~/.cache/vbench/amt_model/amt-s.pth` for motion smoothness.
VBench's other metrics may also use CLIP ViT-L/14 and their own caches.

The historical ViT-B/32 checkpoint SHA-256 is
`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`.
The AMT-S source used by the initializer is
`https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth`.

The adapter allows only the legacy AMT pickle name `typing.OrderedDict` mapped
to `collections.OrderedDict` inside a scoped `torch.serialization.safe_globals`
context. It does not globally disable `weights_only` loading. Download checkpoints
only from trusted upstream sources; do not broadly allowlist arbitrary pickle types.

## DreamSim and other caches

DreamSim 0.2.1 defaults to a `models` directory relative to the process working
directory. The I2V adapter runs scoring in `RUN_ROOT/evaluation/reference_view`,
so DreamSim downloads are stored underneath that directory. The reference-view
symlink also supplies VBench's hardcoded relative `vbench2_beta_i2v/data/crop/16-9`
lookup without modifying installed code or copying the images.

PyIQA, ViCLIP, LAION, and Torch Hub have their own cache conventions. Do not
assume `VBENCH_CACHE_DIR` relocates every dependency. To evaluate fully offline,
provision all relevant caches first and verify the exact metric computations in
that environment. This repository does not bundle or relicense checkpoint files.
