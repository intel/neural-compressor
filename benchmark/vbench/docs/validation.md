# Validation status

Historical full runs and new portable-runner validation are distinct evidence.

## Historical full measurements

- T2V: three checkpoints × 165 videos; five official metric implementations on
  all 165 videos per model. The checked-in result JSONs preserve full precision.
- I2V: three checkpoints × 355 videos; five official metric implementations with
  dimension-tag filtering (246 or 109 inputs per metric).
- Video files were decoded/checked for the stated dimensions, frame count, and fps.
- All historical generation workers and scoring queues finished successfully.

## Portable repository

- 19 CPU-only tests pass using Python 3.12 without site-packages (`python -S`).
- Python compilation and shell syntax checks pass.
- Frozen T2V input ordering matches the historical jobs (72 + 93); I2V metadata
  matches the historical SHA-256, with the same 355 selected inputs and metric counts.
- Evaluator tests mock model computation but exercise real metadata selection,
  filesystem mapping, result persistence, and hash-based invalidation.
- Real I2V NVFP4 one-video generation passed using the new runner on NVIDIA B200,
  with the historical environment: 832 × 480, 81 frames, 16 fps, 40 steps, seed 0.
- Real T2V NVFP4 one-video generation also passed using the new runner on NVIDIA
  B200 with the historical environment: 1280 × 720, 81 frames, 16 fps, 40 steps,
  seed 42, CFG 4.0/3.0. Both real smoke workers exited 0; generated video hashes,
  per-video metadata fingerprints, and decoded formats passed the runner checks.
- A partial I2V smoke root containing one of 355 videos was correctly rejected by
  full verification (354 missing), preventing smoke output from being reported as
  a complete benchmark. Environment capture also resolved the pinned Omni source
  and Torch 2.14.0+cu130 correctly in the historical overlay.

Not validated: a clean environment bootstrap, a fresh full 1,560-video generation
through the new runner, a full new-runner score reproduction, alternate GPUs,
multi-seed statistics, or bitwise equivalence across software/hardware revisions.
The historical total is 495 T2V + 1,065 I2V videos, excluding separate showcases.
Smoke runs do not reduce the selected full-run job count or qualify for full scoring.
