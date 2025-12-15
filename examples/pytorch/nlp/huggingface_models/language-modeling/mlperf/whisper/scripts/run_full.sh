#!/bin/bash

# Prepare workload resources [one-time operations]
bash scripts/download_model.sh
bash scripts/download_dataset.sh
bash scripts/run_calibration.sh 

# Run Benchmark (all scenarios)
SCENARIO=Offline MODE=Performance bash run_mlperf.sh
SCENARIO=Offline MODE=Accuracy    bash run_mlperf.sh

# Run Compliance (all tests)
SCENARIO=Offline MODE=Compliance  bash run_mlperf.sh

# Build submission
VENDOR=OEM SYSTEM=1-node-2S-GNR_128C bash scripts/prepare_submission.sh
