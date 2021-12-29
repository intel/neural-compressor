#!/bin/bash

# Runs benchmark and reports time to convergence

pushd pytorch

# Single GPU training
time python tools/test_net.py --tune --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" \
       SOLVER.IMS_PER_BATCH 2 TEST.IMS_PER_BATCH 1 SOLVER.MAX_ITER 720000 SOLVER.STEPS "(480000, 640000)" SOLVER.BASE_LR 0.0025 MODEL.DEVICE "cpu"
       
popd
