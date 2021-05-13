#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
#WARNING: must have compiled PyTorch and caffe2

#check if extra argument is passed to the test
if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi
#echo $dlrm_extra_option
export data_path=/mnt/local_disk3/dataset/dlrm/dlrm/input
export fp32_load_path=/mnt/local_disk3/dataset/dlrm/dlrm_weight/terabyte_mlperf.pt

# echo "fp32 inference"
# python -u dlrm_s_pytorch_tune.py --arch-sparse-feature-size=128 --arch-mlp-bot="13-512-256-128" --arch-mlp-top="1024-1024-512-256-1" --max-ind-range=40000000 --data-generation=dataset --data-set=terabyte --raw-data-file=${data_path}/day --processed-data-file=${data_path}/terabyte_processed.npz --loss-function=bce --round-targets=True --learning-rate=1.0 --mini-batch-size=2048 --print-freq=2048 --print-time --test-freq=102400 --test-mini-batch-size=16384 --test-num-workers=16 --memory-map --mlperf-logging --mlperf-auc-threshold=0.8025 --mlperf-bin-loader --mlperf-bin-shuffle --load-model=${fp32_load_path}  --inference-only $dlrm_extra_option 2>&1 | tee run_terabyte_mlperf_pt_fp32inference.log
# echo "int8 inference"
# python -u dlrm_s_pytorch_tune.py --arch-sparse-feature-size=128 --arch-mlp-bot="13-512-256-128" --arch-mlp-top="1024-1024-512-256-1" --max-ind-range=40000000 --data-generation=dataset --data-set=terabyte --raw-data-file=${data_path}/day --processed-data-file=${data_path}/terabyte_processed.npz --loss-function=bce --round-targets=True --learning-rate=1.0 --mini-batch-size=2048 --print-freq=2048 --print-time --test-freq=102400 --test-mini-batch-size=16384 --test-num-workers=16 --memory-map --mlperf-logging --mlperf-auc-threshold=0.8025 --mlperf-bin-loader --mlperf-bin-shuffle --load-model=${fp32_load_path}  --inference-only --do-int8-inference  $dlrm_extra_option 2>&1 | tee run_terabyte_mlperf_pt_int8.log
echo "lpot tune"
python -u dlrm_s_pytorch_tune.py --arch-sparse-feature-size=128 --arch-mlp-bot="13-512-256-128" --arch-mlp-top="1024-1024-512-256-1" --max-ind-range=40000000 --data-generation=dataset --data-set=terabyte --raw-data-file=${data_path}/day --processed-data-file=${data_path}/terabyte_processed.npz --loss-function=bce --round-targets=True --learning-rate=1.0 --mini-batch-size=2048 --print-freq=2048 --print-time --test-freq=102400 --test-mini-batch-size=16384 --test-num-workers=16 --memory-map --mlperf-logging --mlperf-auc-threshold=0.8025 --mlperf-bin-loader --mlperf-bin-shuffle --load-model=${fp32_load_path} --tune  $dlrm_extra_option 2>&1 | tee run_terabyte_mlperf_pt_int8.log
echo "done"





