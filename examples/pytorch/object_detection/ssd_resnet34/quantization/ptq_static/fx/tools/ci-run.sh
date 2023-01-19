#!/bin/bash

# where to find stuff
export DATA_ROOT=`pwd`/data
export MODEL_DIR=`pwd`/models

export DATA_ROOT=/data
export MODEL_DIR=$HOME/resnet_for_mlperf


# options for official runs
gopt="--max-batchsize 32 --samples-per-query 2 --threads 2"
gopt="$gopt $@"

result=output/results.csv

function one_run {
    # args: mode framework device model ...
    scenario=$1; shift
    model=$3
    system_desc=$1-$2
    case $model in
    "mobilenet")
      official_model="mobilenet"
      acc_cmd="tools/accuracy-imagenet.py --imagenet-val-file $DATA_ROOT/imagenet2012/val_map.txt";;
    "resnet50")
      official_model="resnet"
      acc_cmd="tools/accuracy-imagenet.py --imagenet-val-file $DATA_ROOT/imagenet2012/val_map.txt";;
    "ssd-mobilenet")
      official_model="ssd-small"
      acc_cmd="tools/accuracy-coco.py --coco-dir $DATA_ROOT/coco";;
    "ssd-resnet34")
      official_model="ssd-large"
      acc_cmd="tools/accuracy-coco.py --use-inv-map --coco-dir $DATA_ROOT/coco";;
    "gnmt")
      official_model="gnmt";;
    esac
    echo "====== $official_model/$scenario ====="
    output_dir=output/$system_desc/$official_model/$scenario

    # accuracy run
    ./run_local.sh $@ --scenario $scenario --accuracy --output $output_dir/accuracy
    python $acc_cmd --verbose --mlperf-accuracy-file $output_dir/accuracy/mlperf_log_accuracy.json \
           >  $output_dir/accuracy/accuracy.txt
    cat $output_dir/accuracy/accuracy.txt

    # performance run
    ./run_local.sh $@ --scenario $scenario --output $output_dir/performance
    
    # summary to csv
    python tools/lglog2csv.py --input $output_dir/performance/mlperf_log_summary.txt --runtime "$1-$2" \
      --machine $HOSTNAME --model $3 --name $1-$2-py >> $result
}

function one_model {
    # args: framework device model ...
    one_run SingleStream $@
    one_run MultiStream $@
    one_run Server $@
    one_run Offline $@
}


mkdir output
echo "build,date,machine,runtime,model,mode,qps,mean,latency_90,latency_99" > $result

# TODO: add gnmt

# using imagenet
export DATA_DIR=$DATA_ROOT/imagenet2012
#one_model onnxruntime cpu mobilenet $gopt
#one_model tf gpu resnet50 $gopt

# using coco
export DATA_DIR=$DATA_ROOT/coco
#one_model tf gpu ssd-mobilenet $gopt
one_model tf gpu ssd-resnet34 $gopt
#one_model onnxruntime cpu ssd-resnet34 $gopt
