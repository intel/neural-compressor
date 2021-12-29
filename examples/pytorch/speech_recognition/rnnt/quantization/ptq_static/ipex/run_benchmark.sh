#!/bin/bash
set -x

export TCMALLOC_DIR=$CONDA_PREFIX/lib
export KMP_BLOCKTIME=1
# tcmalloc:
#export LD_PRELOAD=$TCMALLOC_DIR/libtcmalloc.so

# jemalloc
export LD_PRELOAD=$TCMALLOC_DIR/libjemalloc.so:$TCMALLOC_DIR/libiomp5.so
#export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export MALLOC_CONF="background_thread:true,dirty_decay_ms:8000,muzzy_decay_ms:8000"

PYTHON_VERSION=`python -c 'import sys; print ("{}.{}".format(sys.version_info.major, sys.version_info.minor))'`
SITE_PACKAGES=`python -c 'import site; print (site.getsitepackages()[0])'`
IPEX_VERSION=`conda list |grep torch-ipex | awk '{print $2}' `
export LD_LIBRARY_PATH=$SITE_PACKAGES/torch_ipex-${IPEX_VERSION}-py$PYTHON_VERSION-linux-x86_64.egg/lib/:$LD_LIBRARY_PATH

sockets=`lscpu | grep Socket | awk '{print $2}'`
cores=`lscpu | grep Core.*per\ socket: | awk '{print $4}'`
export DNNL_PRIMITIVE_CACHE_CAPACITY=10485760

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  tuned_checkpoint=saved_results
  scenario=Offline
  backend=pytorch
  for var in "$@"
  do
    case $var in
      --topology=*)
          topology=$(echo $var |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --mode=*)
          mode=$(echo $var |cut -f2 -d=)
      ;;
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)
      ;;
      --iters=*)
          iters=$(echo ${var} |cut -f2 -d=)
      ;;
      --int8=*)
          int8=$(echo ${var} |cut -f2 -d=)
      ;;
      --config=*)
          tuned_checkpoint=$(echo $var |cut -f2 -d=)
      ;;
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

}


# run_benchmark
function run_benchmark {
    if [[ ${mode} == "accuracy" ]]; then
        mode_cmd=" --accuracy_only"
    elif [[ ${mode} == "benchmark" ]]; then
        mode_cmd=" --benchmark --user_conf user_benchmark.sh"
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi

    extra_cmd=""
    if [[ ${int8} == "true" ]]; then
        extra_cmd=$extra_cmd" --int8 --bf16"
    fi
    echo $extra_cmd

    python run.py --dataset_dir ${dataset_location} \
        --manifest $dataset_location/dev-clean-wav.json \
        --pytorch_config_toml pytorch/configs/rnnt.toml \
        --pytorch_checkpoint $input_model \
        --scenario ${scenario} \
        --backend ${backend} \
        --log_dir output \
        --tuned_checkpoint $tuned_checkpoint \
        $mode_cmd \
        ${extra_cmd}
}

main "$@"
