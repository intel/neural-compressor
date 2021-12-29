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

root_dir=`pwd`
work_dir=$root_dir/mlperf-rnnt-librispeech
local_data_dir=$work_dir/local_data
configure_path=calibration_result.json

scenario=Offline
machine_conf=offline.conf
backend=pytorch
for arg in $@; do
    case ${arg} in
        --accuracy) accuracy="--accuracy_only";;
        --debug) debug="--debug";;
        --profile*)
            if [[ $(echo ${arg} | cut -f2 -d=) == "--profile" ]];then
                profile="--profile True"
            else
                profile="--profile $(echo ${arg} | cut -f2 -d=)"
            fi;;
        --server)
            scenario=Server
            machine_conf=server.conf;;
        --verbose*) verbose="--verbose $(echo ${arg} | cut -f2 -d=)";;
        --warmup) warmup="--warmup";;
        *) echo "Error: No such parameter: ${arg}" exit 1;;
    esac
done

log_dir=${work_dir}/${scenario}_${backend}
if [ ! -z ${accuracy} ]; then
    log_dir+=_accuracy
fi
log_dir+=rerun

export DNNL_PRIMITIVE_CACHE_CAPACITY=10485760

python run.py --dataset_dir $local_data_dir \
    --manifest $local_data_dir/dev-clean-wav.json \
    --pytorch_config_toml pytorch/configs/rnnt.toml \
    --pytorch_checkpoint $work_dir/rnnt.pt \
    --scenario ${scenario} \
    --backend ${backend} \
    --log_dir output \
    --configure_path $configure_path \
    --machine_conf $machine_conf \
    ${accuracy} \
    ${warmup} \
    ${debug} \
    ${profile} \
    ${verbose} \
    --bf16 \
    --int8
