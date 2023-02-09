#!/bin/bash
set -xe

framework=$1
batch_size=$2
cores_per_instance=$3
precision=$4
malloc=$5
num_inference_steps=$6
guidance_scale=$7
resolution=$8

function main {
    export KMP_BLOCKTIME=1
    export KMP_SETTINGS=1
    export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
    # ## iomp
    if [[ "$malloc" == *"malloc" ]];then
        export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
    fi
    # ## jemalloc
    if [ "$malloc" == "jemalloc" ];then
        export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libjemalloc.so
        export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
    fi
    # ## tcmalloc
    if [ "$malloc" == "tcmalloc" ];then
        export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
    fi

    # precision
    if [ "$precision" == "bf16" ];then
        extra_args='  --precision bfloat16  '
    else
        extra_args='  --precision float32  '
    fi
    get_instance

    # launch
    log_dir="logs/diffusers-$(echo $@ |sed 's/ /-/g')-$(date +%s)"
    rm -rf *.png ${log_dir} && mkdir -p ${log_dir}
    launch $@

}

function get_instance {
    # CPU
    scaling_governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
    if [ "${scaling_governor}" != "performance" ];then
        echo "[WARN] Your system governor mode is ${scaling_governor}."
        # sudo cpupower frequency-set -g performance
    fi
    if [ $(sudo -n true > /dev/null 2>&1 && echo $? || echo $?) -eq 0 ];then
        # clean cache
        sync; sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
    fi
    sockets_num=$(lscpu |grep 'Socket(s):' |sed 's/[^0-9]//g')
    cores_per_socket=$(lscpu |grep 'Core(s) per socket:' |sed 's/[^0-9]//g')
    phsical_cores_num=$(echo |awk -v sockets_num=${sockets_num} -v cores_per_socket=${cores_per_socket} '{
        print sockets_num * cores_per_socket;
    }')
    numa_nodes_num=$(lscpu |grep 'NUMA node(s):' |sed 's/[^0-9]//g')
    cores_per_node=$(echo |awk -v phsical_cores_num=${phsical_cores_num} -v numa_nodes_num=${numa_nodes_num} '{
        print phsical_cores_num / numa_nodes_num;
    }')
    # cpu array
    cpu_array=($(numactl -H |grep "node [0-9]* cpus:" |sed "s/.*node//;s/cpus://" |sed -n '1p' |\
    awk -v cpn=${cores_per_node} '{for(i=1;i<=cpn+1;i++) {printf(" %s ",$i)} print ""}' |grep "[0-9]" |\
    awk -v cpi=${cores_per_instance} -v cps=${cores_per_node} -v cores=${OOB_TOTAL_CORES_USE} '{
        if(cores == "") { cores = NF; }
        for( i=2; i<=cores; i++ ) {
            if((i-1) % cpi == 0 || (i-1) % cps == 0) {
                print $i";"$1
            }else {
                printf $i","
            }
        }
    }' |sed "s/,$//"))
    instance=${#cpu_array[@]}

}

function launch {

    excute_cmd_file="${log_dir}/itex-run-$(date +'%s').sh"
    rm -f ${excute_cmd_file}

    # generate launch script
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${cpu_array[i]} |awk -F, '{print NF}')
        log_file="${log_dir}/rcpi${real_cores_per_instance}-ins${i}-cpu.log"

        printf "OMP_NUM_THREADS=$(echo ${cpu_array[i]} |awk -F, '{printf("%d", NF)}') \
            numactl -m $(echo ${cpu_array[i]} |awk -F ';' '{print $2}') \
                    -C $(echo ${cpu_array[i]} |awk -F ';' '{print $1}') \
            python run_${framework}.py --per_device_eval_batch_size $batch_size \
                --num_inference_steps $num_inference_steps \
                --resolution $resolution \
                --guidance_scale $guidance_scale \
                --save_image $extra_args \
        > ${log_file} 2>&1 &  \n" |tee -a ${excute_cmd_file}
        if [ "${numa_nodes_use}" == "0" ];then
            break
        fi
    done
    echo -e "\n wait" >> ${excute_cmd_file}
    echo -e "\n\n\n\n Running..."

    # execute
    cat ${excute_cmd_file} |column -t > ${excute_cmd_file}.tmp
    mv ${excute_cmd_file}.tmp ${excute_cmd_file}
    source ${excute_cmd_file}

    # get result
    Latency=$(grep 'inference Latency:' ${log_dir}/rcpi*-cpu.log |sed -e 's/.*inference Latency://;s/[^0-9.]//g' |awk '
        BEGIN {
            num = 0;
            sum = 0;
        }{
            num = num + 1;
            sum = sum + $1;
        }END {
            printf("%.6f", sum / num);
        }
    ')
    cp astronaut_rides_horse-*.png ${log_dir}/sd-$(echo $@ |sed 's/ /-/g')-${throughput_cpu}.png

    echo "Latency: ${Latency} ms"
}

main $@

