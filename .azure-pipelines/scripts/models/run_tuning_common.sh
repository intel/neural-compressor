#!/bin/bash
set -eo pipefail
source /neural-compressor/.azure-pipelines/scripts/change_color.sh

# get parameters
PATTERN='[-a-zA-Z0-9_]*='

starttime=`date +'%Y-%m-%d %H:%M:%S'`

for i in "$@"
do
    case $i in
        --framework=*)
            framework=`echo $i | sed "s/${PATTERN}//"`;;
        --model=*)
            model=`echo $i | sed "s/${PATTERN}//"`;;
        --tuning_cmd=*)
            tuning_cmd=`echo $i | sed "s/${PATTERN}//"`;;
        --log_dir=*)
            log_dir=`echo $i | sed "s/${PATTERN}//"`;;
        --strategy=*)
            strategy=`echo $i | sed "s/${PATTERN}//"`;;
        *)
            echo "Parameter $i not recognized."; exit 1;;
    esac
done

# run tuning
if [ "${framework}" == "onnxrt" ]; then
    output_model=${log_dir}/${framework}-${model}-tune.onnx
elif [ "${framework}" == "mxnet" ]; then
    output_model=${log_dir}/resnet50_v1
else
    output_model=${log_dir}/${framework}-${model}-tune.pb
fi

$BOLD_YELLOW && echo -e "-------- run_tuning_common --------" && $RESET
$BOLD_YELLOW && echo ${tuning_cmd} && $RESET

eval "/usr/bin/time -v ${tuning_cmd} --output_model=${output_model}"

$BOLD_YELLOW && echo "====== finish tuning. echo information. ======" && $RESET
endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
$BOLD_GREEN && echo "Tuning time spend: "$((end_seconds-start_seconds))"s " && $RESET
$BOLD_GREEN && echo "Tuning strategy: ${strategy}" && $RESET
$BOLD_GREEN && echo "Total resident size (kbytes): $(cat /proc/meminfo | grep 'MemTotal' | sed 's/[^0-9]//g')" && $RESET
