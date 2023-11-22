#!/bin/bash
set -eo pipefail
source /neural-compressor/.azure-pipelines/scripts/change_color.sh

# get parameters
PATTERN='[-a-zA-Z0-9_]*='

starttime=`date +'%Y-%m-%d %H:%M:%S'`

for i in "$@"
do
    case $i in
        --tuning_cmd=*)
            tuning_cmd=`echo $i | sed "s/${PATTERN}//"`;;
        --strategy=*)
            strategy=`echo $i | sed "s/${PATTERN}//"`;;
        *)
            echo "Parameter $i not recognized."; exit 1;;
    esac
done

eval "/usr/bin/time -v ${tuning_cmd}"

$BOLD_YELLOW && echo "====== finish tuning. echo information. ======" && $RESET
endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
$BOLD_GREEN && echo "Tuning time spend: "$((end_seconds-start_seconds))"s " && $RESET
$BOLD_GREEN && echo "Tuning strategy: ${strategy}" && $RESET
$BOLD_GREEN && echo "Total resident size (kbytes): $(cat /proc/meminfo | grep 'MemTotal' | sed 's/[^0-9]//g')" && $RESET
