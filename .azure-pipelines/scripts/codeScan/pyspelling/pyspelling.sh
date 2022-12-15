#!/bin/bash

source /neural-compressor/.azure-pipelines/scripts/change_color.sh
RESET="echo -en \\E[0m \\n" # close color

work_dir="/neural-compressor/.azure-pipelines/scripts/codeScan/pyspelling"
log_dir="$work_dir/../scanLog"
mkdir -p $log_dir

pip install -r /neural-compressor/requirements.txt

sed -i "s|\${DICT_DIR}|$work_dir|g" $work_dir/pyspelling_conf.yaml
sed -i "s|\${REPO_DIR}|/neural-compressor|g" $work_dir/pyspelling_conf.yaml

pyspelling -c $work_dir/pyspelling_conf.yaml >$log_dir/pyspelling.log
exit_code=$?

$BOLD_YELLOW && echo "-------------------  Current log file output start --------------------------" && $RESET
cat $log_dir/pyspelling.log
$BOLD_YELLOW && echo "-------------------  Current log file output end ----------------------------" && $RESET

if [ ${exit_code} -ne 0 ]; then
    $BOLD_RED && echo "Error!! Please Click on the artifact button to download and view pyspelling error details." && $RESET
    exit 1
fi
$BOLD_PURPLE && echo "Congratulations, Pyspelling check passed!" && $LIGHT_PURPLE && echo "You can click on the artifact button to see the log details." && $RESET
exit 0
