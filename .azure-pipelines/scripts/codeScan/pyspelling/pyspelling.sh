#!/bin/bash
set -e
source /neural-compressor/.azure-pipelines/scripts/change_color.sh

mkdir -p /neural-compressor/.azure-pipelines/scripts/codeScan/scanLog
pyspelling_dir="/neural-compressor/.azure-pipelines/scripts/codeScan"
pyspelling_log_dir="/neural-compressor/.azure-pipelines/scripts/codeScan/scanLog"

pip install -r /neural-compressor/requirements.txt

sed -i "s|\${VAL_REPO}|$pyspelling_dir|g" $pyspelling_dir/pyspelling/pyspelling_conf.yaml
sed -i "s|\${LPOT_REPO}|/neural-compressor|g" $pyspelling_dir/pyspelling/pyspelling_conf.yaml

pyspelling -c $pyspelling_dir/pyspelling/pyspelling_conf.yaml > $pyspelling_log_dir/lpot_pyspelling.log
# code-scan close 
RESET="echo -en \\E[0m \\n"

$BOLD_YELLOW && echo "-------------------  Current log file output start --------------------------"
cat  $pyspelling_log_dir/lpot_pyspelling.log
$BOLD_YELLOW && echo "-------------------  Current log file output end ----------------------------" && $RESET

exit_code=$?
if [ ${exit_code} -ne 0 ] ; then
    $BOLD_RED && echo "Error!! Please Click on the artifact button to download and view Pyspelling error details." && $RESET; exit 1
fi
$BOLD_PURPLE && echo "Congratulations, Pyspelling check passed!" && $LIGHT_PURPLE && echo "You can click on the artifact button to see the log details." && $RESET; exit 0

