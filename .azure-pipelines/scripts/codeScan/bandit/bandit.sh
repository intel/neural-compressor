#!/bin/bash

source /neural-compressor/.azure-pipelines/scripts/change_color.sh
mkdir -p /neural-compressor/.azure-pipelines/scripts/codeScan/scanLog
bandit_log_dir="/neural-compressor/.azure-pipelines/scripts/codeScan/scanLog"

python -m bandit -r -lll -iii  /neural-compressor/neural_compressor >  $bandit_log_dir/lpot-bandit.log
exit_code=$?

# code-scan close 
RESET="echo -en \\E[0m \\n"

$BOLD_YELLOW && echo " -----------------  Current log file output start --------------------------"
cat  $bandit_log_dir/lpot-bandit.log
$BOLD_YELLOW && echo " -----------------  Current log file output end --------------------------" && $RESET


if [ ${exit_code} -ne 0 ] ; then
    $BOLD_RED && echo "Error!! Please Click on the artifact button to download and view Bandit error details." && $RESET; exit 1
fi
$BOLD_PURPLE && echo "Congratulations, Bandit check passed!" && $LIGHT_PURPLE && echo " You can click on the artifact button to see the log details." && $RESET; exit 0
