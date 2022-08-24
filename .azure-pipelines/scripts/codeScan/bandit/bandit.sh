#!/bin/bash
set -e

mkdir -p /neural-compressor/.azure-pipelines/scripts/codeScan/scanLog
bandit_log_dir="/neural-compressor/.azure-pipelines/scripts/codeScan/scanLog"

# error
RED="echo -en \\E[1;31m"
# succeed
PURPLE="echo -en \\E[1;35m"
# succeed hint
LIGHT_PURPLE="echo -en \\E[35m"
# log output hint
YELLOW="echo -en \\E[1;33m"
# close 
RESET="echo -en \\E[0m \\n"

python -m bandit -r -lll -iii /neural-compressor/neural_compressor >  $bandit_log_dir/lpot-bandit.log

$YELLOW && echo " -----------------  Current log file output start --------------------------"
cat  $bandit_log_dir/lpot-bandit.log
$YELLOW && echo " -----------------   Current log file output end --------------------------" && $RESET

exit_code=$?
if [ ${exit_code} -ne 0 ] ; then
    $RED && echo "Error!! Please Click on the artifact button to download and view Bandit error details. " && $RESET; exit 1
fi
$PURPLE && echo "Congratulations, Bandit check passed!" && $LIGHT_PURPLE && echo " You can click on the artifact button to see the log details." && $RESET; exit 0

