#!/bin/bash

for var in "$@"
do
  case $var in
    --scan_module=*)
        scan_module=$(echo $var |cut -f2 -d=)
    ;;
  esac
done

source /neural-compressor/.azure-pipelines/scripts/change_color.sh
RESET="echo -en \\E[0m \\n" # close color

log_dir="/neural-compressor/.azure-pipelines/scripts/codeScan/scanLog"
mkdir -p $log_dir

python -m bandit -r -lll -iii "/neural-compressor/${scan_module}" >$log_dir/bandit.log
exit_code=$?

$BOLD_YELLOW && echo " -----------------  Current bandit cmd start --------------------------" && $RESET
echo "python -m bandit -r -lll -iii  /neural-compressor/${scan_module} > $log_dir/bandit.log"
$BOLD_YELLOW && echo " -----------------  Current bandit cmd end --------------------------" && $RESET

$BOLD_YELLOW && echo " -----------------  Current log file output start --------------------------"
cat $log_dir/bandit.log
$BOLD_YELLOW && echo " -----------------  Current log file output end --------------------------" && $RESET

if [ ${exit_code} -ne 0 ]; then
    $BOLD_RED && echo "Error!! Please Click on the artifact button to download and view Bandit error details." && $RESET
    exit 1
fi
$BOLD_PURPLE && echo "Congratulations, Bandit check passed!" && $LIGHT_PURPLE && echo " You can click on the artifact button to see the log details." && $RESET
exit 0
