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

work_dir="/neural-compressor/.azure-pipelines/scripts/codeScan/pydocstyle"
log_dir="$work_dir/../scanLog"
mkdir -p $log_dir

scan_path="scan_path.txt"
if [ "${scan_module}" = "neural_solution" ]; then
    scan_path="scan_path_neural_solution.txt"
elif [ "${scan_module}" = "neural_insights" ]; then
    scan_path="scan_path_neural_insights.txt"
fi

exit_code=0
for line in $(cat ${work_dir}/${scan_path})
do
    pydocstyle --convention=google $line >> $log_dir/pydocstyle.log
    if [ $? -ne 0 ]; then
      exit_code=1
    fi
done

$BOLD_YELLOW && echo " -----------------  Current pydocstyle cmd start --------------------------" && $RESET
echo "pydocstyle --convention=google \$line > $log_dir/pydocstyle.log"
$BOLD_YELLOW && echo " -----------------  Current pydocstyle cmd end --------------------------" && $RESET

$BOLD_YELLOW && echo " -----------------  Current log file output start --------------------------"
cat $log_dir/pydocstyle.log
$BOLD_YELLOW && echo " -----------------  Current log file output end --------------------------" && $RESET

if [ ${exit_code} -ne 0 ]; then
    $BOLD_RED && echo "Error!! Please Click on the artifact button to download and view DocStyle error details." && $RESET
    exit 1
fi
$BOLD_PURPLE && echo "Congratulations, DocStyle check passed!" && $LIGHT_PURPLE && echo " You can click on the artifact button to see the log details." && $RESET
exit 0
