source /neural-compressor/.azure-pipelines/scripts/change_color.sh

pip install coverage
export COVERAGE_RCFILE=/neural-compressor/.azure-pipelines/scripts/ut/3x/coverage.${1}
coverage_log="/neural-compressor/log_dir/coverage_log"
coverage_log_base="/neural-compressor/log_dir/coverage_log_base"
coverage_compare="/neural-compressor/log_dir/coverage_compare.html"
cd /neural-compressor/log_dir

$BOLD_YELLOW && echo "collect coverage for PR branch" && $RESET
cp ut_coverage_3x/.coverage /neural-compressor/
mkdir -p coverage_PR
cd /neural-compressor
coverage report -m --rcfile=${COVERAGE_RCFILE} | tee ${coverage_log}
coverage html -d log_dir/coverage_PR/htmlcov --rcfile=${COVERAGE_RCFILE}
coverage xml -o log_dir/coverage_PR/coverage.xml --rcfile=${COVERAGE_RCFILE}
ls -l log_dir/coverage_PR/htmlcov


$BOLD_YELLOW && echo "collect coverage for baseline" && $RESET
cd /neural-compressor
git config --global --add safe.directory /neural-compressor
git fetch
git checkout master
echo y | pip uninstall neural-compressor
cd /neural-compressor/.azure-pipelines/scripts && bash install_nc.sh ${1}

coverage erase
cd /neural-compressor/log_dir
mkdir -p coverage_base
rm -rf /neural-compressor/.coverage || true
cp ut_coverage_3x_baseline/.coverage /neural-compressor

cd /neural-compressor
coverage report -m --rcfile=${COVERAGE_RCFILE} | tee ${coverage_log_base}
coverage html -d log_dir/coverage_base/htmlcov --rcfile=${COVERAGE_RCFILE}
coverage xml -o log_dir/coverage_base/coverage.xml --rcfile=${COVERAGE_RCFILE}
ls -l log_dir/coverage_base/htmlcov

get_coverage_data() {
    # Input argument
    local coverage_xml="$1"

    # Get coverage data
    local coverage_data=$(python3 -c "import xml.etree.ElementTree as ET; root = ET.parse('$coverage_xml').getroot(); print(ET.tostring(root).decode())")
    if [[ -z "$coverage_data" ]]; then
        echo "Failed to get coverage data from $coverage_xml."
        exit 1
    fi

    # Get lines coverage
    local lines_covered=$(echo "$coverage_data" | grep -o 'lines-covered="[0-9]*"' | cut -d '"' -f 2)
    local lines_valid=$(echo "$coverage_data" | grep -o 'lines-valid="[0-9]*"' | cut -d '"' -f 2)
    if [ $lines_valid == 0 ]; then
        local lines_coverage=0
    else
        local lines_coverage=$(awk "BEGIN {printf \"%.3f\", 100 * $lines_covered / $lines_valid}")
    fi

    # Get branches coverage
    local branches_covered=$(echo "$coverage_data" | grep -o 'branches-covered="[0-9]*"' | cut -d '"' -f 2)
    local branches_valid=$(echo "$coverage_data" | grep -o 'branches-valid="[0-9]*"' | cut -d '"' -f 2)
    if [ $branches_valid == 0 ]; then
        local branches_coverage=0
    else
        local branches_coverage=$(awk "BEGIN {printf \"%.3f\", 100 * $branches_covered/$branches_valid}")
    fi

    # Return values
    echo "$lines_covered $lines_valid $lines_coverage $branches_covered $branches_valid $branches_coverage"
}

$BOLD_YELLOW && echo "compare coverage" && $RESET

coverage_PR_xml="log_dir/coverage_PR/coverage.xml"
coverage_PR_data=$(get_coverage_data $coverage_PR_xml)
read lines_PR_covered lines_PR_valid coverage_PR_lines_rate branches_PR_covered branches_PR_valid coverage_PR_branches_rate <<<"$coverage_PR_data"

coverage_base_xml="log_dir/coverage_base/coverage.xml"
coverage_base_data=$(get_coverage_data $coverage_base_xml)
read lines_base_covered lines_base_valid coverage_base_lines_rate branches_base_covered branches_base_valid coverage_base_branches_rate <<<"$coverage_base_data"

$BOLD_BLUE && echo "PR lines coverage: $lines_PR_covered/$lines_PR_valid ($coverage_PR_lines_rate%)" && $RESET
$BOLD_BLUE && echo "PR branches coverage: $branches_PR_covered/$branches_PR_valid ($coverage_PR_branches_rate%)" && $RESET
$BOLD_BLUE && echo "BASE lines coverage: $lines_base_covered/$lines_base_valid ($coverage_base_lines_rate%)" && $RESET
$BOLD_BLUE && echo "BASE branches coverage: $branches_base_covered/$branches_base_valid ($coverage_base_branches_rate%)" && $RESET

$BOLD_YELLOW && echo "clear upload path" && $RESET
rm -fr log_dir/coverage_PR/.coverage*
rm -fr log_dir/coverage_base/.coverage*
rm -fr log_dir/ut-coverage-*

# Declare an array to hold failed items
declare -a fail_items=()

if (( $(bc -l <<< "${coverage_PR_lines_rate}+0.05 < ${coverage_base_lines_rate}") )); then
    fail_items+=("lines")
fi
if (( $(bc -l <<< "${coverage_PR_branches_rate}+0.05 < ${coverage_base_branches_rate}") )); then
    fail_items+=("branches")
fi

if [[ ${#fail_items[@]} -ne 0 ]]; then
    fail_items_str=$(
        IFS=', '
        echo "${fail_items[*]}"
    )
    for item in "${fail_items[@]}"; do
        case "$item" in
        lines)
            decrease=$(echo $(printf "%.3f" $(echo "$coverage_PR_lines_rate - $coverage_base_lines_rate" | bc -l)))
            ;;
        branches)
            decrease=$(echo $(printf "%.3f" $(echo "$coverage_PR_branches_rate - $coverage_base_branches_rate" | bc -l)))
            ;;
        *)
            echo "Unknown item: $item"
            continue
            ;;
        esac
        $BOLD_RED && echo "Unit Test failed with ${item} coverage decrease ${decrease}%" && $RESET
    done
    $BOLD_RED && echo "compare coverage to give detail info" && $RESET
    bash /neural-compressor/.azure-pipelines/scripts/ut/compare_coverage.sh ${coverage_compare} ${coverage_log} ${coverage_log_base} "FAILED" ${coverage_PR_lines_rate} ${coverage_base_lines_rate} ${coverage_PR_branches_rate} ${coverage_base_branches_rate}
    exit 1
else
    $BOLD_GREEN && echo "Unit Test success with coverage lines: ${coverage_PR_lines_rate}%, branches: ${coverage_PR_branches_rate}%" && $RESET
    $BOLD_GREEN && echo "compare coverage to give detail info" && $RESET
    bash /neural-compressor/.azure-pipelines/scripts/ut/compare_coverage.sh ${coverage_compare} ${coverage_log} ${coverage_log_base} "SUCCESS" ${coverage_PR_lines_rate} ${coverage_base_lines_rate} ${coverage_PR_branches_rate} ${coverage_base_branches_rate}
fi
