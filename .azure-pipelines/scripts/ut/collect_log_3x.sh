#!/bin/bash
source ${BUILD_SOURCESDIRECTORY}/.azure-pipelines/scripts/change_color.sh

set -e
pip install coverage
export COVERAGE_RCFILE=${BUILD_SOURCESDIRECTORY}/.azure-pipelines/scripts/ut/coverage.${1}
coverage_log="${BUILD_SOURCESDIRECTORY}/log_dir/coverage_log"
cd ${BUILD_SOURCESDIRECTORY}/log_dir

$BOLD_YELLOW && echo "collect coverage for PR branch" && $RESET
cp ut_3x_coverage/.coverage ${BUILD_SOURCESDIRECTORY}/
mkdir -p coverage_PR
cd ${BUILD_SOURCESDIRECTORY}
coverage report -m --rcfile=${COVERAGE_RCFILE} | tee ${coverage_log}
coverage html -d log_dir/coverage_PR/htmlcov --rcfile=${COVERAGE_RCFILE}
coverage xml -o log_dir/coverage_PR/coverage.xml --rcfile=${COVERAGE_RCFILE}
ls -l log_dir/coverage_PR/htmlcov

$BOLD_YELLOW && echo "clear upload path" && $RESET
rm -fr log_dir/coverage_PR/.coverage*
rm -fr log_dir/ut-coverage-*
