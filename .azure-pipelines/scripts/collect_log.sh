
export COVERAGE_RCFILE=/neural-compressor/.azure-pipelines/scripts/ut/coverage.file
cd /neural-compressor/log_dir
cp ut-coverage-strategy/.coverage.strategy ./
cp ut-coverage-util/.coverage.util ./
coverage combine --keep --rcfile=${COVERAGE_RCFILE}
cp .coverage /neural-compressor/.coverage
cd /neural-compressor
coverage report -m --rcfile=${COVERAGE_RCFILE}
coverage html -d log_dir/htmlcov --rcfile=${COVERAGE_RCFILE}
ls -l log_dir/htmlcov
cat log_dir/htmlcov/index.html
