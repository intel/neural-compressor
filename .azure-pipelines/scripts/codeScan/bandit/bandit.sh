set -ex
bandit_log_dir="/neural-compressor/.azure-pipelines/scripts/codeScan/scanLog"
pip install bandit

python -m bandit -r -lll -iii /neural-compressor/neural_compressor >  $bandit_log_dir/lpot-bandit.log

exit_code=$?
if [ ${exit_code} -eq 0 ] ; then
    $1 = 1;
    echo "Bandit exited with non-zero exit code."; exit 1
fi
exit 0

