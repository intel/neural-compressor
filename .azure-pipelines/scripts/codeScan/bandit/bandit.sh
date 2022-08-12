# set -ex
bandit_log_dir="/neural-compressor/.azure-pipelines/scripts/codeScan/scanLog"
pip install bandit

python -m bandit -r -lll -iii /neural-compressor/neural_compressor >  $bandit_log_dir/lpot-bandit.log

exit_code=$?
if [ ${exit_code} -eq 0 ] ; then
    sed -i "s|CURRENT_STATUS\: true|CURRENT_STATUS\: false|g" /neural-compressor/azure-pipelines.yml
    sed -i "s|ER|QR|g" /neural-compressor/azure-pipelines.yml

    # sed -i 's/CURRENT_STATUS:.*$/CURRENT_STATUS: false' /neural-compressor/azure-pipelines.yml
    echo "Bandit exited with non-zero exit code."; exit 1
fi
exit 0

