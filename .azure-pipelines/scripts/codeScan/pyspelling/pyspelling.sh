set -ex
pyspelling_dir="/neural-compressor/.azure-pipelines/scripts/codeScan"
pyspelling_log_dir="/neural-compressor/.azure-pipelines/scripts/codeScan/scanLog"

pip install pyspelling
pip install -r /neural-compressor/requirements.txt

apt-get install aspell -y
apt-get install aspell-en -y



sed -i "s|\${VAL_REPO}|$1|g" $pyspelling_dir/pyspelling/pyspelling_conf.yaml
sed -i "s|\${LPOT_REPO}|.|g" $pyspelling_dir/pyspelling/pyspelling_conf.yaml

pyspelling -c $pyspelling_dir/pyspelling/pyspelling_conf.yaml > $pyspelling_log_dir/lpot_pyspelling.log
exit_code=$?
if [ ${exit_code} -ne 0 ] ; then
    echo "Pyspelling exited with non-zero exit code."; exit 1
fi
exit 0


