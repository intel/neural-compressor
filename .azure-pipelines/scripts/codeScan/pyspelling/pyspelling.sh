pip install pyspelling
pip install -r requirements.txt

apt-get install aspell -y
apt-get install aspell-en -y



sed -i "s|\${VAL_REPO}|$1|g" /neural_compressor/.azure-pipelines/scripts/codeScan/pyspelling/pyspelling_conf.yaml
sed -i "s|\${LPOT_REPO}|.|g" /neural_compressor/.azure-pipelines/scripts/codeScan/pyspelling/pyspelling_conf.yaml
echo "Modified config:"
cat /neural_compressor/.azure-pipelines/scripts/codeScan/pyspelling/pyspelling_conf.yaml
pyspelling -c /neural_compressor/.azure-pipelines/scripts/codeScan/pyspelling/pyspelling_conf.yaml > /lpot_pyspelling.log
exit_code=$?
if [ ${exit_code} -ne 0 ] ; then
    echo "Pyspelling exited with non-zero exit code."; exit 1
fi
exit 0


