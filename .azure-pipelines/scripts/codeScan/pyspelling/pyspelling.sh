pip install pyspelling
pip install -r requirements.txt

apt-get install aspell -y
apt-get install aspell-en -y



sed -i "s|\${VAL_REPO}|$1|g" $1/.azure-pipelines/scripts/codeScan/pyspelling_conf.yaml
sed -i "s|\${LPOT_REPO}|.|g" $1/.azure-pipelines/scripts/codeScan/pyspelling_conf.yaml
echo "Modified config:"
cat $1/.azure-pipelines/scripts/codeScan/pyspelling_conf.yaml
pyspelling -c $1/.azure-pipelines/scripts/codeScan/pyspelling_conf.yaml > /lpot_pyspelling.log
exit_code=$?
if [ ${exit_code} -ne 0 ] ; then
    echo "Pyspelling exited with non-zero exit code."; exit 1
fi
exit 0


