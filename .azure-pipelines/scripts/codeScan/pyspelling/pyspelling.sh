# $1: $(VAL_PATH) 
# $2: $(BUILD_DIRECTORY) - $(Build.SourcesDirectory)

pip install pyspelling

pip install aspell
pip install aspell-en 

pip install -r requirements.txt


sed -i "s|\${VAL_REPO}|$1|g" $1/pyspelling/pyspelling_conf.yaml
sed -i "s|\${LPOT_REPO}|.|g" $1/pyspelling/pyspelling_conf.yaml
echo "Modified config:"
cat $1/pyspelling/pyspelling_conf.yaml
pyspelling -c $1/pyspelling/pyspelling_conf.yaml > $1/pyspelling/pyspelling_output.log
exit_code=$?
if [ ${exit_code} -ne 0 ] ; then
    echo "Pyspelling exited with non-zero exit code."; exit 1
fi
exit 0


