pip install pyspelling
pip install -r requirements.txt

apt-get install aspell -y
apt-get install aspell-en -y


sed -i "s|\${VAL_REPO}|${TARGET_PATH}|g" ${TARGET_PATH}/${PYSPELLING_CONTAINER_NAME}pyspelling_conf.yaml
sed -i "s|\${LPOT_REPO}|.|g" ${TARGET_PATH}/${PYSPELLING_CONTAINER_NAME}pyspelling_conf.yaml
echo "Modified config:"
cat ${TARGET_PATH}/${PYSPELLING_CONTAINER_NAME}pyspelling_conf.yaml
pyspelling -c ${TARGET_PATH}/${PYSPELLING_CONTAINER_NAME}pyspelling_conf.yaml > /lpot_pyspelling.log
exit_code=$?
if [ ${exit_code} -ne 0 ] ; then
    echo "Pyspelling exited with non-zero exit code."; exit 1
fi
exit 0


