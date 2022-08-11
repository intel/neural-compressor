pip install bandit

echo "-------------------"
pwd
echo "-------------------"
ls
echo "-------------------"

python -m bandit -r -lll -iii /neural_compressor >  /lpot-bandit.log
# python -m bandit -r -lll -iii $(TARGET_PATH) >  /lpot-bandit.log

exit_code=$?
if [ ${exit_code} -ne 0 ] ; then
    echo "Bandit exited with non-zero exit code."; exit 1
fi
exit 0