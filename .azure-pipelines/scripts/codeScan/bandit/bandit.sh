pip install bandit

echo "--------------------"
$1
echo "--------------------"

python -m bandit -r -lll -iii $1/neural_compressor >  /lpot-bandit.log

exit 1
exit_code=$?
if [ ${exit_code} -ne 0 ] ; then
    echo "Bandit exited with non-zero exit code."; exit 1
fi
exit 0