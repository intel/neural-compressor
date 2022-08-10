# $1: $(VAL_PATH) 
# $2: $(target_path) 

pip install bandit


python -m bandit -r -lll -iii $2 >  /lpot-bandit.log

exit_code=$?
if [ ${exit_code} -ne 0 ] ; then
    echo "Bandit exited with non-zero exit code."; exit 1
fi
exit 0