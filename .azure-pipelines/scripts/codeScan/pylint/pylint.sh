# $1: $(VAL_PATH) 
# $2: $(target_path) 

pip install pylint==2.12.1
pip install flask==2.1.3
pip install google
pip install horovod
pip install ofa
pip install fvcore
pip install autograd
pip install pymoo
pip install tf_slim
pip install transformers
pip install -r requirements.txt
pip install onnxruntime_extensions



python -m pylint -f json --disable=R,C,W,E1129 --enable=line-too-long --max-line-length=120 --extension-pkg-whitelist=numpy --ignored-classes=TensorProto,NodeProto --ignored-modules=tensorflow,torch,torch.quantization,torch.tensor,torchvision,mxnet,onnx,onnxruntime ./$2

exit_code=$?
if [ ${exit_code} -ne 0 ] ; then
    echo "PyLint exited with non-zero exit code."; exit 1
fi
exit 0