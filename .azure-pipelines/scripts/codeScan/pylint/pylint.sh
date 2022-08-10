# $1: $(VAL_PATH) 
# $2: $(target_path) 

pip install pylint==2.12.1
pip install -r requirements.txt


python -m pylint -f json --disable=R,C,W,E1129 --enable=line-too-long --max-line-length=120 --extension-pkg-whitelist=numpy --ignored-classes=TensorProto,NodeProto --ignored-modules=tensorflow,torch,torch.quantization,torch.tensor,torchvision,mxnet,onnx,onnxruntime ./$2
#  > /lpot-pylint.json
# python -m pylint -f json --disable=R,C,W,E1129 --enable=line-too-long --max-line-length=120 --extension-pkg-whitelist=numpy --ignored-classes=TensorProto,NodeProto --ignored-modules=tensorflow,torch,torch.quantization,torch.tensor,torchvision,mxnet,onnx,onnxruntime ./$2 > $1/pylint/lpot-pylint.json
   
    exit_code=$?
    if [ ${exit_code} -ne 0 ] ; then
        echo "PyLint exited with non-zero exit code."; exit 1
    fi
    exit 0