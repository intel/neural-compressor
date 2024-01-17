#!/bin/bash

for var in "$@"
do
  case $var in
    --scan_module=*)
        scan_module=$(echo $var |cut -f2 -d=)
    ;;
  esac
done

source /neural-compressor/.azure-pipelines/scripts/change_color.sh
RESET="echo -en \\E[0m \\n" # close color

log_dir="/neural-compressor/.azure-pipelines/scripts/codeScan/scanLog"
mkdir -p $log_dir

apt-get install -y --no-install-recommends --fix-missing \
        autoconf \
        build-essential

pip install -r /neural-compressor/requirements.txt
pip install cmake

pip install torch==1.12.0 \
            horovod \
            google \
            autograd \
            ofa \
            fvcore \
            pymoo \
            onnxruntime_extensions \
            tf_slim \
            transformers \
            accelerate \
            flask==2.1.3 \
            xgboost \
            datasets

if [ "${scan_module}" = "neural_solution" ]; then
    cd /neural-compressor
    python setup.py install

    echo "Install Neural Solution ... "
    bash /neural-compressor/.azure-pipelines/scripts/install_neural_solution.sh

elif [ "${scan_module}" = "neural_insights" ]; then
    cd /neural-compressor
    python setup.py install

    echo "Install Neural Insights ... "
    bash /neural-compressor/.azure-pipelines/scripts/install_neural_insights.sh

fi

python -m pylint -f json --disable=R,C,W,E1129 --enable=line-too-long --max-line-length=120 --extension-pkg-whitelist=numpy --ignored-classes=TensorProto,NodeProto \
--ignored-modules=tensorflow,keras,torch,torch.quantization,torch.tensor,torchvision,fairseq,mxnet,onnx,onnxruntime,intel_extension_for_pytorch,intel_extension_for_tensorflow,torchinfo,horovod,transformers \
/neural-compressor/${scan_module} > $log_dir/pylint.json

exit_code=$?

$BOLD_YELLOW && echo " -----------------  Current pylint cmd start --------------------------" && $RESET
echo "python -m pylint -f json --disable=R,C,W,E1129 --enable=line-too-long --max-line-length=120 --extension-pkg-whitelist=numpy --ignored-classes=TensorProto,NodeProto --ignored-modules=tensorflow,torch,torch.quantization,torch.tensor,torchvision,fairseq,mxnet,onnx,onnxruntime,intel_extension_for_pytorch,intel_extension_for_tensorflow,torchinfo,horovod,transformers
/neural-compressor/${scan_module}>$log_dir/pylint.json"
$BOLD_YELLOW && echo " -----------------  Current pylint cmd end --------------------------" && $RESET

$BOLD_YELLOW && echo " -----------------  Current log file output start --------------------------" && $RESET
cat $log_dir/pylint.json
$BOLD_YELLOW && echo " -----------------  Current log file output end --------------------------" && $RESET

if [ ${exit_code} -ne 0 ]; then
    $BOLD_RED && echo "Error!! Please Click on the artifact button to download and view Pylint error details." && $RESET
    exit 1
fi
$BOLD_PURPLE && echo "Congratulations, Pylint check passed!" && $LIGHT_PURPLE && echo " You can click on the artifact button to see the log details." && $RESET
exit 0
