Step-by-Step
============
This document describes the step-by-step instructions for reproducing Huggingface models with IPEX backend tuning results with Intel® Neural Compressor.
> Note: IPEX version >= 1.10

# Prerequisite

## 1. Environment
Recommend python 3.6 or higher version.
```shell
cd examples/pytorch/nlp/huggingface_models/question-answering/quantization/ptq_static/ipex
pip install -r requirements.txt
pip install torch
python -m pip install intel_extension_for_pytorch -f https://software.intel.com/ipex-whl-stable
```
> Note: Intel® Extension for PyTorch* has PyTorch version requirement. Please check more detailed information via the URL below.

# Quantization

## 1. Quantization with CPU
If IPEX version is equal or higher than 1.12, please install transformers 4.19.0.
```shell
python run_qa.py 
    --model_name_or_path bert-large-uncased-whole-word-masking-finetuned-squad \
    --dataset_name squad \
    --do_eval \
    --max_seq_length 384 \
    --doc_stride 128 \
    --no_cuda \
    --tune \
    --output_dir ./savedresult
```
> NOTE
>
> /path/to/checkpoint/dir is the path to finetune output_dir

## 2. Quantization with XPU
Please build an IPEX docker container with following steps. Please also refer to the [official guide](https://github.com/intel/intel-extension-for-pytorch/tree/xpu-master/docker).
#### 2.1 Build Container and Environment Variables
```bash
wget https://raw.githubusercontent.com/intel/intel-extension-for-pytorch/xpu-master/docker/Dockerfile.xpu
wget https://raw.githubusercontent.com/intel/intel-extension-for-pytorch/xpu-master/docker/build.sh
./build.sh xpu-flex

export IMAGE_NAME=intel/intel-extension-for-pytorch:xpu-flex
export VIDEO=$(getent group video | sed -E 's,^video:[^:]*:([^:]*):.*$,\1,')
export RENDER=$(getent group render | sed -E 's,^render:[^:]*:([^:]*):.*$,\1,')
test -z "$RENDER" || RENDER_GROUP="--group-add ${RENDER}"
```

#### 2.2 Run Container
```bash
docker run --rm \
    -v .:/workspace \
    --group-add ${VIDEO} \
    ${RENDER_GROUP} \
    --device=/dev/dri \
    --ipc=host \
    -e http_proxy=$http_proxy \
    -e https_proxy=$https_proxy \
    -e no_proxy=$no_proxy \
    -it $IMAGE_NAME bash
```

#### 2.3 Environment Settings
Source installed torch is required:
```bash
git clone https://github.com/intel-innersource/frameworks.ai.pytorch.private-gpu.git --depth=1
cd frameworks.ai.pytorch.private-gpu/
git submodule sync
git submodule update --init --recursive
pip install cmake ninja
pip install -r requirements.txt
apt-get update
apt-get install python3-dev
python setup.py install
```
Please set basekit configurations as following:
```bash
bash l_BaseKit_p_2024.0.0.49261_offline.sh -a -s --eula accept --components intel.oneapi.lin.tbb.devel:intel.oneapi.lin.ccl.devel:intel.oneapi.lin.mkl.devel:intel.oneapi.lin.dpcpp-cpp-compiler --install-dir ${HOME}/intel/oneapi
source ./20240921_xmainrel/env/vars.sh
# source ${HOME}/intel/oneapi/compiler/latest/env/vars.sh
source ${HOME}/intel/oneapi/mkl/latest/env/vars.sh
source ${HOME}/intel/oneapi/tbb/latest/env/vars.sh
export MKL_DPCPP_ROOT=${MKLROOT}
export LD_LIBRARY_PATH=${MKL_DPCPP_ROOT}/lib:${MKL_DPCPP_ROOT}/lib64:${MKL_DPCPP_ROOT}/lib/intel64:${LD_LIBRARY_PATH}
export LIBRARY_PATH=${MKL_DPCPP_ROOT}/lib:${MKL_DPCPP_ROOT}/lib64:${MKL_DPCPP_ROOT}/lib/intel64:$LIBRARY_PATH
```
Source installed IPEX:
```bash
git clone -b master --depth=1 --recursive https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-gpu.git
cd frameworks.ai.pytorch.ipex-gpu
pip install -r requirements.txt
python setup.py install
```

#### 2.4 Quantization Command
```shell
python run_qa.py 
    --model_name_or_path bert-large-uncased-whole-word-masking-finetuned-squad \
    --dataset_name squad \
    --do_eval \
    --max_seq_length 384 \
    --doc_stride 128 \
    --xpu \
    --tune \
    --output_dir ./savedresult
```

# Tutorial of How to Enable NLP Model with Intel® Neural Compressor.
### Intel® Neural Compressor supports two usages:

1. User specifies fp32 'model', calibration dataset 'q_dataloader', evaluation dataset "eval_dataloader" and metrics.
2. User specifies fp32 'model', calibration dataset 'q_dataloader' and a custom "eval_func" which encapsulates the evaluation dataset and metrics by itself.

As MRPC's metrics are 'f1', 'acc_and_f1', mcc', 'spearmanr', 'acc', so customer should provide evaluation function 'eval_func', it's suitable for the second use case.

### Code Prepare

We just need update run_qa.py like below

```python
import intel_extension_for_pytorch as ipex

# Initialize our Trainer
trainer = QuestionAnsweringTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    eval_examples=eval_examples if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    post_process_function=post_processing_function,
    compute_metrics=compute_metrics,
)

eval_dataloader = trainer.get_eval_dataloader()
batch_size = eval_dataloader.batch_size
metric_name = "eval_f1"
def take_eval_steps(model, trainer, metric_name, save_metrics=False):
    trainer.model = model
    metrics = trainer.evaluate()
    return metrics.get(metric_name)

def eval_func(model):
    return take_eval_steps(model, trainer, metric_name)

ipex.nn.utils._model_convert.replace_dropout_with_identity(model)
from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor import quantization
conf = PostTrainingQuantConfig(backend="ipex", calibration_sampling_size=800)
q_model = quantization.fit(model,
                           conf,
                           calib_dataloader=eval_dataloader,
                           eval_func=eval_func)
q_model.save(training_args.output_dir)
```


