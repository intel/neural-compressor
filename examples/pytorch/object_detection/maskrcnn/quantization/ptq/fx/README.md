Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch MASK_RCNN tuning results with Intel® Neural Compressor.

# Prerequisite

### 1. Installation

#### Environment

PyTorch >=1.8 version is required with pytorch_fx backend.


```shell
cd examples/pytorch/object_detection/maskrcnn/quantization/ptq/fx
pip install -r requirements.txt
```

#### Maskrcnn_benchmark and Dependencies Installation

Make sure that your conda is setup properly with the right environment. Check that `which conda`, `which pip` and `which python` points to the right path from a clean conda env.
```shell
# this installs the right pip and dependencies for the fresh python
conda install ipython pip
# maskrcnn_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python

cd examples/pytorch/object_detection/maskrcnn/quantization/ptq/fx/pytorch
export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
git checkout 8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9
python setup.py build_ext install

# install cityscapesScripts
cd $INSTALL_DIR
git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts/
git checkout a7ac7b4062d1a80ed5e22d2ea2179c886801c77d
python setup.py build_ext install

# install PyTorch Detection
# some modifications have been made according to this example
# please apply the patch provided before installing
cd $INSTALL_DIR
git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
cd maskrcnn-benchmark
git checkout 57eec25b75144d9fb1a6857f32553e1574177daf
git apply ../maskrnn.patch
python setup.py build develop

cd ../..
unset INSTALL_DIR
```

### 2. Prepare Dataset

You can download COCO2017 dataset use script file:

```
source download_dataset.sh
```

Or you can download COCO2017 dataset to your local path, then link it to pytorch/datasets/coco:

```bash
ln -s /path/of/COCO2017/annotations pytorch/datasets/coco/annotations
ln -s /path/of/COCO2017/train2017 pytorch/datasets/coco/train2017
ln -s /path/of/COCO2017/val2017 pytorch/datasets/coco/val2017
```

### 3. Prepare weights

You can download weights with script file:

```bash
bash download_weights.sh
```

Or you else can link your weights to pytorch folder:

```bash
ln -s /path/of/weights pytorch/e2e_mask_rcnn_R_50_FPN_1x.pth
```

# Run

```shell
bash run_quant.sh --output_model=/path/to/tuned_checkpoint
```

# Saving and loading model:

* Saving model:
  After tuning with Neural Compressor, we can get neural_compressor.model:

```
from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor import quantization
conf = PostTrainingQuantConfig()
q_model = quantization.fit(model,
                            conf,
                            calib_dataloader=cal_dataloader,
                            eval_func=eval_func)
```
Here, q_model is Neural Compressor model class, so it has "save" API:

```python
q_model.save("Path_to_save_configure_file")
```

* loading model:

```python
from neural_compressor.utils.pytorch import load
from neural_compressor.utils.pytorch import load
q_model = load(os.path.abspath(os.path.expanduser(args.tuned_checkpoint)),
            fp32_model,
            dataloader=your_dataloader)
```
Please refer to [Sample code](./pytorch/tools/test_net.py)
