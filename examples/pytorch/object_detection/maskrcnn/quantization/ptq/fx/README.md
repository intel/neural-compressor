Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch MASK_RCNN tuning results with Intel® Neural Compressor.

# Prerequisite

### 1. Installation

PyTorch >=1.8 version is required with pytorch_fx backend.

```shell
cd examples/pytorch/object_detection/maskrcnn/quantization/ptq/fx
pip install -r requirements.txt
bash install.sh
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
bash run_tuning.sh --output_model=/path/to/tuned_checkpoint
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
