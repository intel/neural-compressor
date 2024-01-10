Step-by-Step
============
This document describes the step-by-step instructions for applying post training quantization on Segment Anything Model (SAM) using VOC dataset.

# Prerequisite
## Environment
```shell
# install dependencies
pip install -r ./requirements.txt
# retrieve SAM model codes and pre-trained weight
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

# PTQ
PTQ example on Segment Anything Model (SAM) using VOC dataset.

## 1. Prepare VOC dataset
```shell
python download_dataset.py
```

## 2. Start PTQ
```shell
bash run_quant.sh --voc_dataset_location=./voc_dataset/VOCdevkit/VOC2012/ --pretrained_weight_location=./sam_vit_b_01ec64.pth
```

## 3. Benchmarking
```shell
bash run_benchmark.sh --tuned_checkpoint=./saved_results --voc_dataset_location=./voc_dataset/VOCdevkit/VOC2012/ --int8=True --mode=performance
```

# Result
| | Baseline (FP32) | INT8 
| ------------- | ------------- | -------------
Dice| 0.7939  | 0.7849

# Saving and Loading Model

* Saving model:
  After tuning with Neural Compressor, we can get neural_compressor.model:

```
from neural_compressor import PostTrainingQuantConfig
from neural_compressor import quantization
conf = PostTrainingQuantConfig()
q_model = quantization.fit(model,
                            conf,
                            calib_dataloader=val_loader,
                            eval_func=eval_func)
```

Here, `q_model` is the Neural Compressor model class, so it has "save" API:

```python
q_model.save("Path_to_save_quantized_model")
```

* Loading model:

```python
from neural_compressor.utils.pytorch import load
quantized_model = load(os.path.abspath(os.path.expanduser(args.tuned_checkpoint)),
                        model,
                        dataloader=val_loader)
```

Please refer to main.py for reference.
