Step-by-Step
============

This is an example to show the usage of distillation.

# Prerequisite

## 1. Environment
```shell
pip install -r requirements.txt
```

## 2. Prepare Dataset

Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/imagenet.  The dir include below folder:

```bash
ls /path/to/imagenet
train  val
```

# Run

Command is shown as below:

```shell
bash run_distillation.sh --topology=(resnet18|resnet34|resnet50|resnet101) --teacher=(resnet18|resnet34|resnet50|resnet101)  --dataset_location=/path/to/imagenet --output_model=path/to/output_model
```

> Note: `--topology` is the student model and `--teacher` is the teacher model.
