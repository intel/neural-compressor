# Distillation for torchvision models

This is an example to show the usage of distillation.

## Environment
```shell
pip install -r requirements.txt
```

## Run distillation
```shell
bash run_distillation.sh --topology=(resnet18|resnet34|resnet50|resnet101) --teacher=(resnet18|resnet34|resnet50|resnet101)  --dataset_location=(path to dataset) --output_model=path/to/output_model
```

> Note: `--topology` is the student model and `--teacher` is the teacher model.
