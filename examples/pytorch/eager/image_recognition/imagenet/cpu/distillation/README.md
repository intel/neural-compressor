Details **TBD**
Please update conf.yaml with /PATH/TO/ImageNet
### Prepare dataset
```shell
pip install -r requirements.txt
```
### Run pretraining
```shell
bash run_distillation.sh --topology=(resnet18|resnet34|resnet50|resnet101) --teacher=(resnet18|resnet34|resnet50|resnet101) --config=conf.yaml --output_model=path/to/output_model
```
