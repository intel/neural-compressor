Details **TBD**
Please update conf.yaml with /PATH/TO/ImageNet
### Prepare dataset
```shell
pip install -r requiremnets.txt
```
### Run pretraining
```shell
bash run_pruning.sh --topology=(resnet18|resnet34|resnet50|resnet101) --config=conf.yaml --output_model=path/to/output_model
```
