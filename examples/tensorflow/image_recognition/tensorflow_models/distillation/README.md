Details **TBD**
Please update conf.yaml with /PATH/TO/ImageNet
### Run pretraining
```shell
bash run_distillation.sh --topology=mobilenet --teacher=densenet201 --config=conf.yaml --output_model=path/to/output_model
```
