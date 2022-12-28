Details **TBD**
### Prepare requirements
```shell
pip install -r requirements.txt
```
### Run self distillation
```shell
bash run_distillation.sh --topology=(resnet18|resnet34|resnet50|resnet101) --output_model=path/to/output_model --dataset_location=path/to/dataset --use_cpu=(0|1)
```
### CIFAR100 benchmark
https://github.com/weiaicunzai/pytorch-cifar100

### Paper:
[Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation](https://openaccess.thecvf.com/content_ICCV_2019/html/Zhang_Be_Your_Own_Teacher_Improve_the_Performance_of_Convolutional_Neural_ICCV_2019_paper.html)

[Self-Distillation: Towards Efficient and Compact Neural Networks](https://ieeexplore.ieee.org/document/9381661)

### Our results in CIFAR100
| model    | Baseline | Classifier1 | Classifier2 | Classifier3 | Classifier4 | Ensemble |
| :------: | :-------:| :---------: | :---------: | :---------: | :---------: | :------: |
| Resnet50 |  80.88   |    82.06    |   83.64     |    83.85    |    83.41    |  85.10   |

