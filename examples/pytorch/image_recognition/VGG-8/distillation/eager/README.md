Step-by-Step
============
This document describes the step-by-step instructions for reproducing the distillation on CIFAR100 dataset. The example is a distillation of the VGG-13 to the VGG-8.

# Prerequisite
## Environment
```shell
# install dependencies
cd examples/pytorch/image_recognition/VGG-8/distillation/eager
pip install -r requirements.txt
```

# Distillation
Distillation examples on CIFAR100.

## 1. Train Teacher Model
```shell
# for training of the teacher model CNN-10
python train_without_distillation.py --model_type VGG-13 --epochs 200 --lr 0.02 --tensorboard
```

## 2. Distilling The Student Model with The Teacher Model
```shell
# for distillation of the student model VGG-8 with the teacher model VGG-13
python main.py --epochs 200 --lr 0.02 --name VGG-8-distillation --student_type VGG-8 --teacher_type VGG-13 --teacher_model runs/VGG-13/model_best.pth.tar --tensorboard --temperature 4 --loss_weights 0.2 0.8
```
