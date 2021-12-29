## CIFAR10 Distillation Example
This example is used for distillation on CIFAR10 dataset. Use following commands to install requirements and execute demo of distillation of the WideResNet40-2 to the MobileNetV2-0.35.

```shell
# install dependencies
pip install -r requirements.txt
# for training of the teacher model WideResNet40-2
python train_without_distillation.py --epochs 200 --lr 0.1 --layers 40 --widen-factor 2 --name WideResNet-40-2 --tensorboard
# for distillation of the teacher model WideResNet40-2 to the student model MobileNetV2-0.35
python main.py --epochs 200 --lr 0.02 --name MobileNetV2-0.35-distillation --teacher_model runs/WideResNet-40-2/model_best.pth.tar --tensorboard --seed 9 
```