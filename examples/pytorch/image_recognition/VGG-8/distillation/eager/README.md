## CIFAR100 Distillation Example
This example is used for distillation on CIFAR100 dataset. Use following commands to install requirements and execute demo of distillation of the VGG-13 to the VGG-8.

```shell
# install dependencies
pip install -r requirements.txt
# for training of the teacher model VGG-13
python train_without_distillation.py --model_type VGG-13 --epochs 200 --lr 0.02 --tensorboard
# for distillation of the student model VGG-8 with the teacher model VGG-13
python main.py --epochs 200 --lr 0.02 --name VGG-8-distillation --student_type VGG-8 --teacher_type VGG-13 --teacher_model runs/VGG-13/model_best.pth.tar --tensorboard --temperature 4 --loss_weights 0.2 0.8
```
