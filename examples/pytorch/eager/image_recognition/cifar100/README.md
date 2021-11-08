## CIFAR100 Distillation Example
This example is used for distillation on CIFAR100 dataset. Use following commands to install requirements and execute demo of distillation of the CNN-10 or VGG-13 to the CNN-2 or VGG-8.

```shell
pip install -r requirements.txt
# for CNN-10 teacher and CNN-2 student pair
python train_without_distillation.py --model_type CNN-10 --epochs 200 --lr 0.1 --tensorboard
python main.py --epochs 200 --lr 0.02 --name CNN-2-distillation --student_type CNN-2 --teacher_type CNN-10 --teacher_model runs/CNN-10/model_best.pth.tar --tensorboard
# for VGG-13 teacher and VGG-8 student pair
python train_without_distillation.py --model_type VGG-13 --epochs 200 --lr 0.02 --tensorboard
python main.py --epochs 200 --lr 0.02 --name VGG-8-distillation --student_type VGG-8 --teacher_type VGG-13 --teacher_model runs/VGG-13/model_best.pth.tar --tensorboard
```
