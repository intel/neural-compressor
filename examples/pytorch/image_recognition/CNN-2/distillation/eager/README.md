## CIFAR100 Distillation Example
This example is used for distillation on CIFAR100 dataset. Use following commands to install requirements and execute demo of distillation of the CNN-10 to the CNN-2.

```shell
# install dependencies
pip install -r requirements.txt
# for training of the teacher model CNN-10
python train_without_distillation.py --model_type CNN-10 --epochs 200 --lr 0.1 --tensorboard
# for distillation of the student model CNN-2 with the teacher model CNN-10
python main.py --epochs 200 --lr 0.02 --name CNN-2-distillation --student_type CNN-2 --teacher_type CNN-10 --teacher_model runs/CNN-10/model_best.pth.tar --tensorboard
```
