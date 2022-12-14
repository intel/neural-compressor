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

We also supported Distributed Data Parallel training on single node and multi nodes settings for distillation. To use Distributed Data Parallel to speedup training, the bash command needs a small adjustment.
<br>
For example, bash command will look like the following, where *`<MASTER_ADDRESS>`* is the address of the master node, it won't be necessary for single node case, *`<NUM_PROCESSES_PER_NODE>`* is the desired processes to use in current node, for node with GPU, usually set to number of GPUs in this node, for node without GPU and use CPU for training, it's recommended set to 1, *`<NUM_NODES>`* is the number of nodes to use, *`<NODE_RANK>`* is the rank of the current node, rank starts from 0 to *`<NUM_NODES>`*`-1`.
<br>
Also please note that to use CPU for training in each node with multi nodes settings, argument `--no_cuda` is mandatory. In multi nodes setting, following command needs to be launched in each node, and all the commands should be the same except for *`<NODE_RANK>`*, which should be integer from 0 to *`<NUM_NODES>`*`-1` assigned to each node.

```bash
python -m torch.distributed.launch --master_addr=<MASTER_ADDRESS> --nproc_per_node=<NUM_PROCESSES_PER_NODE> --nnodes=<NUM_NODES> --node_rank=<NODE_RANK> \
   main.py --epochs 200 --lr 0.02 --name CNN-2-distillation --student_type CNN-2 --teacher_type CNN-10 --teacher_model runs/CNN-10/model_best.pth.tar --tensorboard
```