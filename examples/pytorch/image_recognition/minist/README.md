## Distributed MNIST Example
This example is used for distributed training on MNIST dataset. Use following commands to install requirements and execute demo on multi-nodes. For horovod usage, please refer to [horovod](https://github.com/horovod/horovod).

```
pip install -r requirements.txt
horovodrun -np <num_of_processes> -H <hosts> python mnist.py
```
