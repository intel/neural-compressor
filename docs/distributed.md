Distributed Training
============

## Introduction

Neural Compressor uses [horovod](https://github.com/horovod/horovod) for distributed training.

## horovod installation

Please check horovod installation documentation and use following commands to install horovod:
```
pip install horovod
```

## Distributed training

Distributed training is supported in PyTorch currently, TensorFlow support is working in progress. To enable distributed training, the steps are:

1. Setting up distributed training scripts. We have 2 options here:
    - Option 1: Enable distributed training with pure yaml configuration. In this case, Neural Compressor builtin training function is used.
    - Option 2: Pass the user defined training function to Neural Compressor. In this case, please follow the horovod documentation and below example to know how to write such training function with horovod on different frameworks.
2. use horovodrun to execute your program.

### Option 1: pure yaml configuration

To enable distributed training in Neural Compressor, user only need to add a field: `Distributed: True` in dataloader configuration:

```
dataloader:
  batch_size: 30
  distributed: True
  dataset:
    ImageFolder:
      root: /path/to/dataset
```

In user's code, pass the yaml file to Neural Compressor components, in which it constructs the real dataloader for the distributed training. The example codes are as following:

```
from neural_compressor.experimental import Quantization, common
quantizer = Quantization(yaml_file)
quantizer.model = common.Model(model)
q_model = quantizer()
```

### Option2: user defined training function

Neural Compressor supports User defined training function for distributed training which requires user to modify training script following horovod requirements. We provide a MNIST example to show how to do that and following are the steps for PyTorch.

- Partition dataset via DistributedSampler:

```
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs, sampler=train_sampler)
```

- Wrap Optimizer:

```
optimizer = optim.Adadelta(model.parameters(), lr=args.lr * hvd.size())
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
```

- Broadcast parameters to processes from rank 0:

```
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)
```

- Prepare training function:

```
def train(args, model, train_loader, optimizer):
    model.train()
    for epoch in range(1, args.epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.sampler),
                    100. * batch_idx / len(train_loader.sampler), loss.item()))
                if args.dry_run:
                    break

def train_func(model):
    return train(args, model, train_loader, optimizer)
```

- Use user defined training function in Neural Compressor:

```
from neural_compressor.experimental import Component, common
component = Component(yaml_file)
component.model = common.Model(model)
component.train_func = train_func
component.eval_func = test_func
model = component()
```

### horovodrun

User needs to use horovodrun to execute distributed training. For more usage, please refer to [horovod documentation](https://horovod.readthedocs.io/en/stable/running_include.html).

For example, following command specified the number of processes and hosts to do distributed training.

```
horovodrun -np <num_of_processes> -H <hosts> python train.py
```

## security

horovodrun requires user set up SSH on all hosts without any prompts. To do distributed training with Neural Compressor, user needs to ensure the SSH setting on all hosts.

## Examples
Following PyTorch examples are supported:
- [MNIST](../examples/pytorch/eager/image_recognition/mnist/README.md)
- [QAT](../examples/pytorch/eager/image_recognition/imagenet/cpu/distributed/README.md)
