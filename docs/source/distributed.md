Distributed Training and Inference (Evaluation)
============

1. [Introduction](#introduction)
2. [Supported Feature Matrix](#supported-feature-matrix)
3. [Get Started with Distributed Training and Inference API](#get-started-with-distributed-training-and-inference-api)

    3.1. [Option 1: Pure Yaml Configuration](#option-1-pure-yaml-configuration)

    3.2. [Option 2: User Defined Training Function](#option-2-user-defined-training-function)

    3.3. [Horovodrun Execution](#horovodrun-execution)

    3.4. [Security](#security)
4. [Examples](#examples)

    4.1. [Pytorch Examples](#pytorch-examples)

    4.2. [Tensorflow Examples](#tensorflow-examples) (Deprecated)
## Introduction

Neural Compressor uses [horovod](https://github.com/horovod/horovod) for distributed training.

Please check horovod installation documentation and use following commands to install horovod:
```
pip install horovod
```

## Supported Feature Matrix
Distributed training and inference are supported in PyTorch and TensorFlow currently.

| Framework  | Type    | Distributed Support |
|------------|---------|:-------------------:|
| PyTorch    | QAT     |       &#10004;      |
| PyTorch    | PTQ     |       &#10004;      |
| TensorFlow (Deprecated) | PTQ     |       &#10004;      |
| Keras (Deprecated)      | Pruning |       &#10004;      |

## Get Started with Distributed Training and Inference API
To enable distributed training or inference, the steps are:

1. Setting up distributed training or inference scripts. We have 2 options here:
    - Option 1: Enable distributed training or inference with pure yaml configuration. In this case, Neural Compressor builtin training function is used.
    - Option 2: Pass the user defined training function to Neural Compressor. In this case, please follow the horovod documentation and below example to know how to write such training function with horovod on different frameworks.
2. use horovodrun to execute your program.

### Option 1: Pure Yaml Configuration

To enable distributed training in Neural Compressor, user only need to add a field: `Distributed: True` in dataloader configuration:

```
dataloader:
  batch_size: 256
  distributed: True
  dataset:
    ImageFolder:
      root: /path/to/dataset
```

In user's code, pass the yaml file to Neural Compressor components, in which it constructs the real dataloader for the distributed training or inference. The example codes are as following. (TensorFlow 1.x additionally needs to enable Eager execution):

Do quantization based on distributed training/inference
``` 
# import tensorflow as tf                      (Only TensorFlow 1.x needs)
# tf.compat.v1.enable_eager_execution()        (Only TensorFlow 1.x needs)
from neural_compressor.experimental import Quantization, common
quantizer = Quantization("yaml_file_path")
quantizer.model = model
q_model = quantizer.fit()                          # q_model -> quantized low-precision model 
```

Only do model accuracy evaluation based on distributed inference
``` 
# import tensorflow as tf                      (Only TensorFlow 1.x needs)
# tf.compat.v1.enable_eager_execution()        (Only TensorFlow 1.x needs)
from neural_compressor.experimental import Quantization, common
quantizer = Quantization("yaml_file_path")
quantizer.model = model 
quantizer.pre_process()                        # If you simply want to do model evaluation with no need quantization, you should preprocess the quantizer before evaluation.
result = quantizer.strategy.evaluation_result  # result -> (accuracy, evaluation_time_cost)
```


### Option 2: User Defined Training Function

Neural Compressor supports User defined PyTorch training function for distributed training which requires user to modify training script following horovod requirements. We provide a MNIST example to show how to do that and following are the steps for PyTorch.

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
component.model = model
component.train_func = train_func
component.eval_func = test_func
model = component()
```

### Horovodrun Execution

User needs to use horovodrun to execute distributed training. For more usage, please refer to [horovod documentation](https://horovod.readthedocs.io/en/stable/running_include.html).

Following command specified the number of processes and hosts to do distributed training.
```
horovodrun -np <num_of_processes> -H <hosts> python example.py
```

For example, the following command means that two processes will be assigned to the two nodes 'node-001' and 'node-002'. The two processes will execute 'example.py' at the same time. One process is executed on node 'node-001' and one process is executed on node 'node-002'.
```
horovodrun -np 2 -H node-001:1,node-002:1 python example.py
```

### Security

Horovodrun requires user set up SSH on all hosts without any prompts. To do distributed training with Neural Compressor, user needs to ensure the SSH setting on all hosts.

## Examples
### PyTorch Examples:
- PyTorch example-1: MNIST
  - Please follow this README.md exactly：[MNIST](../../examples/pytorch/image_recognition/mnist)

- PyTorch example-2: QAT (Quantization Aware Training)
  - Please follow this README.md exactly：[QAT](../../examples/pytorch/image_recognition/torchvision_models/quantization/qat/eager/distributed)

### TensorFlow Examples: (Deprecated)
- TensorFlow example-1: 'ResNet50 V1.0' PTQ (Post Training Quantization) with distributed inference    
  - Step-1: Please cd (change directory) to the [TensorFlow Image Recognition Example](../../examples/tensorflow/image_recognition) and follow the readme to run PTQ, ensure that PTQ of 'ResNet50 V1.0' can be successfully executed.
  - Step-2: We only need to modify the [resnet50_v1.yaml](../../examples/tensorflow/image_recognition/tensorflow_models/quantization/ptq/resnet50_v1.yaml), add a line 'distributed: True' in the 'evaluation' field.
    ```
    # only need to modify the resnet50_v1.yaml, add a line 'distributed: True'
    ......
    ......
    evaluation:                                          # optional. required if user doesn't provide eval_func in neural_compressor.Quantization.
      accuracy:                                          # optional. required if user doesn't provide eval_func in neural_compressor.Quantization.
        metric:
          topk: 1                                        # built-in metrics are topk, map, f1, allow user to register new metric.
        dataloader:
          batch_size: 32
          distributed: True                              # add a line 'distributed: True'
          dataset:
            ImageRecord:
              root: /path/to/evaluation/dataset          # NOTE: modify to evaluation dataset location if needed
          transform:
            ResizeCropImagenet: 
              height: 224
              width: 224
    ......
    ......
    ```
  - Step-3: Execute 'main.py' with horovodrun as following command, it will realize PTQ based on two-node tow-process distributed inference. Please replace these fields according to your actual situation: 'your_node1_name', 'your_node2_name', '/PATH/TO/'. (Note that if you use TensorFlow 1.x now, you need to add a line 'tf.compat.v1.enable_eager_execution()' into 'main.py' to enable Eager execution.)
    ```
    horovodrun -np 2 -H your_node1_name:1,your_node2_name:1 python main.py --tune --config=resnet50_v1.yaml --input-graph=/PATH/TO/resnet50_fp32_pretrained_model.pb --output-graph=./nc_resnet50_v1.pb
    ```
- TensorFlow example-2: 'resnet_v2' pruning on Keras backend with distributed training and inference
   - Please follow this README.md exactly：[Pruning](../../examples/tensorflow/image_recognition/resnet_v2)
