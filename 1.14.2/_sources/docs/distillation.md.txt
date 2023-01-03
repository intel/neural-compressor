Distillation
============

## Introduction

Knowledge distillation is one of popular approaches of network compression, which transfers knowledge from a large model to a smaller one without loss of validity. As smaller models are less expensive to evaluate, they can be deployed on less powerful hardware (such as a mobile device). Graph shown below is the workflow of the distillation, the teacher model will take the same input that feed into the student model to produce the output that contains knowledge of the teacher model to instruct the student model.
<br>
![Distillation Workflow](./imgs/Distillation_workflow.png)

## Distillation API

### User facing API

Neural Compressor distillation API is defined under `neural_compressor.experimental.Distillation`, which takes a user defined yaml file as input. The user defined yaml defines distillation and evaluation behaviors.

```python
# distillation.py in neural_compressor/experimental
class Distillation():
    def __init__(self, conf_fname_or_obj):
        # The initialization function of distillation, taking the path or Distillation_Conf class to user-defined yaml as input
        ...

    def __call__(self):
        # The main entry of distillation, executing distillation according to user configuration.
        ...

    @model.setter
    def student_model(self, user_model):
        # The wrapper of framework model. `user_model` is the path to framework model or framework runtime model object.
        # This attribute needs to be set before invoking self.__call__().
        ...

    @model.setter
    def teacher_model(self, user_model):
        # The wrapper of framework model. `user_model` is the path to framework model or framework runtime model object.
        # This attribute needs to be set before invoking self.__call__().
        ...    

    @train_func.setter
    def train_func(self, user_train_func)
        # The training function provided by user. This function takes framework runtime model object as input parameter, 
        # and executes entire training process with self contained training hyper-parameters.
        # It is optional if training could be configured by neural_compressor built-in dataloader/optimizer/criterion.
        ...

    @eval_func.setter
    def eval_func(self, user_eval_func)
        # The evaluation function provided by user. This function takes framework runtime model object as input parameter and executes evaluation process.
        # It is optional if evaluation could be configured by neural_compressor built-in dataloader/optimizer/criterion.
        ...

    @train_dataloader.setter
    def train_dataloader(self, dataloader):
        # The dataloader used in training phase. It is optional if training dataloader is configured in user-define yaml.
        ...

    @eval_dataloader.setter
    def eval_dataloader(self, dataloader):
        # The dataloader used in evaluation phase. It is optional if training dataloader is configured in user-define yaml.
        ...

    @optimizer.setter
    def optimizer(self, dataloader):
        # The optimizer used in training phase. It is optional if optimizer is configured in user-define yaml.
        ...

    @criterion.setter
    def criterion(self, dataloader):
        # The criterion used in training phase. It is optional if criterion is configured in user-define yaml.
        ...

    def on_train_begin(self):
        # The hook point used by distillation algorithm
        ...

    def on_epoch_end(self):
        # The hook point used by distillation algorithm
        ...

    def on_after_compute_loss(self, input, student_output, student_loss, teacher_output=None):
        # The hook point used by distillation algorithm
        ...

```

### Launcher code

Simplest launcher code if training behavior is defined in user-defined yaml.

```python
from neural_compressor.experimental import Distillation, common
distiller = Distillation('/path/to/user/yaml')
distiller.student_model = student_model
distiller.teacher_model = teacher_model
model = distiller.fit()
```
Distillation class also support DistillationConf class as it's argument.

```python
from lpot.experimental import Distillation, common
from lpot.conf.config import DistillationConf
conf = DistillationConf('/path/to/user/yaml')
distiller = Distillation(conf)
distiller.student_model = student_model
distiller.teacher_model = teacher_model
model = distiller.fit()
```

### User-defined yaml

The user-defined yaml follows below syntax, note `train` section is optional if user implements `train_func` and sets to `train_func` attribute of distillation instance.

```yaml
distillation:
  train:                    # optional. No need if user implements `train_func` and pass to `train_func` attribute of pruning instance.
    start_epoch: 0
    end_epoch: 10
    iteration: 100
    
    dataloader:
      batch_size: 256
      dataset:
        ImageFolder:
          root: /path/to/imagenet/train
      transform:
        RandomResizedCrop:
          size: 224
        RandomHorizontalFlip:
        ToTensor:
        Normalize:
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225] 
    criterion:
      KnowledgeDistillationLoss:
        temperature: 1.0
        loss_types: ['CE', 'KL']
        loss_weights: [0.5, 0.5]
    optimizer:
      SGD:
        learning_rate: 0.1
        momentum: 0.9
        weight_decay: 0.0004
        nesterov: False
evaluation:                              # optional. required if user doesn't provide eval_func in neural_compressor.Quantization.
  accuracy:                              # optional. required if user doesn't provide eval_func in neural_compressor.Quantization.
    metric:
      topk: 1                            # built-in metrics are topk, map, f1, allow user to register new metric.
    dataloader:
      batch_size: 256
      dataset:
        ImageFolder:
          root: /path/to/imagenet/val
      transform:
        RandomResizedCrop:
          size: 224
        RandomHorizontalFlip:
        ToTensor:
        Normalize:
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225] 
```

#### `train`

The `train` section defines the training behavior, including what training hyper-parameter would be used and which dataloader is used during training. For criterion, we provided a built-in knowledge distillation loss class for distillation loss calculation. It is defined under `neural_compressor.experimental.common.criterion` with following structure.

```python
# criterion.py in neural_compressor/experimental/common
class KnowledgeDistillationLoss():
    def __init__(self, temperature=1.0, 
                 loss_types=['CE', 'CE'], 
                 loss_weights=[0.5, 0.5]):
        # The initialization function, taking the distillation hyper-parameters as input
        ...

    def __call__(self, student_outputs, targets):
        # The main entry of distillation loss, calculating distillation loss according to the model outputs and labels.
        ...

    def teacher_model_forward(self, input, teacher_model=None):
        # The teacher model inference function, providing for distillation loss calculation with corresponding teacher model outputs. Must be called before loss calculation if attribute teacher_outputs is not provided accordingly.
        ...
    
    @teacher_model.setter
    def teacher_model(self, model):
        # The teacher model attribute setter.
        ...
```

### Distillation with user-defined train_func()

User can pass the customized training/evaluation functions to `Distillation` for flexible scenarios. In this case, distillation process can be done by pre-defined hooks in Neural Compressor. User needs to put those hooks inside the training function.

Neural Compressor defines several hooks for user pass

```
on_train_begin() : Hook executed before training begins
on_after_compute_loss(input, student_output, student_loss) : Hook executed after each batch inference of student model
on_epoch_end() : Hook executed at each epoch end
```

Following section shows how to use hooks in user pass-in training function which is part of example from BlendCNN distillation:

```python
def train_func(model):
    distiller.on_train_begin()
    for nepoch in range(epochs):
        model.train()
        cnt = 0
        loss_sum = 0.
        iter_bar = tqdm(train_dataloader, desc='Iter (loss=X.XXX)')
        for batch in iter_bar:
            teacher_logits, input_ids, segment_ids, input_mask, target = batch
            cnt += 1
            output = model(input_ids, segment_ids, input_mask)
            loss = criterion(output, target)
            loss = distiller.on_after_compute_loss(
                {'input_ids':input_ids, 'segment_ids':segment_ids, 'input_mask':input_mask},
                output,
                loss,
                teacher_logits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if cnt >= iters:
                break
        print('Average Loss: {}'.format(loss_sum / cnt))
        distiller.on_epoch_end()
...
```

In this case, the launcher code is like the following:

```python
from neural_compressor.experimental import Distillation, common
from neural_compressor.experimental.common.criterion import PyTorchKnowledgeDistillationLoss
distiller = Distillation(args.config)
distiller.student_model = model
distiller.teacher_model = teacher
distiller.criterion = PyTorchKnowledgeDistillationLoss()
distiller.train_func = train_func
model = distiller.fit()
```

## Examples

### Examples in Neural Compressor
Following examples are supported in Neural Compressor:

- Image Classification Examples:
  - [MobileNetV2 example](../examples/pytorch/image_recognition/MobileNetV2-0.35/distillation/eager/README.md): distillation of WideResNet40-2 to MobileNetV2-0.35 on CIFAR-10 dataset.
  - [CNN example](../examples/pytorch/image_recognition/CNN-2/distillation/eager/README.md): distillation of CNN-10 to CNN-2 on CIFAR-100 dataset.
  - [VGG example](../examples/pytorch/image_recognition/VGG-8/distillation/eager/README.md): distillation of VGG-13-BN to VGG-8-BN on CIFAR-100 dataset.
  - [ResNet example](../examples/pytorch/image_recognition/torchvision_models/distillation/eager/README.md): distillation of ResNet50 to ResNet18 on ImageNet dataset.
- Natural Language Processing Examples:
  - [BlendCnn example](../examples/pytorch/nlp/blendcnn/distillation/eager/README.md): distillation of BERT-Base to BlendCnn on MRPC of GLUE dataset.
  - [BiLSTM example](../examples/pytorch/nlp/huggingface_models/text-classification/distillation/eager/README.md): distillation of RoBERTa-Base to BiLSTM on SST-2 of GLUE dataset.
  - [DistilBERT example](../examples/pytorch/nlp/huggingface_models/question-answering/distillation/eager/README.md): distillation of BERT-Base to DistilBERT on SQuAD dataset.
  - [TinyBERT example](../examples/pytorch/nlp/huggingface_models/text-classification/distillation/eager/README.md): distillation of BERT-Base to TinyBERT on MNLI of GLUE dataset.
  - [BERT-3 example](../examples/pytorch/nlp/huggingface_models/text-classification/distillation/eager/README.md): distillation of BERT-Base to BERT-3 on QQP of GLUE dataset.
  - [DistilRoBERTa example](../examples/pytorch/nlp/huggingface_models/text-classification/distillation/eager/README.md): distillation of RoBERTa-Large to DistilRoBERTa on COLA of GLUE dataset.


### Results of distillation examples
Below are results of examples shown above:

|  Example Name       | Dataset   | Student<br>(Metrics)                 | Teacher<br>(Metrics)               | Student With Distillation<br>(Metrics Improvement)  |
|---------------------|-----------|--------------------------------------|------------------------------------|-----------------------------------------------------|
| MobileNet example   | CIFAR-10  | MobileNetV2-0.35<br>(0.7965 Acc)     | WideResNet40-2<br>(0.9522 Acc)     |   0.8178 Acc<br>(0.0213 Acc)                        |
| CNN example         | CIFAR-100 | CNN-2<br>(0.5494 Acc)                | CNN-10<br>(0.7153 Acc)             |   0.5540 Acc<br>(0.0046 Acc)                        |
| VGG example         | CIFAR-100 | VGG-8-BN<br>(0.7022 Acc)             | VGG-13-BN<br>(0.7415 Acc)          |   0.7025 Acc<br>(0.0003 Acc)                        |
| ResNet example      | ImageNet  | ResNet18<br>(0.6739 Acc)             | ResNet50<br>(0.7399 Acc)           |   0.6845 Acc<br>(0.0106 Acc)                        |
| BlendCnn example    |   MRPC    | BlendCnn<br>(0.7034 Acc)             | BERT-Base<br>(0.8382 Acc)          |   0.7034 Acc<br>(0 Acc)                             |
| BiLSTM example      |  SST-2    | BiLSTM<br>(0.8314 Acc)               | RoBERTa-Base<br>(0.9403 Acc)       |   0.9048 Acc<br>(0.0734 Acc)                        |
|DistilBERT example   |  SQuAD    | DistilBERT<br>(0.7323/0.8256 EM/F1)  | BERT-Base<br>(0.8084/0.8814 EM/F1) |   0.7442/0.8371 EM/F1<br>(0.0119/0.0115 EM/F1)      |
|TinyBERT example     |  MNLI     | TinyBERT<br>(0.8018/0.8044 m/mm)     | BERT-Base<br>(0.8363/0.8411 m/mm)  |   0.8025/0.8074 m/mm<br>(0.0007/0.0030 m/mm)        |
|BERT-3 example       |  QQP      | BERT-3<br>(0.8626/0.8213 EM/F1)      | BERT-Base<br>(0.9091/0.8782 EM/F1) |   0.8684/0.8259 EM/F1<br>(0.0058/0.0046 EM/F1)      |
|DistilRoBERTa example|  COLA     | DistilRoBERTa<br>(0.6057 ACC)        | RoBERTa-Large<br>(0.6455 ACC)      |   0.6187 ACC<br>(0.0130 ACC)                        |
