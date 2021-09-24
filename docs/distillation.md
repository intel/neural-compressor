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

    def pre_epoch_begin(self):
        # The hook point used by distillation algorithm
        ...

    def on_epoch_end(self):
        # The hook point used by distillation algorithm
        ...

    def on_post_forward(self, batch, teacher_output=None):
        # The hook point used by distillation algorithm
        ...

```

### Launcher code

Simplest launcher code if training behavior is defined in user-defined yaml.

```python
from neural_compressor.experimental import Distillation, common
distiller = Distillation('/path/to/user/yaml')
distiller.student_model = common.Model(student_model)
distiller.teacher_model = common.Model(teacher_model)
model = distiller()
```
Distillation class also support Distillation_Conf class as it's argument.

```python
from lpot.experimental import Distillation, common
from lpot.conf.config import Distillation_Conf
conf = Distillation_Conf('/path/to/user/yaml')
distiller = Distillation(conf)
distiller.student_model = common.Model(student_model)
distiller.teacher_model = common.Model(teacher_model)
model = distiller()
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
pre_epoch_begin() : Hook executed before training begins
on_post_forward(batch) : Hook executed after each batch inference of student model
on_epoch_end() : Hook executed at each epoch end
```

Following section shows how to use hooks in user pass-in training function which is part of example from BlendCNN distillation:

```python
def train_func(model):
    distiller.pre_epoch_begin()
    for nepoch in range(epochs):
        model.train()
        cnt = 0
        loss_sum = 0.
        iter_bar = tqdm(train_dataloader, desc='Iter (loss=X.XXX)')
        for batch in iter_bar:
            teacher_logits, input_ids, segment_ids, input_mask, target = batch
            cnt += 1
            output = model(input_ids, segment_ids, input_mask)
            distiller.on_post_forward({'input_ids':input_ids, 
                                       'segment_ids':segment_ids, 
                                       'input_mask':input_mask}, \
                                      teacher_logits)
            loss = distiller.criterion(output, target)
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
distiller.student_model = common.Model(model)
distiller.teacher_model = common.Model(teacher)
distiller.criterion = PyTorchKnowledgeDistillationLoss()
distiller.train_func = train_func
model = distiller()
```

## Examples

### Examples in Neural Compressor
Following examples are supported in Neural Compressor:

- CNN Examples:
  - [ResNet example](../examples/pytorch/eager/image_recognition/imagenet/cpu/distillation/README.md): distillation of ResNet50 to ResNet18 on ImageNet dataset.
- NLP Examples:
  - [BlendCnn example](../examples/pytorch/eager/blendcnn/distillation/README.md): distillation of BERT-Base to BlendCnn on MRPC of GLUE dataset.
  - [BiLSTM example](../examples/pytorch/eager/huggingface_models/README.md): distillation of RoBERTa-Base to BiLSTM on SST-2 of GLUE dataset.

### Results of distillation examples
Below are results of examples shown above:

|  Example Name    | Dataset  | Student<br>(Accuracy) | Teacher<br>(Accuracy)    | Student With Distillation<br>(Accuracy Improvement) |
|------------------|----------|-----------------------|--------------------------|-----------------------------------------------------|
| ResNet example   | ImageNet | ResNet18<br>(0.6739)  | ResNet50<br>(0.7399)     |   0.6845<br>(0.0106)                                |
| BlendCnn example |   MRPC   | BlendCnn<br>(0.7034)  | BERT-Base<br>(0.8382)    |   0.7034<br>(0)                                     |
| BiLSTM example   |  SST-2   | BiLSTM<br>(0.7913)    | RoBERTa-Base<br>(0.9404) |   0.8085<br>(0.0172)                                |
