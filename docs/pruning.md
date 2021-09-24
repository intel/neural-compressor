Pruning
============

## Introduction

Network pruning is one of popular approaches of network compression, which reduces the size of a network by removing parameters with minimal drop in accuracy.

- Structured Pruning

Structured pruning means pruning sparsity patterns, in which there is some structure, most often in the form of blocks.

- Unstructured Pruning

Unstructured pruning means pruning unstructured sparsity (aka random sparsity) patterns, where the nonzero patterns are irregular and could be anywhere in the matrix.

- Filter/Channel Pruning

Filter/Channel pruning means pruning a larger part of the network, such as filters or layers, according to some rules.

## Pruning Algorithms supported by Neural Compressor

|    Pruning Type        |                 Algorithm                   | PyTorch | Tensorflow |
|------------------------|---------------------------------------------|---------|------------|
| unstructured pruning   | basic_magnitude                             |   Yes   |     Yes    |
|                        | pattern_lock                                |   Yes   |     N/A    | 
|  structured pruning    | pattern_lock                                |   Yes   |     N/A    | 
| filter/channel pruning | gradient_sensitivity                        |   Yes   |     N/A    |

Neural Compressor also supports the two-shot execution of unstructured pruning and post-training quantization.

- basic_magnitude:

  - The algorithm prunes the weight by the lowest absolute value at each layer with given sparsity target.

- gradient_sensitivity:

  - The algorithm prunes the head, intermediate layers, and hidden states in NLP model according to importance score calculated by following the paper [FastFormers](https://arxiv.org/abs/2010.13382). 

- pattern_lock

  - The algorithm takes a sparsity model as input and starts to fine tune this sparsity model and locks the sparsity pattern by freezing those zero values in weight tensor after weight update during training. 

- pruning and then post-training quantization

  - The algorithm executes unstructured pruning and then executes post-training quantization. 

- pruning during quantization-aware training

  - The algorithm executes unstructured pruning during quantization-aware training.

## Pruning API

### User facing API

Neural Compressor pruning API is defined under `neural_compressor.experimental.Pruning`, which takes a user defined yaml file as input. The user defined yaml defines training, pruning and evaluation behaviors.

```
# pruning.py in neural_compressor/experimental
class Pruning():
    def __init__(self, conf_fname_or_obj):
        # The initialization function of pruning, taking the path or Pruning_Conf class to user-defined yaml as input
        ...

    def __call__(self):
        # The main entry of pruning, executing pruning according to user configuration.
        ...

    @model.setter
    def model(self, user_model):
        # The wrapper of framework model. `user_model` is the path to framework model or framework runtime model 
        object.
        # This attribute needs to be set before invoking self.__call__().
        ...

    @pruning_func.setter
    def pruning_func(self, user_pruning_func)
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

    def on_epoch_begin(self, epoch):
        # The hook point used by pruning algorithm
        ...

    def on_epoch_end(self):
        # The hook point used by pruning algorithm
        ...

    def on_batch_begin(self, batch):
        # The hook point used by pruning algorithm
        ...

    def on_batch_end(self):
        # The hook point used by pruning algorithm
        ...

    def on_post_grad(self):
        # The hook point used by pruning algorithm
        ...

```

### Launcher code

Simplest launcher code if training behavior is defined in user-defined yaml.

```
from neural_compressor.experimental import Pruning, common
prune = Pruning('/path/to/user/pruning/yaml')
prune.model = common.Model(model)
model = prune()
```

Pruning class also support Pruning_Conf class as it's argument.

```
from lpot.experimental import Pruning, common
from lpot.conf.config import Pruning_Conf
conf = Pruning_Conf('/path/to/user/pruning/yaml')
prune = Pruning(conf)
prune.model = common.Model(model)
model = prune()
```

### User-defined yaml

The user-defined yaml follows below syntax, note `train` section is optional if user implements `pruning_func` and sets to `pruning_func` attribute of pruning instance.

```
pruning:
  train:                    # optional. No need if user implements `pruning_func` and pass to `pruning_func` attribute of pruning instance.
    start_epoch: 0
    end_epoch: 10
    iteration: 100
    frequency: 2
    
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
      CrossEntropyLoss:
        reduction: None
    optimizer:
      SGD:
        learning_rate: 0.1
        momentum: 0.9
        weight_decay: 0.0004
        nesterov: False

  approach:
    weight_compression:
      initial_sparsity: 0.0
      target_sparsity: 0.3
      pruners:
        - !Pruner
            initial_sparsity: 0.0
            target_sparsity: 0.97
            start_epoch: 0
            end_epoch: 2
            prune_type: basic_magnitude
            update_frequency: 0.1
            names: ['layer1.0.conv1.weight']
        - !Pruner
            start_epoch: 0
            end_epoch: 1
            prune_type: gradient_sensitivity
            update_frequency: 1
            names: [
                     'bert.encoder.layer.0.attention.output.dense.weight',
                   ]
            parameters: {
                          target: 8,
                          transpose: True,
                          stride: 64,
                          index: 0,
                          normalize: True,
                          importance_inputs: ['head_mask'],
                          importance_metric: abs_gradient
                        }

```

#### `train`

The `train` section defines the training behavior, including what training hyper-parameter would be used and which dataloader is used during training. 

#### `approach`

The `approach` section defines which pruning algorithm is used and how to apply it during training process.

- ``weight compression``: pruning target, currently only ``weight compression`` is supported. ``weight compression`` means zeroing the weight matrix. The parameters for `weight compression` is divided into global parameters and local parameters in different ``pruners``. Global parameters may contain `start_epoch`, `end_epoch`, `initial_sparsity`, `target_sparsity` and `frequency`. 

  - `start_epoch`:  on which epoch pruning begins
  - `end_epoch`: on which epoch pruning ends
  - `initial_sparsity`: initial sparsity goal, default 0.
  - `target_sparsity`: target sparsity goal
  - `frequency`: frequency to updating sparsity

- `Pruner`:

  - `prune_type`: pruning algorithm, currently ``basic_magnitude`` and ``gradient_sensitivity`` are supported.

  - `names`: weight name to be pruned. If no weight is specified, all weights of the model will be pruned.

  - `parameters`: Additional parameters is required ``gradient_sensitivity`` prune_type, which is defined in ``parameters`` field. Those parameters determined how a weight is pruned, including the pruning target and the calculation of weight's importance. it contains:

    - `target`: the pruning target for weight.
    - `stride`: each stride of the pruned weight.
    - `transpose`: whether to transpose weight before prune.
    - `normalize`: whether to normalize the calculated importance.
    - `index`: the index of calculated importance.
    - `importance_inputs`: inputs of the importance calculation for weight.
    - `importance_metric`: the metric used in importance calculation, currently ``abs_gradient`` and ``weighted_gradient`` are supported.

    Take above as an example, if we assume the 'bert.encoder.layer.0.attention.output.dense.weight' is the shape of [N, 12\*64]. The target 8 and stride 64 is used to control the pruned weight shape to be [N, 8\*64]. `Transpose` set to True indicates the weight is pruned at dim 1 and should be transposed to [12\*64, N] before pruning. `importance_input` and `importance_metric` specify the actual input and metric to calculate importance matrix.


### Pruning with user-defined pruning_func()

User can pass the customized training/evaluation functions to `Pruning` for flexible scenarios. `Pruning`  In this case, pruning process can be done by pre-defined hooks in Neural Compressor. User needs to put those hooks inside the training function.

Neural Compressor defines several hooks for user pass

```
on_epoch_begin(epoch) : Hook executed at each epoch beginning
on_batch_begin(batch) : Hook executed at each batch beginning
on_batch_end() : Hook executed at each batch end
on_epoch_end() : Hook executed at each epoch end
```

Following section shows how to use hooks in user pass-in training function which is part of example from BERT training:

```python
def pruning_func(model):
    for epoch in range(int(args.num_train_epochs)):
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        model.train()
        prune.on_epoch_begin(epoch)
        for step, batch in enumerate(train_dataloader):
            prune.on_batch_begin(step)
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            #inputs['token_type_ids'] = batch[2]
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
    
            prune.on_batch_end()
...
```

In this case, the launcher code is like the following:

```python
from neural_compressor.experimental import Pruning, common
prune = Pruning(args.config)
prune.model = common.Model(model)
prune.pruning_func = pruning_func
model = prune()
```

### Scheduler for Pruning and Quantization

Neural Compressor defined Scheduler to automatically pipeline execute prune and post-training quantization. After appending separate component into scheduler pipeline, scheduler executes them one by one. In following example it executes the pruning and then post-training quantization.

```python
from neural_compressor.experimental import Quantization, common, Pruning, Scheduler
prune = Pruning(prune_conf)
quantizer = Quantization(post_training_quantization_conf)
scheduler = Scheduler()
scheduler.model = common.Model(model)
scheduler.append(prune)
scheduler.append(quantizer)
opt_model = scheduler()
```

## Examples

### Examples in Neural Compressor
Following examples are supported in Neural Compressor:

- CNN Examples:
  - [resnet example](../examples/pytorch/eager/image_recognition/imagenet/cpu/prune/README.md): magnitude pruning on resnet.
  - [pruning and post-training quantization](../examples/pytorch/eager/image_recognition/imagenet/cpu/prune_and_ptq/README.md): magnitude pruning and then post-training quantization on resnet.
  - [resnet_v2 example](../examples/tensorflow/pruning/resnet_v2/README.md): magnitude pruning on resnet_v2 for tensorflow.
- NLP Examples:
  - [BERT example](../examples/pytorch/eager/language_translation/prune/README.md): magnitude pruning on DistilBERT.
  - [BERT example](../examples/pytorch/eager/huggingface_models/README.md): Pattern-lock and head-pruning on BERT-base.

