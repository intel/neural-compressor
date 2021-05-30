Pruning
=======

## Introduction

Sparsity is a measure of how many percents of elements in a tensor are [exact zeros][^1]. A tensor is considered sparse if most of its elements are zero. Only non-zero elements will be stored and computed so the inference process could be accelerated due to TOPS (teraoperations/second) and memory saved (acceleration needs sparse compute kernels which are a work in process).

The <a href="https://en.wikipedia.org/wiki/Lp_space#When_p_=_0"><img src="http://latex.codecogs.com/svg.latex?\l_{1}&space;" title="http://latex.codecogs.com/svg.latex?\l_{1} " />-"norm" function</a> measures how many zero-elements are in a tensor <em>x</em>:
<img src="http://latex.codecogs.com/svg.latex?\left|\left|&space;x\right|&space;\right|_{0}\doteq&space;\left|x_{1}&space;\right|^{0}&plus;&space;\left|x_{2}&space;\right|^{0}&plus;...&plus;\left|x_{n}&space;\right|^{0}&space;" title="http://latex.codecogs.com/svg.latex?\left|\left| x\right| \right|_{0}\doteq \left|x_{1} \right|^{0}+ \left|x_{2} \right|^{0}+...+\left|x_{n} \right|^{0} " />
In other words, an element contributes either a value of 1 or 0 to \(l_0\).  Anything but an exact zero contributes a value of 1 - which is good. Sometimes it helps to think about density, the number of non-zero elements (NNZ) and sparsity's complement:
\[
density = 1 - sparsity
\]
A common method for introducing sparsity in weights and activations is called **pruning**. Pruning is the application of a binary criteria to decide which weights to prune: weights which match the pruning criteria are assigned a value of zero. Pruned elements are "trimmed" from the model: we replace their values with zero and also make sure they don't take part in the back-propagation process.</p>


## Design

The pruning process is similar to quantization-aware training (QAT). Intel® Low Precision Optimization Tool will do related model transformation during training and retrain the model to meet the accuracy goal.

We implemented two kinds of object: Pruner and PrunePolicy. First, we define a sparsity goal (model-wise or op-wise, depending on whether there are ops not suitable for pruning) and the way to reach the sparsity goal (usually we increase the sparsity target linearly as the epoches). The pruner is in singleton mode, and will update the sparsity goal and schedule all PrunePolicy during different phases of training.

PrunePolicy carries different pruning algos. For example, MagnitudePrunePolicy sets thresholds of absolute value so that elements whose absolute value lower than the threshold will be zeroed. The zeroing process happens at the beginning and end of each minibatch iteration.

## Usage

Pruning configs need to be added into yaml as a ```pruning``` field. 


```yaml
pruning:                                             # mandotory only for pruning.
  train:
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
            prune_type: basic_magnitude              # currently only support basic_magnitude
            names: ['layer1.0.conv1.weight']         # tensor name to be pruned.
            start_epoch: 0
            end_epoch: 2
            update_frequency: 0.1
```

### Training
Most of pruning methods need ``training`` to keep the accuracy. There are two ways that Users can define the training process in ``lpot``. One is completely configured in the yaml and ``lpot`` will create a training function automatically as the above example yaml. 

Or users can pass in ``def train`` by themselves and insert ``pruner`` manually like the previous version. This is more suitable for complex and customize training function like NLP tasks especially text-generation models. 

We provide examples of both 2 usages. For completely Yaml config, please refer to [resnet example](examples/pytorch/eager/image_recognition/imagenet/cpu/prune/conf.yaml). For users' training function, please refer to [BERT example](examples/pytorch/eager/language_translation/prune/conf.yaml).

### Pruning config
We dived the pruning into 2 kinds: ``weight compression`` and ``activation compression``, the laster is WIP. ``weight compression`` means zeroing the weight matrixs.

For ``weight_compression``, we dived params into global parameters and local paramers in different ``pruners``. Global prameters may contain **start_epoch** (on which epoch pruning begins), **end_epoch** (on which epoch pruning ends), **initial_sparsity** (initial sparsity goal default 0), **target_sparsity** (target sparsity goal) and **frequency** (of updating sparsity). At least one pruner instance needs to be defined under specific algos (currently only ``basic_magnitude`` supported). You can override all global params in a specific pruner using field names and specify names of which weight of model to be pruned. If no weight is specified, all weights of the model will be pruned.

## Example of user pass-in training function

Users pass a modified training function to Intel® Low Precision Optimization Tool. The following is part of example from BERT training:

```python
...
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
Then users can use LPOT like the following:

```python
from lpot.experimental import Pruning, common
prune = Pruning(args.config)
prune.model = common.Model(model)
prune.train_dataloader = train_dataloader
prune.pruning_func = train_func
prune.eval_dataloader = train_dataloader
prune.eval_func = eval_func
model = prune()
```

[^1]: https://nervanasystems.github.io/distiller/pruning.html 