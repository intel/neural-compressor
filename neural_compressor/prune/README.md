Pruning
============



1. [Introduction](#introduction)



        1.1. [Neural Network Pruning](#neural-network-pruning)



        1.2. [Pruning Patterns](#pruning-patterns)



        1.3. [Pruning Criteria](#pruning-criteria)



        1.4. [Pruning Schedule](#pruning-schedule)



        1.5. [Pruning type](#pruning-type)



        1.6. [Regularization](#regularization)



2. [Get Started With Pruning API](#get-started-with-pruning-api)



3. [Examples](#examples)



## Introduction



### Neural Network Pruning
Neural network pruning is a promising model compression technique that removes the least important parameters/neurons in the network and achieves compact architectures with minimal accuracy drop and maximal inference acceleration. As state-of-the-art model sizes have grown at an unprecedented speed, pruning has become increasingly crucial for reducing the computational and memory footprint that huge neural networks require.

<div align=center>
<a target="_blank" href="./../../docs/source/_static/imgs/pruning/pruning.PNG">
    <img src="./../../docs/source/_static/imgs/pruning/pruning.PNG" width=350 height=170 alt="pruning intro">
</a>
</div>


### Pruning Patterns



Pruning patterns defines the rules of pruned weights' arrangements in space. INC currently supports unstructured, N:M and NxM patterns. Please note that N:M pattern is applied to input channels while NxM pattern is applied to output ones. [Details](../../docs/source/pruning_details.md#pruning-patterns).

<div align=center>
<a target="_blank" href="./../../docs/source/_static/imgs/pruning/Pruning_patterns.PNG">
    <img src="./../../docs/source/_static/imgs/pruning/Pruning_patterns.PNG" width=700 height=160 alt="Sparsity Pattern">
</a>
</div>

### Pruning Criteria



Pruning Criteria determines how should the weights of a neural network be scored and pruned. In the image below, pruning scores are represented by neurons' color and those with the lowest scores are pruned. The magnitude and gradient are widely used to score the weights. Currently, INC supports **magnitude**, **snip** and **snip_momentum** criteria. [Details](../../docs/source/pruning_details.md#pruning-criteria).

<div align=center>
<a target="_blank" href="./../../docs/source/_static/imgs/pruning/pruning_criteria.PNG">
    <img src="./../../docs/source/_static/imgs/pruning/pruning_criteria.PNG" width=385 height=187 alt="Pruning criteria">
</a>
</div>

### Pruning Schedule



Pruning schedule defines the way the model reach the target sparsity (the ratio of pruned weights). Both **one-shot** and **iterative** pruning schedules are supported. [Details](../../docs/source/pruning_details.md#pruning-schedule).

<div align=center>
<a target="_blank" href="./../../docs/source/_static/imgs/pruning/Pruning_schedule.PNG">
    <img src="./../../docs/source/_static/imgs/pruning//Pruning_schedule.PNG" width=950 height=210 alt="Pruning schedule">
</a>  
</div>


### Pruning Type



Pruning type defines how the masks are generated and applied to a neural network. Both **pattern_lock** and **progressive** types are supported by INC. [Details](../../docs/source/pruning_details.md#pruning-type).



### Regularization



Regularization is a technique that discourages learning a more complex model and therefore performs variable-selection. In the image below, some weights are pushed to be as small as possible and the connections are thus sparsified. **Group-lasso** method is used in INC. 
[Details](../../docs/source/pruning_details.md#regularization).

<div align=center>
<a target="_blank" href="./../../docs/source/_static/imgs/pruning/Regularization.PNG">
    <img src="./../../docs/source/_static/imgs/pruning/Regularization.PNG" width=350 height=170 alt="Regularization">
</a>
</div>


## Get Started with Pruning API



Neural Compressor `Pruning` API is defined under `neural_compressor.pruning`, which takes a user defined yaml file as input. 
Users can pass the customized training/evaluation functions to `Pruning` in various scenarios. 

In this case, pruning process can be done by pre-defined hooks in Neural Compressor. Users need to put those hooks inside the training function. The pre-defined Neural Compressor hooks are listed below.



```
on_train_begin() : Implement at the beginning of training phase.
on_epoch_begin(epoch) : Implement at the beginning of each epoch.
on_step_begin(batch) : Implement at the beginning of each batch.
on_step_end() : Implement at the end of each batch.
on_epoch_end() : Implement at the end of each epoch.
on_before_optimizer_step() : Implement before optimization step.
on_after_optimizer_step() : Implement after optimization step.
```



The following section is an example of how to use hooks in user pass-in training function to perform BERT training:



```python
from neural_compressor.pruning import Pruning

prune = Pruning(config_file)
prune.update_config()
prune.model = model
prune.on_train_begin()
for epoch in range(num_train_epochs):
    model.train()
    prune.on_epoch_begin(epoch)
    for step, batch in enumerate(train_dataloader):
        prune.on_step_begin(step)
        outputs = model(**batch)
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()
        if (step + 1) % gradient_accumulation_steps == 0:
            prune.on_before_optimizer_step()
            optimizer.step()
            prune.on_after_optimizer_step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
        prune.on_step_end()
...
```



## Examples



We validate the pruning technique on typical models across various domains (including CV, NLP, and Recommendation System) and the examples are listed in [Pruning Examples](../../docs/source/pruning_details.md#examples). A complete overview of validated examples including quantization, pruning and distillation results could be found in  [INC Validated examples](../../docs/source/validated_model_list.md#validated-pruning-examples).


Please refer to pruning examples([TensorFlow](../../examples/README.md#Pruning), [PyTorch](../../examples/README.md#Pruning-1)) for more information.