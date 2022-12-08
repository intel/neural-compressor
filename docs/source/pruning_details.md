Pruning details
============



1. [Introduction](#introduction)



    1.1. [Neural Network Pruning](#neural-network-pruning)



    1.2. [Pruning Patterns](#pruning-patterns)



    1.3. [Pruning Criteria](#pruning-criteria)



    1.4. [Pruning Schedule](#pruning-schedule)



    1.5. [Regularization](#regularization)





2. [Examples and results](#examples)



## Introduction



### Neural Network Pruning
Neural network pruning is a promising model compression technique that removes the least important parameters in the network and achieves compact architectures with minimal accuracy drop and maximal inference acceleration. As state-of-the-art model sizes have grown at an unprecedented speed, pruning has become increasingly crucial for reducing the computational and memory footprint that huge neural networks require.



### Pruning Patterns


#### Unstructured Pruning


Unstructured pruning means pruning the least salient connections in the model. The nonzero patterns are irregular and could be anywhere in the matrix.


#### Structured Pruning


Structured pruning means pruning parameters in groups and deleting entire blocks, filters, or channels according to some pruning criterions. In general, structured pruning leads to lower accuracy due to restrictive structure compared to unstructured pruning but it can significantly accelerate the model execution as it fits better with hardware designs.


#### Progressive Pruning


Progressive pruning aims at smoothing the structured pruning by automatically interpolating a group of interval masks during the pruning process. In this method, a sequence of masks are generated to enable a more flexible pruning process and those masks would gradually change into ones to fit the target pruning structure. 



### Pruning Criteria



Pruning criteria determines how should the weights of a neural network be scored and pruned. The magnitude and gradient are widely used to score the weights.


- Magnitude


  The algorithm prunes the weight by the lowest absolute value at each layer with given sparsity target.


- SNIP


  The algorithm prunes the dense model at its initialization, by analyzing the weights' effect to the loss function when they are masked. Please refer to the original [paper](https://arxiv.org/abs/1810.02340) for details


- SNIP with momentum


  The algorithm improves original SNIP algorithms and introduces weights' score maps which updates in a momentum way.\
  In the following formula, $n$ is the pruning step and $W$ and $G$ are model's weights and gradients respectively.
  $$Score_{n} = 1.0 \times Score_{n-1} + 0.9 \times |W_{n} \times G_{n}|$$


### Pruning Schedule


Pruning schedule defines the way the model reach the target sparsity (the ratio of pruned weights).


- One-shot Pruning


  One-shot pruning means the model is pruned to its target sparsity with one single step. This pruning method often works at model's initialization step. It can easily cause accuracy drop, but save much training time.



- Iterative Pruning


  Iterative pruning means the model is gradually pruned to its target sparsity during a training process. The pruning process contains several pruning steps, and each step raises model's sparsity to a higher value. In the final pruning step, the model reaches target sparsity and the pruning process ends.



### Regularization


Regularization is a technique that discourages learning a more complex model and therefore performs variable-selection.


- Group Lasso


  The Group-lasso algorithm is used to prune entire rows, columns or blocks of parameters that result in a smaller dense network.



## Examples



We validate the sparsity on typical models across different domains (including CV, NLP, and Recommendation System) and the examples are listed below. A complete overview of validated examples including quantization, pruning and distillation result could be found in  [INC Validated examples](../../docs/source/validated_model_list.md#validated-pruning-examples).



Please refer to pruning examples([TensorFlow](../../examples/README.md#Pruning), [PyTorch](../../examples/README.md#Pruning-1)) for more information.
 
