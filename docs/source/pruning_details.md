Pruning details
============




 

1. [Introduction](#introduction)




 

>>>[Neural Network Pruning](#neural-network-pruning)




 

>>>[Pruning Patterns](#pruning-patterns)




 

>>>[Pruning Criteria](#pruning-criteria)




 

>>>[Pruning Schedule](#pruning-schedule)




 

>>>[Pruning Type](#pruning-type)




 

>>>[Regularization](#regularization)






 

2. [Pruning examples](#examples)




 

3. [Reference](#reference)




 

## Introduction




 

### Neural Network Pruning

Neural network pruning is a promising model compression technique that removes the least important parameters in the network and achieves compact architectures with minimal accuracy drop and maximal inference acceleration. As state-of-the-art model sizes have grown at an unprecedented speed, pruning has become increasingly crucial for reducing the computational and memory footprint that huge neural networks require.





 

### Pruning Patterns



 

- Unstructured Pruning



 

  Unstructured pruning means pruning the least salient connections in the model. The nonzero patterns are irregular and could be anywhere in the matrix.



 

- Structured Pruning



 

  Structured pruning means pruning parameters in groups and deleting entire blocks, filters, or channels according to some pruning criterions. In general, structured pruning leads to lower accuracy due to restrictive structure compared to unstructured pruning but it can significantly accelerate the model execution as it fits better with hardware designs.







 

### Pruning Criteria




 

Pruning criteria determines how should the weights of a neural network be scored and pruned. The magnitude and gradient are widely used to score the weights.



 

- Magnitude



 

  The algorithm prunes the weight by the lowest absolute value at each layer with given sparsity target.



 

- Gradient


 

  The algorithm prunes the weight by the lowest gradient value at each layer with given sparsity target.


 

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





 

### Pruning Type




 

- Pattern_lock Pruning



 

  Pattern_lock pruning type uses masks of a fixed pattern during the pruning process. It locks the sparsity pattern in finetuning phase by freezing those zero values of weight tensor during weight update of training. It can be applied in the following scenario: after the model is pruned under a large dataset, pattern lock can be used to retrain the sparse model on a downstream task (a smaller dataset). Please refer to [Prune once for all](https://arxiv.org/pdf/2111.05754.pdf) for more information.



 

- Progressive Pruning





  Progressive pruning aims at smoothing the structured pruning by automatically interpolating a group of interval masks during the pruning process. In this method, a sequence of masks are generated to enable a more flexible pruning process and those masks would gradually change into ones to fit the target pruning structure.
  Progressive pruning is used mainly for channel-wise pruning and currently only supports NxM pruning pattern.

  <div style = "width: 77%; margin-bottom: 2%;">
    <a target="_blank" href="./imgs/pruning/progressive_pruning.png">
      <img src="./imgs/pruning/progressive_pruning.png" alt="Architecture" width=800 height=500>
    </a>
  </div>
  (a) refers to the traditional structured iterative pruning; (b, c, d) demonstrates some typical implementations of mask interpolation. (b) uses masks with smaller structured blocks during every pruning step. (c) inserts masks with smaller structured blocks between every pruning steps. (d) inserts unstructured masks which prune some weights by referring to pre-defined score maps. We use (d) as the mask interpolation implementation of progressive pruning.

  


### Pruning Scope
Range of sparse score calculation in iterative pruning, default scope is global.


 

- Global




  The score map is computed out of entire parameters, Some layers are higher than the target sparsity and some are lower, the total sparsity of the model reaches the target.




- Local




  The score map is computed from the corresponding layer's weight, The sparsity of each layer is equal to the target.




### Sparsity Decay Type




Growth rules for the sparsity of iterative pruning, "exp", "linear", "cos" and "cube" are available，We use exp by default.







### Regularization



 

Regularization is a technique that discourages learning a more complex model and therefore performs variable-selection.



 

- Group Lasso



 

  The main idea of Group Lasso is to construct an objective function that penalizes the L2 parametrization of the grouped variables, determines the coefficients of some groups of variables to be zero, and obtains a refined model by feature filtering.





 

## Pruning Examples


 

We validate the pruning technique on typical models across various domains (including CV and NLP).


 

## Reference


 

[1] Namhoon Lee, Thalaiyasingam Ajanthan, and Philip Torr. SNIP: Single-shot network pruning based on connection sensitivity. In International Conference on Learning Representations, 2019.




 




 


