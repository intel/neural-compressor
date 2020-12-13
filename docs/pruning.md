# Introduction
Sparsity is a a measure of how many percents of elements in a tensor are exact zeros[^1].  A tensor is considered sparse if "most" of its elements are zero. Appeartly only non-zero elements will be stored and computed so the inference process could be accelerated due to Tops and memory saved[^2].
[^1]: https://nervanasystems.github.io/distiller/pruning.html 
[^2]: acceleration needs sparse compute kernels which are WIP

The <a href="https://en.wikipedia.org/wiki/Lp_space#When_p_=_0">\(l_0\)-"norm" function</a> measures how many zero-elements are in a tensor <em>x</em>:
\[\lVert x \rVert_0\;=\;|x_1|^0 + |x_2|^0 + ... + |x_n|^0 \]
In other words, an element contributes either a value of 1 or 0 to \(l_0\).  Anything but an exact zero contributes a value of 1 - that's pretty cool. Sometimes it helps to think about density, the number of non-zero elements (NNZ) and sparsity's complement:
\[
density = 1 - sparsity
\]
A common method for introducing sparsity in weights and activations is called <em>pruning</em>.  Pruning is the application of a binary criteria to decide which weights to prune: weights which match the pruning criteria are assigned a value of zero.  Pruned elements are "trimmed" from the model: we replace their values with zero and also make sure they don't take part in the back-propagation process.</p>


# Design
Pruning process is sometimes similiar with quantization-aware training(QAT). Intel® Low Precision Optimization Tool will do related model transformation during training and retrain the model to meet the accuracy goal.
 We implemented 2 kinds of object: Pruner and PrunePolicy. Firstly we define a sparsity goal(model-wise or op-wise, depending on whether there are ops not suitable for pruning) and the way to reach sparsity goal(Usually we increase the sparsity target linearly as the epoches). The pruner is in singeleton mode, and will update sparsity goal and schedule all PrunePolicy on different phase of training.
 PrunePolicy carries different pruning algos. For example, MagnitudePrunePolicy set thresholds of absolute value so that elements whose absolute value lower than the threshold will be zeroed. The zeroing process happens on begining and end of each minbatch iteration. 

# Usage
Pruning configs need to be added into yaml as pruning field. Global parameters contain **start_epoch** (on which epoch pruning begins), **end_epoch** (on which epoch pruning ends), **init_sparsity** (initial sparsity goal default 0), **target_sparsity** (target sparsity goal) and **frequency** (of updating sparsity). At least one pruner instance need to be defined, under specific algos (currently only magnitude supported). you can override all global params in specific pruner using field names and specify which weight of model to be pruned. if no weight specified, all weights of model will be pruned.
```yaml
pruning:
  magnitude:
      prune1:
        # weights: ['layer1.0.conv1.weight',  'layer1.0.conv2.weight']
        # target_sparsity: 0.3
        # start_epoch: 1
  start_epoch: 0
  end_epoch: 4
  frequency: 1
  init_sparsity: 0.05
  target_sparsity: 0.25
```

# Examples
Users must pass a modified training function to Intel® Low Precision Optimization Tool. Take a typical pytorch training function as example.
```python
def p_func(model):
    # from lpot.pruning import Pruner
    # prune = Pruner(*args, **kwargs)
    for epoch in range(epochs)
        # pruner.on_epoch_begin(epoch=epoch)
        for x, label in dataloader:
            # pruner.on_batch_begin()  
            y = model(x)
            loss = criterion(y, label)
            loss = pruner.on_minibatch_end(model, loss)
            optimizer.zero_grad()            
            loss.backward()           
            optimizer.step()
            # pruner.on_batch_end()
        # pruner.on_epoch_end(epoch=epoch)
        evaluate(model)
```
Note the commented lines are how pruner do model transformation.
