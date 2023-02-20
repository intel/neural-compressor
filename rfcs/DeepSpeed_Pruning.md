## **Summary**

This is a detail proposal on introducing the **structural pruning** method into DeepSpeed.

The DeepSpeed Compression have supported some pruning methods like sparse pruning, row pruning, head pruning, and channel pruning.

In this proposal, we would like to introduce structural pruning functionalities by enabling "N in M" and "N x M" block sparsity pattern with snip_momentum criteria and progressive pruning.

## **Motivation**

Current state-of-the-art deep neural networks have a great number of parameters, which lead to their heavy computing resource consumption during training and inference. To address this issue, network pruning (also known as network sparsity) has been proposed in order to make large models to be deployed on some edge devices with limited computation and memory footprint. However, when a high ratio of parameters has been pruned from the original dense model, its accuracy can be prone to drop. Therefore, pruning a model to a high sparsity while maintaining its original accuracy is a challenge. More pruning algorithms and utilizes need to be supported to meet the requirments of the users. And meanwhile, the structural pruning are getting more and more focus by the industry as the facts of structural pruning having better performance.

## **Proposed Implementation on Pruning**

We propose a two-phases support on the structural pruning method.

**Phase 1** Basic but complete structural pruning functionality by extending the json config file format and implement the structural sparsity algorithm in `compression` dir. This way leverages the existing DeepSpeed sparsity design which only has a global sparse ratio control. If the accuracy doesn't meet expectation, user has to tune the training process by manually specifying and exlporing the proper sparse ratio per layer.

The json config file format is extended like below. 

~~~jason
{
        "sparse_pruning": {
        "shared_parameters": {
          "enabled": True,
          "method": "snip_momentum",          # new value
          "pattern": "4x1",                   # new field
          "dense_ratio": 0.1,                 # new value
          "gradient_accumulation_steps": 1,   # new field
          "sparsity_decay_type": "exp",       # new field
          "start_step": 0,                    # new field
          "end_tep": 10000                    # new field
        },
        "different_groups": {
          "sp1": {
            "params": {
              "dense_ratio": 0.5
            },
            "modules": [
              "attention.self"
            ]
          }
        }
      },
}
~~~

As for the structural sparsity implementation in `compression` dir, let's taking `LinearLayer_Compress` class in `deepspeed/compression/basic_layer.py` as an example, this class is enhanced like this to support structural sparsity algorithm. 

<a target="_blank" href="./pics/linear_example.png">
  <img src="./pics/linear_example.png" alt="Extension" width="100%" height="100%">
</a>

**NOTE**: In this phase 1, the DeepSpeed user facing API keeps unchanged. The only change user need to be aware of is the extended Json file format. 

**Phase 2** Advanced structural pruning functionality which supports the adaptive sparse ratio adjustment algorithm per layer to reach pre-defined accuracy goal.

This way needs to extend the `initialize()` API to return one more parameter `callbacks` besides ``engine``, ``optimizer``, ``training_dataloader``, ``lr_scheduler``. The json config file needs to be adjusted accordingly.

~~~python
def initialize(args=None,
               model: torch.nn.Module = None,
               optimizer: Optional[Union[Optimizer,
                                         DeepSpeedOptimizerCallable]] = None,
               model_parameters: Optional[torch.nn.Module] = None,
               training_data: Optional[torch.utils.data.Dataset] = None,
               lr_scheduler: Optional[Union[_LRScheduler,
                                            DeepSpeedSchedulerCallable]] = None,
               mpu=None,
               dist_init_required: Optional[bool] = None,
               collate_fn=None,
               config=None,
               config_params=None):
    # return A tuple of ``engine``, ``optimizer``, ``training_dataloader``, ``lr_scheduler``, ``callbacks``
~~~

This `callbacks` class object returned by `initialize` function is used to register hooks for user into the normal training process.

~~~python
class callbacks():
    def on_epoch_begin(self, epoch):
      ...
    
    def on_epoch_end(self):
      ...

    def on_step_begin(self, step):
      ...
    
    def on_step_end(self):
      ...
    
    ...  # other hooks during training

~~~

The user need to manually insert such hooks into their training code for fine-grain sparsity control per layer.

~~~python
    model, optimizer, _, lr_scheduler, callbacks = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    for epoch in range(args.num_train_epochs):
        model.train()
        start_time = time.time()
        callbacks.on_epoch_start(epoch)               # new code
        for step, batch in enumerate(train_dataloader):
            callbacks.on_step_start(step)             # new code
            batch = to_device(batch, device)
            all_loss = forward_fun(batch, model, teacher_model=teacher_model)
            model.backward(all_loss[0])
            model.step()
            callbacks.on_step_end()                   # new code
        
        callbacks.on_epoch_end()                      # new code
        ...

~~~

## **Structural Sparsity Results**
As for the structural sparsity results, please refer to below chart.

<a target="_blank" href="./pics/sparse_result.png">
  <img src="./pics/sparse_result.png" alt="Extension" width="80%" height="80%">
</a>

## **Summary**

We recommend to split this contribution into two phases. The first phase focuses on adding the entire structural sparsity methods supported by [Intel(R) Neural Compressor](https://github.com/intel/neural-compressor) into DeepSpeed with minor changes. This way provoides the complete structural sparsity capability except for the adaptive sparse ratio adjustment. The second phase focuses on productivity improvement by supporting the adaptive sparse ratio adjustment to support broad pruning algorithm. By this way, user just need set a global sparse goal, the structural sparsity algorithm will automatically adjust the sparse ration per layer for better accuracy.

