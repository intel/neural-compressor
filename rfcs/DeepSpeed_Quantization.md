## **Summary**

This is a proposal of extending the pruning capability of DeepSpeed.

As we know, the DeepSpeed Compression supports these compression methods: layer reduction via knowledge distillation, weight quantization, activation quantization, sparse pruning, row pruning, head pruning, and channel pruning.

In this proposal, we would like to 

1. enhance post training quantization functionality to 

2. introduce device-agnostic structural pruning functionalities by enabling "N in M" and "N x M" block sparsity pattern with snip_momentum criteria and progressive pruning.

## **Motivation**

1. On post training quantization side, 

2. Current state-of-the-art deep neural networks have a great number of parameters, which lead to their heavy computing resource consumption during training and inference. To address this issue, network pruning (also known as network sparsity) has been proposed in order to make large models to be deployed on some edge devices with limited computation and memory footprint. However, when a high ratio of parameters has been pruned from the original dense model, its accuracy can be prone to drop. Therefore, pruning a model to a high sparsity while maintaining its original accuracy is a challenge. More pruning algorithms and utilizes need to be supported to meet the requirments of the users. 


## **Proposed Implementation on Pruning**

We have two proposals on the structural pruning support.

### Proposal 1:

The only change user aware of is on json config file like below, the DeepSpeed APIs keep unchanged.

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

The internal of DeepSpeed compression will be updated accordingly to support this new structural pruning.

Taking `deepspeed/compression/basic_layer.py` as an example, the `LinearLayer_Compress` class will be enhanced like. 

<a target="_blank" href="./deep_speed_flow.png">
  <img src="./deep_speed_flow2.png" alt="Extension" width="80%" height="100%">
</a>

**Pros**: No impact on existing example codes. only Json file needs to be modified.

**Cons**: Like other pruning methods supported by DeepSpeed, the sparsity ratio is global and each layer has same sparse ratio. This way limits to do fine-grain control on per layer's sparse ratio.

### proposal 2:

Modifying the `initialize()` API to return one more parameter `callbacks` besides ``engine``, ``optimizer``, ``training_dataloader``, ``lr_scheduler``.

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

This `callbacks` class object returned by `initialize` function is used to provide hooks for user use during training.

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

~~~

The user need to manually insert such hooks into their training code for fine-grain sparsity control on each layers.

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

~~~

The json config file needs to be adjusted accordingly.

**Pros**: Fine-grain control on per layer's sparse ratio. Easier to get the optimial sparse model which meet stricter accuracy goal.

**Cons**: Impact on existing example codes and Json file also needs to be modified.

## **Proposal Recommendation**
Phase 1 to add more pruning algorithms.
Phase 2 to add advanced usage to improve pruning productivity and reach higher accuracy.

At first phase, We recommend to proposal 1 as it has minimal impact on user. In the future, we prefer to 

## **Structural Sparsity Results**
As for the structural sparsity results, please refer to below chart.

<a target="_blank" href="./sparse_result.png">
  <img src="./sparse_result.png" alt="Extension" width="80%" height="80%">
</a>

## **Future work**

This RFC focus on quantization and pruning parts on those models which can be loaded into CPU or GPU easily. As you know, the large language models like `GPT3` or `Bloom` are getting more insights from industry. How to effecticvely load and inference such model is our next focus. we plan to apply those algos 

