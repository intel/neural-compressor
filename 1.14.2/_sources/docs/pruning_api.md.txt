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

    @train_func.setter
    def train_func(self, user_pruning_func)
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

    def on_step_begin(self, batch):
        # The hook point used by pruning algorithm
        ...

    def on_step_end(self):
        # The hook point used by pruning algorithm
        ...

    def on_before_optimizer_step(self):
        # The hook point used by pruning algorithm
        ...

```

