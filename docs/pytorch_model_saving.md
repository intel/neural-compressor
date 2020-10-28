# Introduction
This document provides solutions regarding the saving and loading of tuned models with Intel® Low Precision Optimization Tool.

PyTorch 
================================
# Design
For PyTorch eager model, Intel® Low Precision Optimization Tool will automatically save tuning configure and weights of model which meet target goal when tuning process.
```python
# In ilit/strategy/strategy.py
def stop(self, timeout, trials_count):
    if self.objective.compare(self.best_tune_result, self.baseline):
        ......
        self.adaptor.save(self.best_qmodel, os.path.dirname(self.deploy_path))

# In ilit/adaptor/pytorch.py
def save(self, model, path):
    '''The function is used by tune strategy class for saving model.

       Args:
           model (object): The model to saved.
           path (string): The path where to save.
    '''

    path = os.path.expanduser(path)
    os.makedirs(path, exist_ok=True)
    try:
        with open(os.path.join(path, "best_configure.yaml"), 'w') as f:
            yaml.dump(self.tune_cfg, f, default_flow_style=False)
    except IOError as e:
        logger.error("Unable to save configure file. %s" % e)

    torch.save(model.state_dict(), os.path.join(path, "best_model_weights.pt"))
```
Here, deploy_path is defined in configure yaml file. Default path is ./ilit_workspace/$framework/$module_name/, this folder will saving tuning history, deploy yaml, tuning configure and model weights. Tuning configure and weights files name are "best_configure.yaml" and "best_model_weights.pt".

```yaml
tuning:
  workspace:
    path: /path/to/saving/directory
```
If you want get tuned model, you can load tuning configure and weights in saving folder.

```python
# In utils/pytorch.py
def load(tune_cfg_file, weights_file, model):
    """Execute the quantize process on the specified model.

    Args:
        tune_cfg_file (file): the tune configure file.
        model (object): fp32 model need to do quantization.

    Returns:
        (object): quantized model
    """

    assert os.path.exists(os.path.expanduser(tune_cfg_file)), \
           "tune configure file %s didn't exist" % tune_cfg_file
    assert os.path.exists(os.path.expanduser(weights_file)), \
           "weight file %s didn't exist" % weights_file

    q_model = copy.deepcopy(model.eval())

    with open(os.path.expanduser(tune_cfg_file), 'r') as f:
        tune_cfg = yaml.load(f, Loader=yaml.UnsafeLoader)

    op_cfgs = _cfg_to_qconfig(tune_cfg)
    _propagate_qconfig(q_model, op_cfgs)
    # sanity check common API misusage
    if not any(hasattr(m, 'qconfig') and m.qconfig for m in q_model.modules()):
        logger.warn("None of the submodule got qconfig applied. Make sure you "
                    "passed correct configuration through `qconfig_dict` or "
                    "by assigning the `.qconfig` attribute directly on submodules")
    add_observer_(q_model)
    q_model = convert(q_model, inplace=True)
    weights = torch.load(os.path.expanduser(weights_file))
    q_model.load_state_dict(weights)
    return q_model
```

# Usage
* Saving model:  
Intel® Low Precision Optimization Tool will automatically save tuning configure and weights of model which meet target goal when tuning process.
* loading model:  
```python
model                 # fp32 model
from ilit.utils.pytorch import load
quantized_model = load(
    os.path.join(Path, 'best_configure.yaml'),
    os.path.join(Path, 'best_model_weights.pt'), model)
```

# Examples
[example of PyTorch resnet50](../examples/pytorch/image_recognition/imagenet/cpu/ptq/README.md)


