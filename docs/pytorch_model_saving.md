# Introduction

This document provides solutions regarding the saving and loading of tuned models with Intel® Low Precision Optimization Tool.

PyTorch
=======

# Design

### Without Intel PyTorch Extension(IPEX)

For PyTorch eager model, the tool will save the tuned configure and weights of model which meet target goal by lpot.model save function.

```python
# In lpot/model/model.py
class PyTorchModel(PyTorchBaseModel):
    ...
    def save(self, path):
        path = os.path.expanduser(path)
        os.makedirs(path, exist_ok=True)
        try:
            with open(os.path.join(path, "best_configure.yaml"), 'w') as f:
                yaml.dump(self.tune_cfg, f, default_flow_style=False)
            torch.save(self._model.state_dict(), os.path.join(path, "best_model_weights.pt"))
            logger.info("save config file and weights of quantized model to path %s" % path)
        except IOError as e:
            logger.error("Unable to save configure file and weights. %s" % e)
```

Here, the model is class of PyTorchModel, The tuned configure and weights files named "best_configure.yaml" and "best_model_weights.pt" will be saved to the given path.

If you want to get the tuned model, you can load tuned configure and weights in the saving folder.

```python
# In utils/pytorch.py
def load(checkpoint_dir, model):
    """Execute the quantize process on the specified model.

    Args:
        checkpoint_dir (dir): The folder of checkpoint.
                              'best_configure.yaml' and 'best_model_weights.pt' are needed
                              in This directory. 'checkpoint' dir is under workspace folder
                              and workspace folder is define in configure yaml file.
        model (object): fp32 model need to do quantization.

    Returns:
        (object): quantized model
    """

    tune_cfg_file = os.path.join(os.path.abspath(os.path.expanduser(checkpoint_dir)),
                                 'best_configure.yaml')
    weights_file = os.path.join(os.path.abspath(os.path.expanduser(checkpoint_dir)),
                                'best_model_weights.pt')
    assert os.path.exists(
        tune_cfg_file), "tune configure file %s didn't exist" % tune_cfg_file
    assert os.path.exists(
        weights_file), "weight file %s didn't exist" % weights_file

    q_model = copy.deepcopy(model.eval())

    with open(tune_cfg_file, 'r') as f:
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
    weights = torch.load(weights_file)
    q_model.load_state_dict(weights)
    return q_model
```

### With IPEX

If you use IPEX to tune model, the tool will only save the tuned configure which meet target goal by lpot.model save function.

```python
class PyTorchIpexModel(PyTorchBaseModel):
    ...
    def save(self, path):
        path = os.path.expanduser(path)
        os.makedirs(path, exist_ok=True)
        try:
            with open(os.path.join(path, "best_configure.json"), 'w') as f:
                json.dump(self.tune_cfg, f)
            logger.info("save config file of quantized model to path %s" % path)
        except IOError as e:
            logger.error("Unable to save configure file and weights. %s" % e)
```

If you want to run the tuned model, you can load the tuned configure in the saving folder.

# Usage

* Saving model:

```python
from lpot import Quantization
quantizer = Quantization("./conf.yaml")
LPOT_model = quantizer(model)
LPOT_model.save("./saved_path")
```

For IPEX backend, only the tuned configure  "best_configure.json" will be saved, and for non_IPEX backend, the tuned configure and weights files named "best_configure.yaml" and "best_model_weights.pt" will be saved.

* Loading model:

```python
# Without IPEX
model                 # fp32 model
from lpot.utils.pytorch import load
quantized_model = load(
    os.path.abspath(os.path.expanduser(Path)), model)

# With IPEX
import intel_pytorch_extension as ipex
model                 # fp32 model
model.to(ipex.DEVICE)
try:
    new_model = torch.jit.script(model)
except:
    new_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224).to(ipex.DEVICE))
ipex_config_path = os.path.join(os.path.expanduser(args.tuned_checkpoint),
                                "best_configure.json")
conf = ipex.AmpConf(torch.int8, configure_file=ipex_config_path)
with torch.no_grad():
    for i, (input, target) in enumerate(val_loader):
        with ipex.AutoMixPrecision(conf, running_mode='inference'):
            output = new_model(input.to(ipex.DEVICE))
```

# Examples

[example of PyTorch resnet50](../examples/pytorch/image_recognition/imagenet/cpu/ptq/README.md)
