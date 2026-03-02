Torch
=================================================

1. [Introduction](#introduction)
2. [Torch-like APIs](#torch-like-apis)
3. [Support matrix](#supported-matrix)
4. [Common Problems](#common-problems)

## Introduction

`neural_compressor.torch` provides a Torch-like API and integrates various model compression methods fine-grained to the torch.nn.Module. Supports a comprehensive range of models, including but not limited to CV models, NLP models, and large language models. A variety of quantization methods are available, including classic INT8 quantization, SmoothQuant, and the popular weight-only quantization. Neural compressor also provides the latest research in simulation work, such as FP8 emulation quantization, MX data type emulation quantization.

In terms of ease of use, neural compressor is committed to providing an easy-to-use user interface and easy to extend the structure design, on the one hand, reuse the PyTorch prepare, convert API, on the other hand, through the Quantizer base class for prepare and convert customization to provide a convenient.

For more details, please refer to [link](https://github.com/intel/neural-compressor/discussions/1527) in Neural Compressor discussion space.

So far, `neural_compressor.torch` still relies on the backend to generate the quantized model and run it on the corresponding backend, but in the future, neural_compressor is planned to provide generalized device-agnostic Q-DQ model, so as to achieve one-time quantization and arbitrary deployment.

## Torch-like APIs

Currently, we provide below three user scenarios, through `prepare`&`convert`, `autotune` and `load` APIs.

- One-time quantization of the model
- Get the best quantized model by setting the search scope and target
- Direct deployment of the quantized model

### Quantization APIs

```python
def prepare(
    model: torch.nn.Module,
    quant_config: BaseConfig,
    inplace: bool = True,
    example_inputs: Any = None,
):
    """Prepare the model for calibration.

    Insert observers into the model so that it can monitor the input and output tensors during calibration.

    Args:
        model (torch.nn.Module): origin model
        quant_config (BaseConfig): path to quantization config
        inplace (bool, optional): It will change the given model in-place if True.
        example_inputs (tensor/tuple/dict, optional): used to trace torch model.

    Returns:
        prepared and calibrated module.
    """
```

```python
def convert(
    model: torch.nn.Module,
    quant_config: BaseConfig = None,
    inplace: bool = True,
):
    """Convert the prepared model to a quantized model.

    Args:
        model (torch.nn.Module): the prepared model
        quant_config (BaseConfig, optional): path to quantization config, for special usage.
        inplace (bool, optional): It will change the given model in-place if True.

    Returns:
        The quantized model.
    """
```

### Autotune API

```python
def autotune(
    model: torch.nn.Module,
    tune_config: TuningConfig,
    eval_fn: Callable,
    eval_args=None,
    run_fn=None,
    run_args=None,
    example_inputs=None,
):
    """The main entry of auto-tune.

    Args:
        model (torch.nn.Module): _description_
        tune_config (TuningConfig): _description_
        eval_fn (Callable): for evaluation of quantized models.
        eval_args (tuple, optional): arguments used by eval_fn. Defaults to None.
        run_fn (Callable, optional): for calibration to quantize model. Defaults to None.
        run_args (tuple, optional): arguments used by run_fn. Defaults to None.
        example_inputs (tensor/tuple/dict, optional): used to trace torch model. Defaults to None.

    Returns:
        The quantized model.
    """
```

### Load API

`neural_compressor.torch` links the save function to the quantized model. If `model.save` already exists, Neural Compressor renames the previous function to `model.orig_save`.

```python
def save(self, output_dir="./saved_results"):
"""
    Args:
        self (torch.nn.Module): the quantized model.
        output_dir (str, optional): path to save the quantized model 
"""
```

```python
def load(output_dir="./saved_results", model=None):
    """The main entry of load for all algorithms.

    Args:
        output_dir (str, optional): path to quantized model folder. Defaults to "./saved_results".
        model (torch.nn.Module, optional): original model, suggest to use empty tensor.

    Returns:
        The quantized model
    """
```

## Supported Matrix

<table class="tg"><thead>
  <tr>
    <th class="tg-9wq8">Method<br></th>
    <th class="tg-9wq8">Algorithm</th>
    <th class="tg-9wq8">Backend</th>
    <th class="tg-9wq8">Support Status</th>
    <th class="tg-9wq8">Usage Link</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="6">Weight Only Quantization<br></td>
    <td class="tg-9wq8">Round to Nearest (RTN)<br></td>
    <td class="tg-9wq8">PyTorch eager mode</td>
    <td class="tg-9wq8">&#10004</td>
    <td class="tg-9wq8"><a href="PT_WeightOnlyQuant.md#rtn">link</a></td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href=https://arxiv.org/abs/2210.17323>GPTQ</a><br></td>
    <td class="tg-9wq8">PyTorch eager mode</td>
    <td class="tg-9wq8">&#10004</td>
    <td class="tg-9wq8"><a href="PT_WeightOnlyQuant.md#gptq">link</a></td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href=https://arxiv.org/abs/2306.00978>AWQ</a></td>
    <td class="tg-9wq8">PyTorch eager mode</td>
    <td class="tg-9wq8">&#10004</td>
    <td class="tg-9wq8"><a href="PT_WeightOnlyQuant.md#awq">link</a></td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href=https://arxiv.org/abs/2309.05516>AutoRound</a></td>
    <td class="tg-9wq8">PyTorch eager mode</td>
    <td class="tg-9wq8">&#10004</td>
    <td class="tg-9wq8"><a href="PT_WeightOnlyQuant.md#autoround">link</a></td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href=https://arxiv.org/abs/2310.10944>TEQ</a></td>
    <td class="tg-9wq8">PyTorch eager mode</td>
    <td class="tg-9wq8">&#10004</td>
    <td class="tg-9wq8"><a href="PT_WeightOnlyQuant.md#teq">link</a></td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href=https://mobiusml.github.io/hqq_blog>HQQ</a></td>
    <td class="tg-9wq8">PyTorch eager mode</td>
    <td class="tg-9wq8">&#10004</td>
    <td class="tg-9wq8"><a href="PT_WeightOnlyQuant.md#hqq">link</a></td>
  </tr>
  <tr>
    <td class="tg-9wq8">Smooth Quantization</td>
    <td class="tg-9wq8"><a href=https://proceedings.mlr.press/v202/xiao23c.html>SmoothQuant</a></td>
    <td class="tg-9wq8"><a href=https://pytorch.org/tutorials/recipes/recipes/intel_extension_for_pytorch.html>intel-extension-for-pytorch</a></td>
    <td class="tg-9wq8">&#10004</td>
    <td class="tg-9wq8"><a href="PT_SmoothQuant.md">link</a></td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="3">Static Quantization</td>
    <td class="tg-9wq8" rowspan="3"><a href=https://pytorch.org/docs/master/quantization.html#post-training-static-quantization>Post-traning Static Quantization</a></td>
    <td class="tg-9wq8">intel-extension-for-pytorch (INT8)</td>
    <td class="tg-9wq8">&#10004</td>
    <td class="tg-9wq8"><a href="PT_StaticQuant.md">link</a></td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href=https://pytorch.org/docs/stable/torch.compiler_deepdive.html>TorchDynamo (INT8)</a></td>
    <td class="tg-9wq8">&#10004</td>
    <td class="tg-9wq8"><a href="PT_StaticQuant.md">link</a></td>
  <tr>
    <td class="tg-9wq8"><a href=https://docs.habana.ai/en/latest/index.html>Intel Gaudi AI accelerator (FP8)</a></td>
    <td class="tg-9wq8">&#10004</td>
    <td class="tg-9wq8"><a href="PT_FP8Quant.md">link</a></td>
  </tr>
  <tr>
    <td class="tg-9wq8">Dynamic Quantization</td>
    <td class="tg-9wq8"><a href=https://pytorch.org/docs/master/quantization.html#post-training-dynamic-quantization>Post-traning Dynamic Quantization</a></td>
    <td class="tg-9wq8">TorchDynamo</td>
    <td class="tg-9wq8">&#10004</td>
    <td class="tg-9wq8"><a href="PT_DynamicQuant.md">link</a></td>
  </tr>
  <tr>
    <td class="tg-9wq8">MX Quantization</td>
    <td class="tg-9wq8"><a href=https://arxiv.org/pdf/2310.10537>Microscaling Data Formats for
Deep Learning</a></td>
    <td class="tg-9wq8">PyTorch eager mode</td>
    <td class="tg-9wq8">&#10004</td>
    <td class="tg-9wq8"><a href="PT_MXQuant.md">link</a></td>
  </tr>
  <tr>
    <td class="tg-9wq8">Mixed Precision</td>
    <td class="tg-9wq8"><a href=https://arxiv.org/abs/1710.03740>Mixed precision</a></td>
    <td class="tg-9wq8">PyTorch eager mode</td>
    <td class="tg-9wq8">&#10004</td>
    <td class="tg-9wq8"><a href="PT_MixedPrecision.md">link</a></td>
  </tr>
  <tr>
    <td class="tg-9wq8">Quantization Aware Training</td>
    <td class="tg-9wq8"><a href=https://pytorch.org/docs/master/quantization.html#quantization-aware-training-for-static-quantization>Quantization Aware Training</a></td>
    <td class="tg-9wq8">PyTorch eager mode</td>
    <td class="tg-9wq8">&#10004</td>
    <td class="tg-9wq8"><a href="PT_QAT.md">link</a></td>
  </tr>
</tbody></table>

## Common Problems

1. How to choose backend between `intel-extension-for-pytorch` and `PyTorchDynamo`?
    > Neural Compressor provides automatic logic to detect which backend should be used.
    > <table class="tg"><thead>
    <tr>
        <th class="tg-9wq8">Environment</th>
        <th class="tg-9wq8">Automatic Backend</th>
    </tr></thead>
    <tbody>
    <tr>
        <td class="tg-9wq8">import torch</td>
        <td class="tg-9wq8">torch.dynamo</td>
    </tr>
    <tr>
        <td class="tg-9wq8">import torch<br>import intel-extension-for-pytorch</td>
        <td class="tg-9wq8">intel-extension-for-pytorch</td>
    </tr>
    </tbody>
    </table>

2. How to set different configuration for specific op_name or op_type?
    > Neural Compressor extends a `set_local` method based on the global configuration object to set custom configuration.

    ```python
    def set_local(self, operator_name_or_list: Union[List, str, Callable], config: BaseConfig) -> BaseConfig:
        """Set custom configuration based on the global configuration object.

        Args:
            operator_name_or_list (Union[List, str, Callable]): specific operator
            config (BaseConfig): specific configuration
        """
    ```

    > Demo:

    ```python
    quant_config = RTNConfig()  # Initialize global configuration with default bits=4
    quant_config.set_local(".*mlp.*", RTNConfig(bits=8))  # For layers with "mlp" in their names, set bits=8
    quant_config.set_local("Conv1d", RTNConfig(dtype="fp32"))  # For Conv1d layers, do not quantize them.
    ```

3. How to specify an accelerator?

    > Neural Compressor provides automatic accelerator detection, including HPU, Intel GPU, CUDA, and CPU.

    > The automatically detected accelerator may not be suitable for some special cases, such as poor performance, memory limitations. In such situations, users can override the detected accelerator by setting the environment variable `INC_TARGET_DEVICE`.

    > Usage:

    ```bash
    export INC_TARGET_DEVICE=cpu
    ```
