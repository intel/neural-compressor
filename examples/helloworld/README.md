# Hello World Examples

## PyTorch Examples
### Examples List
* [torch_llm](/examples/helloworld/torch_llm): apply the weight-only quantization to LLMs.
* [torch_non_llm](/examples/helloworld/torch_non_llm): apply the static quantization to non-LLMs.

## Tensorflow Examples
### Prerequisite
Enter the following commands to prepare a dataset and pretrained models for the included Hello World examples:

```shell
pip install intel-tensorflow==2.10.0
python train.py
```
The `train.py` script generates a saved model and a frozen pb at ./models for your use.
> Note: Validated TensorFlow [Version](/docs/source/installation_guide.md#validated-software-environment).

### Examples List
* [tf_example1](/examples/helloworld/tf_example1): quantize with built-in dataloader and metric.
* [tf_example2](/examples/helloworld/tf_example2): quantize keras model with customized metric and dataloader.
* [tf_example3](/examples/helloworld/tf_example3): convert model with mix precision.
* [tf_example4](/examples/helloworld/tf_example4): quantize checkpoint with dummy dataloader.
* [tf_example5](/examples/helloworld/tf_example5): configure performance and accuracy measurement.
* [tf_example6](/examples/helloworld/tf_example6): use default user-facing APIs to quantize a pb model.
* [tf_example7](/examples/helloworld/tf_example7): quantize model with dummy dataset.
