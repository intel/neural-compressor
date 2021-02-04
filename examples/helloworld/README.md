Enter the following commands to prepare a dataset and pretrained models for the included Hello World examples:

```Shell
pip install intel-tensorflow==2.3.0
python train.py

```
The `train.py` script generates a saved model and a frozen pb for your use.

The following Hello World examples are available:

*  Buildin dataloader and metric, with pb model: [tf_example1](../../examples/helloworld/tf_example1/README.md)
*  Customized dataloader and metric, with Keras saved model: [tf_example2](../../examples/helloworld/tf_example2/README.md)
*  TensorFlow slim model: [tf_example3](../../examples/helloworld/tf_example3/README.md)
*  TensorFlow checkpoint: [tf_example4](../../examples/helloworld/tf_example4/README.md)
*  Enable benchmark for performanace and accuracy measurement: [tf_example5](../../examples/helloworld/tf_example5/README.md)

