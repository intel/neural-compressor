# Hello World Examples

Enter the following commands to prepare a dataset and pretrained models for the included Hello World examples:

```shell
pip install intel-tensorflow==2.7.0
python train.py

```
The `train.py` script generates a saved model and a frozen pb at ./models for your use.

The following Hello World examples are available:

* [tf_example1](/examples/helloworld/tf_example1): quantize with built-in dataloader and metric. 
* [tf_example2](/examples/helloworld/tf_example2): quantize keras model with customized metric and dataloader.  
* [tf_example3](/examples/helloworld/tf_example3): quantize slim model. 
* [tf_example4](/examples/helloworld/tf_example4): quantize checkpoint with dummy dataloader.  
* [tf_example5](/examples/helloworld/tf_example5): config performance and accuracy measurement.  
* [tf_example6](/examples/helloworld/tf_example6): use default user-facing APIs to quantize a pb model. 
* [tf_example7](/examples/helloworld/tf_example7): enable quantization and benchmark with python-flavor config.
* [tf_example8](/examples/helloworld/tf_example8): quantize with pure python API.

