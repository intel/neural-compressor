Step-by-Step
============

This is Hello World to demonstrate how to quick start with Intel® Low Precision Optimization Tool.The example is based on frozen pb and mnist dataset.


## Prerequisite

### 1. Installation
```Shell
# Install Intel® Low Precision Optimization Tool
pip install ilit
```
### 2. Install Intel Tensorflow
```shell
pip install intel-tensorflow==1.5.3
```
### 3. Run Command
  # The cmd of quantization and predict with the quantized model 
  ```Shell
  python test.py 
  ```
### Examples of enabling Intel® Low Precision Optimization Tool 
This exmaple can demonstrate the steps to do quantization on frozen pb. 
### 1. Prepare conf.yaml  
Add inputs and outputs information into conf.yaml
```
   framework:
  - name: tensorflow                         # possible values are tensorflow, mxnet and pytorch
  - inputs: 'x'                               
  - outputs: 'Identity'
```
### 2. Run ilit to get the quantized Graph. 
```PyThon
    # Run ilit to get the quantized pb
    tuner = ilit.Tuner('./conf.yaml')
    dataloader = tuner.dataloader(dataset=(test_images, test_labels))
    quantized_graph = tuner.tune(frozen_model.graph, q_dataloader=dataloader, eval_func=eval_func)
```
### 3. Run quantized model.
```PyThon
    # Run quantized model 
    with tf.compat.v1.Session() as sess:
        tf.compat.v1.import_graph_def(quantized_model.as_graph_def())
        styled_image = sess.run(['import/Identity:0'], feed_dict={'import/x:0':test_images})

```
 
