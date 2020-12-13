Step-by-Step
============

This is Hello World to demonstrate how to quick start with Intel® Low Precision Optimization Tool.The example is based on frozen pb and mnist dataset and it will use a model self defined evalutor function to do quantization.


## Prerequisite

### 1. Installation
```Shell
# Install Intel® Low Precision Optimization Tool
pip install lpot
```
### 2. Install Intel Tensorflow
```shell
pip install intel-tensorflow==1.15.2
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
    model:
      name: helloworld
      framework: tensorflow                         # possible values are tensorflow, mxnet and pytorch
      inputs: 'x'                               
      outputs: 'Identity'
```
### 2. Run lpot to get the quantized Graph. 
```PyThon
    model_file = "../frozen_models/simple_frozen_graph.pb"
    graph = load_graph(model_file)

    # Run lpot to get the quantized pb, eval_func is a model self defined evaluator.
    quantizer = Quantization('./conf.yaml')
    dataloader = quantizer.dataloader(dataset=list(zip(test_images, test_labels)))
    quantized_model = quantizer(graph, q_dataloader=dataloader, eval_func=eval_func)

```
### 3. Run quantized model.
```PyThon
    # Run quantized model 
    with tf.compat.v1.Graph().as_default(), tf.compat.v1.Session() as sess:
        tf.compat.v1.import_graph_def(quantized_model.as_graph_def(), name='')
        styled_image = sess.run(['Identity:0'], feed_dict={'x:0':test_images})
    
```
 
