Step-by-Step
============

This is Hello World to demonstrate how to quick start with Intel® Low Precision Optimization Tool.The Hello World is a Keras model with mnist dataset.


## Prerequisite

### 1. Installation
```Shell
# Install Intel® Low Precision Optimization Tool
pip install ilit
```
### 2. Install Intel Tensorflow
```shell
pip install intel-tensorflow==2.2.0
```

### 3. Run Command
  # The cmd of quantization and predict with the quantized model 
  ```Shell
  python test.py 
  ```
### Examples of enabling Intel® Low Precision Optimization Tool 
This exmaple can demonstrate the steps to do quantization on Keras generated saved model. 
### 1.Add inputs and outputs information into conf.yaml

    model:
      name: helloworld
      framework: tensorflow                         # possible values are tensorflow, mxnet and pytorch
      inputs: 'args_0'                                                       
      outputs: 'Identity'


### 2. Gererate the quantized model. 
```PyThon

    # Load saved model
    model = tf.keras.models.load_model("../models/simple_model")

    # Run ilit to get the quantized graph 
    quantizer = Quantization('./conf.yaml')
    dataloader = quantizer.dataloader(dataset=(test_images, test_labels))
    quantized_model = quantizer(model, q_dataloader=dataloader, eval_func=eval_func)

```
### 3. Run quantized model.
```PyThon

    # Get the concrete_function from quantized model
    concrete_function = get_concrete_function(graph_def=quantized_model.as_graph_def(),
                                     inputs=["args_0:0"],
                                     outputs=["Identity:0"],
                                     print_graph=True)

    # Run inference with quantized model 
    frozen_graph_predictions = concrete_function(args_0=tf.constant(test_images))[0]
   
    
 
```
 
