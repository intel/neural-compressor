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
   framework:
  - name: tensorflow                         # possible values are tensorflow, mxnet and pytorch
  - inputs: 'x'                               
  - outputs: 'Identity'

### 2. Get ConcreteFunction from saved model 
```PyThon
    # Convert Keras model to ConcreteFunction with x as input
    full_model = tf.function(lambda x: model(x)) 
    concrete_function = full_model.get_concrete_function(
        x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_model = convert_variables_to_constants_v2(concrete_function)

```

### 3. Gererate the quantized model. 
```PyThon
    tuner = ilit.Tuner('./conf.yaml')
    dataloader = tuner.dataloader(dataset=(test_images, test_labels))
    quantized_model = tuner.tune(frozen_model.graph, q_dataloader=dataloader, eval_func=eval_func)
```
### 3. Run quantized model.
```PyThon

    concrete_function = get_concrete_function(graph_def=quantized_model.as_graph_def(),
                                     inputs=["x:0"],
                                     outputs=["Identity:0"],
                                     print_graph=True)

    # Run inference with quantized model
    frozen_graph_predictions = concrete_function(x=tf.constant(test_images))[0]
```
 
