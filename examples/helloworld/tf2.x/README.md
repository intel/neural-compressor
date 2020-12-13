Step-by-Step
============

This is Hello World to demonstrate how to quick start with Intel® Low Precision Optimization Tool. It is a Keras model with mnist dataset, will config an evaluator in yaml and do quantization.


## Prerequisite

### 1. Installation
```Shell
# Install Intel® Low Precision Optimization Tool
pip install lpot
```
### 2. Install Intel Tensorflow
```shell
pip install intel-tensorflow==2.3.0
```

### 3. Run Command
  # The cmd of quantization and predict with the quantized model 
  ```Shell
  python test.py 
  ```
### Examples of enabling Intel® Low Precision Optimization Tool 
This exmaple can demonstrate the steps to do quantization on Keras generated saved model. 
### 1.Add inputs and outputs information into conf.yaml, to get the input and out put, run the python code  

```
model = tf.keras.models.load_model("../models/simple_model")
print('input', model.input_names)
print('output', model.output_names)
```
conf.yaml: 

 model:
   name: helloworld
   framework: tensorflow                         # possible values are tensorflow, mxnet and pytorch
   inputs: input                                                       
   outputs: output 

### 2. Config an evaluator in yaml and the quantization tool will create an evaluator for you:
 evaluation: 
   accuracy:
     metric:
       topk: 1

### 3. Gererate the quantized model. 
```PyThon

    # Load saved model
    model = tf.keras.models.load_model("../models/simple_model")

    # Run lpot to get the quantized graph 
    import lpot
    quantizer = lpot.Quantization('./conf.yaml')
    dataloader = quantizer.dataloader(dataset=list(zip(test_images, test_labels)))
    quantized_model = quantizer(model, q_dataloader=dataloader, eval_dataloader=dataloader)

```
### 4. Run quantized model.
```PyThon
    # Run inference with quantized model
    concrete_function = get_concrete_function(graph_def=quantized_model.as_graph_def(),
                                     inputs=["input:0"],
                                     outputs=["output:0"],
                                     print_graph=True)

    frozen_graph_predictions = concrete_function(input=tf.constant(test_images))[0]

  
```
 
