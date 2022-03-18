tf_example8 example
=====================
This example is used to demonstrate how to quantize a TensorFlow model with pure python API.  

### 1. Installation
```shell
pip install -r requirements.txt
```

### 2. Download the FP32 model
```shell
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/mobilenet_v1_1.0_224_frozen.pb
```

### 3. Run Command
```shell
python test.py
``` 

### 4. Introduction
We can create a quantizer without config yaml, only need to set the dataloader with dummy dataset to generate an int8 model.   
```python
    quantizer = Quantization()
    quantizer.model = './mobilenet_v1_1.0_224_frozen.pb'
    dataset = quantizer.dataset('dummy', shape=(20, 224, 224, 3))
    quantizer.calib_dataloader = common.DataLoader(dataset)
    quantized_model = quantizer.fit()
```