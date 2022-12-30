tf_example7 example
=====================
This example is used to demonstrate how to quantize a TensorFlow model with dummy dataset.

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
We can quantize a model only needing to set the dataloader with dummy dataset to generate an int8 model.
```python
    dataset = Datasets('tensorflow')['dummy'](shape=(1, 224, 224, 3))
    from neural_compressor.quantization import fit
    config = PostTrainingQuantConfig()
    quantized_model = fit(
        model="./mobilenet_v1_1.0_224_frozen.pb",
        conf=config,
        calib_dataloader=DataLoader(framework='tensorflow', dataset=dataset),
        eval_dataloader=DataLoader(framework='tensorflow', dataset=dataset))
```