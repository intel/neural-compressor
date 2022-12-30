tf_example4 example
=====================
This example is used to demonstrate how to quantize a TensorFlow checkpoint and run with a dummy dataloader.  

### 1. Installation
```shell
pip install -r requirements.txt
```

### 2. Download the FP32 model
```shell
git clone https://github.com/openvinotoolkit/open_model_zoo.git
cd open_model_zoo
git checkout 2021.4
cd ..
python ./open_model_zoo/tools/downloader/downloader.py --name rfcn-resnet101-coco-tf --output_dir model 
```

### 3. Run Command
```shell
python test.py
``` 

### 4. Introduction
We will create a dummy dataloader and only need to add the following lines for quantization to create an int8 model.
```python
    dataset = Datasets('tensorflow')['dummy_v2']( \
        input_shape=(100, 100, 3), label_shape=(1, ))

    config = PostTrainingQuantConfig(
            inputs=['image_tensor'],
            outputs=['detection_boxes', 'detection_scores', 'detection_classes', 'num_detections'],
            calibration_sampling_size=[20]
            )
    quantized_model = fit(
        model='./model/public/rfcn-resnet101-coco-tf/rfcn_resnet101_coco_2018_01_28/',
        conf=config,
        calib_dataloader=DataLoader(framework='tensorflow', dataset=dataset, batch_size=1))

```
