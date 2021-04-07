tf_example4 example
=====================
This example is used to demonstrate how to quantize a TensorFlow checkpoint and run with a dummy dataloader.  

1. Download the FP32 model
    ```
    git clone https://github.com/openvinotoolkit/open_model_zoo.git
    python ./open_model_zoo/tools/downloader/downloader.py --name rfcn-resnet101-coco-tf --output_dir model 
    ```

2. Run quantizaiton
We will create a dummy dataloader and only need to add the following lines for quantization to create an int8 model.
    ```python
    quantizer = Quantization('./conf.yaml')
    dataset = quantizer.dataset('dummy', shape=(100, 100, 100, 3), label=True)
    quantizer.model = common.Model('./model/public/rfcn-resnet101-coco-tf/rfcn_resnet101_coco_2018_01_28/')
    quantizer.calib_dataloader = common.DataLoader(dataset)
    quantized_model = quantizer()
    
    ```
* Run quantization and evaluation:
```shell
    python test.py
``` 

