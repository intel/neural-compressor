tf_example4 example
=====================
This example is used to demonstrate how to quantize a TensorFlow checkpoint and run with a dummy dataloader.  

1. Download the FP32 model
git clone https://github.com/openvinotoolkit/open_model_zoo.git
cd open_model_zoo/tools/downloader/

python ./open_model_zoo/tools/downloader/downloader.py --name rfcn-resnet101-coco-tf --output_dir model 


2. Run quantizaiton
We will create a dummy dataloader and only need to add the following lines for quantization to create an int8 model.
```python
    import lpot
    quantizer = lpot.Quantization('./conf.yaml')
    
    dataset = quantizer.dataset('dummy', shape=(100, 100, 100, 3), label=True)
    data_loader = DataLoader('tensorflow', dataset)

    quantizer = lpot.Quantization('./conf.yaml')
    quantized_model = quantizer('./model/public/rfcn-resnet101-coco-tf/model/public/rfcn-resnet101-coco-tf/rfcn_resnet101_coco_2018_01_28/', q_dataloader=data_loader )

```
* Run quantization and evaluation:
```shell
    python test.py
``` 

