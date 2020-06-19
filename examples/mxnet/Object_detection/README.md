# Single Shot Multibox Object Detection

From [GluonCV SSD](https://github.com/dmlc/gluon-cv/blob/master/scripts/detection/ssd/eval_ssd.py)


# Quantization with iLiT
```
# SSD-Mobilenet1.0
python eval_ssd.py --network=mobilenet1.0 --data-shape=512 --batch-size=256 --datasete coco

# SSD-ResNet50_v1
python eval_ssd.py --network=renset50_v1 --data-shape=512 --batch-size=256 --datasete coco
```

# Dependancy

```
pip install mxnet-mkl gluoncv

```