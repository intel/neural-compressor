# ImageNet Quantization

This implements quantization of popular model architectures, such as ResNet on the ImageNet dataset.

## Requirements

- Install requirements
- `pip install -r requirements.txt`
- Download the ImageNet dataset from http://www.image-net.org/
  - Then, move and extract the training and validation images to labeled subfolders, using [the following shell script](extract_ILSVRC.sh)

## Quantizaiton

To quant a model and validate accaracy, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

```bash
export ImageNetDataPath=/path/to/imagenet
python main.py $ImageNetDataPath --pretrained -a resnet18 --tune --calib_iters 5
```


## Use Dummy Data

ImageNet dataset is large and time-consuming to download. To get started quickly, run `main.py` using dummy data by "--dummy". Note that the loss or accuracy is useless in this case.

```bash
python main.py -a resnet18 --dummy -q -e
```