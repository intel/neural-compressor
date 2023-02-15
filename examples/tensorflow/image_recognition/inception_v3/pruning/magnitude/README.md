Step-by-Step
============

This document is used to list steps of reproducing Intel® Neural Compressor magnitude pruning feature on Inception-V3 model.


# Prerequisite

## 1. Environment

### Install Intel® Neural Compressor
```shell
pip install neural-compressor
```
### Install other requirements
```shell
pip install -r requirements.txt
```

## 2. Prepare Dataset
TensorFlow [models](https://github.com/tensorflow/models) repo provides [scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data) to download, process and convert the ImageNet dataset to the TF records format.
We also prepared related scripts in `imagenet_prepare` directory. To download the raw images, the user must create an account with image-net.org. If you have downloaded the raw data and preprocessed the validation data by moving the images into the appropriate sub-directory based on the label (synset) of the image. we can use below command ro convert it to tf records format.

```shell
cd examples/tensorflow/image_recognition/inception_v3/
# convert validation subset
bash prepare_dataset.sh --output_dir==/pruning/magnitude/data/  --raw_dir=/PATH/TO/img_raw/val/ --subset=validation
# convert train subset
bash prepare_dataset.sh --output_dir=/pruning/magnitude/data/ --raw_dir=/PATH/TO/img_raw/train/ --subset=train
cd ./pruning/magnitude/
```

# Run
Run the command to get baseline model which will be saved into './Inception-V3_Model'. Then it will be pruned and saved into a given path.

```shell
python main.py --output_model=/path/to/output_model/ --dataset_location=/path/to/dataset
```

If you want to accelerate pruning with multi-node distributed training and evaluation, you only need to add twp arguments and use horovod to run main.py.  Run the command to get pruned model with multi-node distributed training and evaluation.

```shell
horovodrun -np <num_of_processes> -H <hosts> python main.py --output_model=/path/to/output_model/ --dataset_location=/path/to/dataset --train_distributed --evaluation_distributed
```
