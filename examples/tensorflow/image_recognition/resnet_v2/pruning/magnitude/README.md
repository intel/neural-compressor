Step-by-Step
============

This document is used to list steps of reproducing Intel® Neural Compressor magnitude pruning feature.


## Prerequisite

### 1. Installation
```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```
### 2. Install TensorFlow 2.10.0 or above.
```shell
pip install tensorflow==2.10.0
```
### 3. Train and save a ResNet-V2 model
According to the following link [Trains a ResNet on the CIFAR10 dataset.](https://keras.io/zh/examples/cifar10_resnet), set 'version = 2' and train a ResNet-V2 model as the baseline.  Please add a line 'model.save("./ResNetV2_Model")' at the end of the code to save the model to the directory './ResNetV2_Model'.

```python
......
......
# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
model.save("./ResNetV2_Model") # Add a line at the end

```
## Run command to prune the model
Run the command to get pruned model which overwritten and saved into './ResNetV2_Model'.
```shell
python main.py
```
If you want to accelerate pruning with multi-node distributed training and evaluation, you only need to add a small amount of code and use horovod to run main.py.
As shown in main.py, uncomment two lines 'prune.train_distributed = True' and 'prune.evaluation_distributed = True' is all you need.
Use horovod to run main.py to get pruned model with multi-node distributed training and evaluation.
```shell
horovodrun -np <num_of_processes> -H <hosts> python main.py 
```

Run the command to get pruned model performance.
```shell
python benchmark.py   
```