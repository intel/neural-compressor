Step-by-Step
============

This document is used to list steps of reproducing Intel® Neural Compressor magnitude pruning feature on ViT model.


## Prerequisite

### 1. Installation
```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```
### 2. Install requirements
```shell
pip install -r requirements.txt
```
### 3. Train and save a ViT model
According to the following link [Image classification with Vision Transformer](https://github.com/keras-team/keras-io/blob/master/examples/vision/md/image_classification_with_vision_transformer.md), train a ViT model as the baseline. Please add a line 'model.save("./ViT_Model")' in the function 'def run_experiment' to save the model to the directory './ViT_Model'.

```python
def run_experiment(model):
......
......
    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
    model.save("./ViT_Model") # Add this line
    return history
......
......
```
## Run command to prune the model
Run the command to get pruned model which overwritten and saved into './ViT_Model'.
```shell
python main.py 
```
If you want to accelerate pruning with multi-node distributed training and evaluation, you only need to add a small amount of code and use horovod to run main.py. As shown in main.py, uncomment two lines 'prune.train_distributed = True' and 'prune.evaluation_distributed = True' in main.py is all you need. Run the command to get pruned model with multi-node distributed training and evaluation.
```shell
horovodrun -np <num_of_processes> -H <hosts> python main.py
```
