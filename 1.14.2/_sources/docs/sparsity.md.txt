# Sparsity

Sparsity is one of promising model compression techniques that can be used to accelerate the deep learning inference. Typically, sparsity can be classified as 1) structured sparsity, and 2) unstructured sparsity. Structured sparsity indicates an observed structure pattern of zero (or non-zero) values, while unstructured sparsity indicates no such pattern for zero (or non-zero) values. In general, structured sparsity has lower accuracy due to restrictive structure than unstructured sparsity; however, it can accelerate the model execution significantly with software or hardware sparsity.

The document describes the sparsity definition, sparsity training flow, validated models, and performance benefit using software sparsity. Note that the document discusses the sparse weight (with dense activation) for inference acceleration. Sparse activation or sparse embedding for inference acceleration or training acceleration is out of the scope.

> **Note**: training for sparsity with 2:4 or similar structured pattern is supported, please refer it at our new [API](../neural_compressor/experimental/pytorch_pruner/), [question-answering examples](../examples/pytorch/nlp/huggingface_models/question-answering/pruning/pytorch_pruner/eager) and [text-classification examples](../examples/pytorch/nlp/huggingface_models/text-classification/pruning/pytorch_pruner/eager)

## Sparsity Definition
NVidia proposed [2:4 sparsity](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/) (or known as "2in4 sparsity") in Ampere architecture, for every 4 continuous elements in a matrix, two of them are zero and others are non-zero.

<a target="_blank" href="./docs/imgs/2in4_sparsity_demo.png">
    <img src="../docs/imgs/2in4_sparsity_demo.png" width=600 height=200 alt="Sparsity Pattern">
</a>

Different from 2:4 sparsity above, we propose the block-wise structured sparsity patterns that we are able to demonstrate the performance benefits on existing Intel hardwares even without the support of hardware sparsity. A block-wise sparsity pattern with block size ```S``` means the contiguous ```S``` elements in this block are all zero values.

For a typical GEMM, the weight dimension is ```IC``` x ```OC```, where ```IC``` is the number of input channels and ```OC``` is the number of output channels. Note that sometimes ```IC``` is also called dimension ```K```, and ```OC``` is called dimension ```N```. The sparsity dimension is on ```OC``` (or ```N```).

For a typical Convolution, the weight dimension is ```OC x IC x KH x KW```, where ```OC``` is the number of output channels, ```IC``` is the number of input channels, and ```KH``` and ```KW``` is the kernel height and weight. The sparsity dimension is also on ```OC```.

Here is a figure showing a matrix with ```IC``` = 32 and ```OC``` = 16 dimension, and a block-wise sparsity pattern with block size 4 on ```OC``` dimension.
<a target="_blank" href="./docs/imgs/sparse_dim.png">
    <img src="../docs/imgs/sparse_dim.png" width=854 height=479 alt="Sparsity Pattern">
</a>

## Training Flow & Sample Code
The following image describes the typical flow of training for sparsity. Compared with normal training flow, training for sparsity requires more steps (e.g., regularization and pruning) to meet the goal of sparsity ratio.

<a target="_blank" href="./docs/imgs/train_for_sparsity.png">
    <img src="../docs/imgs/train_for_sparsity.png" width=336 height=465 alt="Sparsity Training Flow">
</a>

Here is the pseudo code of a modified training function on ```PyTorch```.

```python
def train():
    for x,label in dataloader:
        y = model(x)
        loss = loss_func(y, label)
        optimizer.zero_grad()
        loss.backward()
        prune_gradient_with_magnitude()    # prune gradients
        group_lasso_regularize(alpha)     # update gradients by sparsification rate
        optimizer.step()
        lr_scheduler.step()
        prune_weights_with_magnitude()     # prune weights
```

## Validated Models

We validate the sparsity on typical models across different domains (including CV, NLP, and Recommendation System). The below table shows the sparsity pattern, sparsity ratio, and accuracy of sparse and dense (Reference) model for each model. We also provide a simplified [BERT example](../examples/pytorch/nlp/huggingface_models/question-answering/pruning/group_lasso/eager) with only one sparse layer.

|   Model   | Sparsity Pattern | Sparsity Ratio |Dataset| Accuracy (Sparse Model) | Accuracy (Dense Model) |
|-----------|:----------------:|:--------------:|:-------------:|:-----------------------:|:-----------------------:|
| Bert Large| [***2***x1](../examples/pytorch/nlp/huggingface_models/question-answering/pruning/group_lasso/eager)          | 70%            |SQuAD| 90.70%                  | 91.34%                  |
| DLRM      | 4x***16***         | 85%            |Criteo Terabyte| 80.29%                  | 80.25%                  |
| Bert Mini | [***4***x1](../examples/pytorch/nlp/huggingface_models/text-classification/pruning/pytorch_pruner/eager)         | 90%            |MRPC| 87.22%                  | 87.52%                  |
| Bert Mini | [***4***x1](../examples/pytorch/nlp/huggingface_models/text-classification/pruning/pytorch_pruner/eager)         | 90%            |SST-2| 86.92%                  | 87.61%                  |
| Bert Mini | [***4***x1](../examples/pytorch/nlp/huggingface_models/question-answering/pruning/pytorch_pruner/eager)         | 80%            |SQuAD| 76.27%                  | 76.87%                  |
| Bert Mini | [2 in ***4***](../examples/pytorch/nlp/huggingface_models/text-classification/pruning/pytorch_pruner/eager)       | 50%            |MRPC| 86.95%                  | 87.52%                  |
| Bert Mini | [2 in ***4***](../examples/pytorch/nlp/huggingface_models/text-classification/pruning/pytorch_pruner/eager)         | 50%            |SST-2| 86.93%                  | 87.61%                  |
| Bert Mini | [2 in ***4***](../examples/pytorch/nlp/huggingface_models/question-answering/pruning/pytorch_pruner/eager)        | 50%            |SQuAD| 76.85%                  | 76.87%                  |
|ResNet50 v1.5 | [***2***x1](../examples/pytorch/image_recognition/torchvision_models/pruning/magnitude/eager)         | 78%            |Image-Net| 75.3%                  | 76.13%                  |
|SSD-ResNet34 | ***2***x1         | 75%            |Coco| 22.85%                  | 23%                  |
|ResNext101| ***2***x1         | 73%            |Image-Net| 79.14%                  | 79.37%                  |

Note: 
* ***bold*** means the sparsity dimension (```OC```).
* Bert-Mini related examples are developed based on our [Pytorch Pruner API](../neural_compressor/experimental/pytorch_pruner/). Examples of [question answering](../examples/pytorch/nlp/huggingface_models/question-answering/pruning/pytorch_pruner/eager) and [text classification](../examples/pytorch/nlp/huggingface_models/text-classification/pruning/pytorch_pruner/eager) are developed.

## Performance

We explore kernels development with software sparsity and apply to DLRM, a very popular industrial recommendation model as one of [MLPerf](https://mlcommons.org/en/) benchmarks. We achieve 1.6x performance gains on INT8 sparse model over INT8 dense model, and 6.4x total performance gains over FP32 dense model in [MLPerf inference submissions](https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/A-Double-Play-for-MLPerf-Inference-Performance-Gains-with-3rd/post/1335759). We expect further performance speedup with the support of hardware sparsity. 

|   | Dense Model (FP32)  | Dense Model (INT8)   | Sparse Model (INT8)   |
|---|:---:|:---:|:---:|
|Accuracy   |80.25% (100%)   |80.21% (99.96%)   |79.91% (99.57%)   |
|Offline QPS   |5732   |23174 (1.0x)   |36883 (1.6x)   |
|Online QPS   |NA   |20245 (1.0x)   |30396 (1.5x)   |
