# PeleeNet
PeleeNet: An efficient DenseNet architecture for mobile devices

An implementation of PeleeNet in PyTorch. PeleeNet is an efficient Convolutional Neural Network (CNN) architecture built with
conventional convolution. Compared to other efficient architectures,PeleeNet has a great speed advantage and esay to be applied to the computer vision tasks other than image classification. 

For more information, check the paper:
[Pelee: A Real-Time Object Detection System on Mobile Devices](https://arxiv.org/pdf/1804.06882.pdf) (NeurIPS 2018)

 

### Citation
If you find this work useful in your research, please consider citing:

```

@incollection{NIPS2018_7466,
title = {Pelee: A Real-Time Object Detection System on Mobile Devices},
author = {Wang, Robert J. and Li, Xiang and Ling, Charles X.},
booktitle = {Advances in Neural Information Processing Systems 31},
editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
pages = {1963--1972},
year = {2018},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/7466-pelee-a-real-time-object-detection-system-on-mobile-devices.pdf}
}


```
## Results on ImageNet ILSVRC 2012
The table below shows the results on the
ImageNet ILSVRC 2012 validation set, with single-crop testing.

| Model | FLOPs | # parameters |Top-1 Acc |FPS (NVIDIA TX2)|
|:-------|:-----:|:-------:|:-------:|:-------:|
| MobileNet | 569 M | 4.2 M | 70.0 | 136 |
| ShuffleNet 2x | 524 M | 5.2 M | 73.7 | 110 |
| Condensenet (C=G=8)  | 274M | 4.0M | 71 | 40 |
|MobileNet v2   | 300 M   | 3.5 M   | 72.0   | 123|
|ShuffleNet v2 1.5x  |  300 M  | 5.2 M  | 72.6  | 164|
|**PeleeNet (our)**  |  508 M  |  2.8 M  |  72.6  |  **240**|
|**PeleeNet v2 (our)** | 621 M  |  4.4 M  |  **73.9**  |  **245**|
