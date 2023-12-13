Step-by-Step
============
This document describes the step-by-step instructions for applying post training quantization on Segment Anything Model (SAM) using VOC dataset.

# Prerequisite
## Environment
```shell
# install dependencie
pip install -r ./requirements.txt
# retrieve SAM model codes and pretrained weight
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

# PTQ
PTQ example on Segment Anything Model (SAM) using VOC dataset.

## 1. 
```shell
# start PTQ on SAM
python ptq_sam.py
```