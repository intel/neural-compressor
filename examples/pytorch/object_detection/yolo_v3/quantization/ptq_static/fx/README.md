Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch YOLO v3 tuning results with Intel® Neural Compressor.

> **Note**
>
> PyTorch quantization implementation in imperative path has limitation on automatically execution.
> It requires to manually add QuantStub and DequantStub for quantizable ops, it also requires to manually do fusion operation.
> Neural Compressor requires users to complete these two manual steps before triggering auto-tuning process.
> For details, please refer to https://pytorch.org/docs/stable/quantization.html

# Prerequisite

### 1. Environmental

```shell
cd examples/pytorch/object_detection/yolo_v3/quantization/ptq/eager
pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
bash get_coco_dataset.sh
```

### 3. Prepare Weights

```bash
bash download_weights.sh
```

# Run

```shell
python test.py --weights_path weights/yolov3.weights -t
```

Examples Of Enabling Neural Compressor Auto Tuning On PyTorch YOLOV3
=======================================================

This is a tutorial of how to enable a PyTorch model with Intel® Neural Compressor.


### Prepare
PyTorch quantization requires two manual steps:

1. Add QuantStub and DeQuantStub for all quantizable ops.
2. Fuse possible patterns, such as Conv + Relu and Conv + BN + Relu.

The related code please refer to examples/pytorch/object_detection/yolo_v3/quantization/ptq/eager/models.py.

### Code Update

After prepare step is done, we just need update test.py like below.

```python
class yolo_dataLoader(object):
    def __init__(self, loader=None, model_type=None, device='cpu'):
        self.loader = loader
        self.device = device
        self.batch_size = loader.batch_size
    def __iter__(self):
        labels = []
        for _, imgs, targets in self.loader:
            # Extract labels
            labels += targets[:, 1].tolist()
            # Rescale target
            targets[:, 2:] = xywh2xyxy(targets[:, 2:])
            targets[:, 2:] *= opt.img_size

            Tensor = torch.FloatTensor
            imgs = Variable(imgs.type(Tensor), requires_grad=False)
            yield imgs, targets
def eval_func(model):
    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
    )
    return AP.mean()
model.eval()
model.fuse_model()
dataset = ListDataset(valid_path, img_size=opt.img_size, augment=False, multiscale=False)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=opt.batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
)
nc_dataloader = yolo_dataLoader(dataloader)
from neural_compressor import quantization
from neural_compressor.config import PostTrainingQuantConfig
conf = PostTrainingQuantConfig()
q_model = quantization.fit(model,
                           conf=conf,
                           eval_func=eval_func,
                           calib_dataloader=nc_dataloader
                          )

```

