Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch YOLO v3 tuning results with Intel® Low Precision Optimization Tool.

> **Note**
>
> PyTorch quantization implementation in imperative path has limitation on automatically execution.
> It requires to manually add QuantStub and DequantStub for quantizable ops, it also requires to manually do fusion operation.
> ILiT requires users to complete these two manual steps before triggering auto-tuning process.
> For details, please refer to https://pytorch.org/docs/stable/quantization.html

# Prerequisite

### 1. Installation

  ```Shell
  pip install -r requirements.txt
  ```

### 2. Prepare Dataset
  ```
  cd examples/pytorch/object_detection/yolo_v3/data/
  bash get_coco_dataset.sh
  ```

### 3. Prepare Weights
  ```
  cd examples/pytorch/object_detection/yolo_v3/weights/
  bash download_weights.sh
  ```


# Run

  ```Shell
  cd examples/pytorch/object_detection/yolo_v3/
  python test.py --weights_path weights/yolov3.weights -t
  ```

Examples Of Enabling ILiT Auto Tuning On PyTorch YOLOV3
=======================================================

This is a tutorial of how to enable a PyTorch model with Intel® Low Precision Optimization Tool.

# User Code Analysis

Intel® Low Precision Optimization Tool supports three usage as below:

1. User only provide fp32 "model", and configure calibration dataset, evaluation dataset and metric in model-specific yaml config file.
2. User provide fp32 "model", calibration dataset "q_dataloader" and evaluation dataset "eval_dataloader", and configure metric in tuning.metric field of model-specific yaml config file.
3. User specifies fp32 "model", calibration dataset "q_dataloader" and a custom "eval_func" which encapsulates the evaluation dataset and metric by itself.

Here we integrate PyTorch YOLO V3 with Intel® Low Precision Optimization Tool by the third use case for simplicity.

### Write Yaml Config File

In examples directory, there is a template.yaml. We could remove most of items and only keep mandotory item for tuning. 


```
#conf.yaml

framework:
  - name: pytorch

tuning:
    accuracy_criterion:
      - relative: 0.01
    timeout: 0
    random_seed: 9527
```

Here we set accuracy target as tolerating 0.01 relative accuracy loss of baseline. The default tuning strategy is basic strategy. The timeout 0 means unlimited tuning time until accuracy target is met, but the result maybe is not a model of best accuracy and performance.

### Prepare

PyTorch quantization requires two manual steps:

1. Add QuantStub and DeQuantStub for all quantizable ops.
2. Fuse possible patterns, such as Conv + Relu and Conv + BN + Relu.

The related code please refer to examples/pytorch/object_detection/yolo_v3/models.py.

### Code Update

After prepare step is done, we just need update test.py like below.

```
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
from lpot import Quantization
dataset = ListDataset(valid_path, img_size=opt.img_size, augment=False, multiscale=False)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=opt.batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
)
lpot_dataloader = yolo_dataLoader(dataloader)
quantizer = Quantization("./conf.yaml")
q_model = quantizer(model, q_dataloader=lpot_dataloader, eval_func=eval_func)
```

The quantizer() function will return a best quantized model during timeout constrain.
