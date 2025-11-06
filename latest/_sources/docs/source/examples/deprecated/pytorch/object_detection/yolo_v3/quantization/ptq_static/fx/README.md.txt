Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch YOLO v3 tuning results with Intel® Neural Compressor.


# Prerequisite

### 1. Environmental

```shell
cd examples/pytorch/object_detection/yolo_v3/quantization/ptq/fx
pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
cd data
bash get_coco_dataset.sh
# the data will save to `coco` folder.
```
### 3. Prepare Weights

```bash
cd weights
bash download_weights.sh
# the weights will save to `yolov3.weights` file.
```

# Run

## Tune
```bash
bash run_quant.sh --input_model=weights/yolov3.weights  --dataset_location=coco
```
## Benchmark
```
# performance
bash run_benchmark.sh --input_model=weights/yolov3.weights --dataset_location=coco --mode=performance --int8=true

## accuracy_only
bash run_benchmark.sh --input_model=weights/yolov3.weights --dataset_location=coco --mode=accuracy --int8=true
```

Examples Of Enabling Neural Compressor Auto Tuning On PyTorch YOLOV3
=======================================================

This is a tutorial of how to enable a PyTorch model with Intel® Neural Compressor.




### Code Update

We update test.py like below.

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

