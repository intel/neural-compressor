Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch MASK_RCNN tuning results with Intel® Neural Compressor.

# Prerequisite

### 1. Environment

PyTorch >=1.8 and <=1.11 version is needed.

```shell
cd examples/pytorch/object_detection/maskrcnn/quantization/ptq/fx
pip install -r requirements.txt
bash install.sh
```

### 2. Prepare Dataset

You can download COCO2017 dataset use script file:

```
source download_dataset.sh
```

Or you can download COCO2017 dataset to your local path, then link it to pytorch/datasets/coco

```bash
ln -s /path/of/COCO2017/annotations pytorch/datasets/coco/annotations
ln -s /path/of/COCO2017/train2017 pytorch/datasets/coco/train2017
ln -s /path/of/COCO2017/val2017 pytorch/datasets/coco/val2017
```

### Prepare weights

You can download weights with script file:

```bash
bash download_weights.sh
```

Or you else can link your weights to pytorch folder

```bash
ln -s /path/of/weights pytorch/e2e_mask_rcnn_R_50_FPN_1x.pth
```

# Run

```shell
bash run_tuning.sh --output_model=/path/to/tuned_checkpoint
```


Please refer to [Sample code](./pytorch/tools/test_net.py)

Examples of enabling Neural Compressor auto tuning on PyTorch ResNet
=======================================================

This is a tutorial of how to enable a PyTorch classification model with Neural Compressor.

### Prepare
The related code please refer to examples/pytorch/object_detection/maskrcnn/quantization/ptq/fx/pytorch/tools/test_net.py.

### Code Update

After prepare step is done, we just need update main.py like below.

```python
class MASKRCNN_DataLoader(object):
        def __init__(self, loaders=None):
            self.loaders = loaders
            self.batch_size = 1

        def __iter__(self):
            for loader in self.loaders:
                for batch in loader:
                    images, targets, image_ids = batch
                    yield images, targets

def eval_func(q_model):
        output_folders = [None] * len(cfg.DATASETS.TEST)
        dataset_names = cfg.DATASETS.TEST
        if cfg.OUTPUT_DIR:
            for idx, dataset_name in enumerate(dataset_names):
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
                mkdir(output_folder)
                output_folders[idx] = output_folder
        data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
        for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
            results, _ = inference(
                q_model,
                data_loader_val,
                dataset_name=dataset_name,
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder,
            )
            synchronize()
        print('Batch size = %d' % cfg.SOLVER.IMS_PER_BATCH)
        return results.results['bbox']['AP']

if args.tune:
        from neural_compressor import quantization
        from neural_compressor.config import PostTrainingQuantConfig
        from neural_compressor.model import Model
        conf = PostTrainingQuantConfig()
        prepare_custom_config_dict = {"non_traceable_module_class": [
           AnchorGenerator, RPNPostProcessor, Pooler, PostProcessor, MaskRCNNFPNFeatureExtractor,
           MaskPostProcessor, FPN, RPNHead
        ]}
        data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed, is_calib=True)
        cal_dataloader = MASKRCNN_DataLoader(data_loaders_val)
        new_model = Model(model, kwargs=prepare_custom_config_dict)
        q_model = quantization.fit(model,
                                    conf=conf,
                                    eval_func=eval_func,
                                    calib_dataloader=cal_dataloader)
        q_model.save(args.tuned_checkpoint)
        return
```

