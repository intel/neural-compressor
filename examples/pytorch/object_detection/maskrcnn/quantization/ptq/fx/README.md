Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch MASK_RCNN tuning results with Intel® Neural Compressor.

# Prerequisite

### 1. Installation

PyTorch >=1.8 and <=1.11 version is needed with pytorch_fx backend.

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

# Saving and loading model:

* Saving model:
  After tuning with Neural Compressor, we can get neural_compressor.model:

```
from neural_compressor.experimental import Quantization, common
quantizer = Quantization("./conf.yaml")
quantizer.model = common.Model(model)
nc_model = quantizer.fit()
```

Here, nc_model is Neural Compressor model class, so it has "save" API:

```python
nc_model.save("Path_to_save_configure_file")
```

* loading model:

```python
from neural_compressor.utils.pytorch import load
quantized_model = load(
    os.path.join(Path, 'best_configure.yaml'),
    os.path.join(Path, 'best_model_weights.pt'),
    fp32_model,
    dataloader=your_dataloader)
```

Please refer to [Sample code](./pytorch/tools/test_net.py)

Examples of enabling Neural Compressor auto tuning on PyTorch ResNet
=======================================================

This is a tutorial of how to enable a PyTorch classification model with Neural Compressor.

# User Code Analysis

Neural Compressor supports three usages:

1. User only provide fp32 "model", and configure calibration dataset, evaluation dataset and metric in model-specific yaml config file.
2. User provide fp32 "model", calibration dataset "q_dataloader" and evaluation dataset "eval_dataloader", and configure metric in tuning.metric field of model-specific yaml config file.
3. User specifies fp32 "model", calibration dataset "q_dataloader" and a custom "eval_func" which encapsulates the evaluation dataset and metric by itself.

Here we integrate PyTorch maskrcnn with Intel® Neural Compressor by the third use case for simplicity.

### Write Yaml Config File

In examples directory, there is a template.yaml. We could remove most of items and only keep mandatory item for tuning.

```yaml
#conf.yaml

model:                                         
    name: maskrcnn
    framework: pytorch 

quantization:                                 
    approach: post_training_static_quant

tuning:
    accuracy_criterion:
        relative:  0.01                            
    exit_policy:
        timeout: 0                                
        max_trials: 600
    random_seed: 9527   
```

Here we set accuracy target as tolerating 0.01 relative accuracy loss of baseline. The default tuning strategy is basic strategy. The timeout 0 means unlimited time for a tuning config meet accuracy target.

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
        from neural_compressor.experimental import Quantization, common
        model.eval()
        quantizer = Quantization("./conf.yaml")
        prepare_custom_config_dict = {"non_traceable_module_class": [
           AnchorGenerator, RPNPostProcessor, Pooler, PostProcessor, MaskRCNNFPNFeatureExtractor,
           MaskPostProcessor, FPN, RPNHead
        ]}
        data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed, is_calib=True)
        cal_dataloader = MASKRCNN_DataLoader(data_loaders_val)
        quantizer.model = common.Model(model, kwargs=prepare_custom_config_dict)
        quantizer.calib_dataloader = cal_dataloader
        quantizer.eval_func = eval_func
        q_model = quantizer.fit()
        q_model.save(args.tuned_checkpoint)
        return
```

The quantizer.fit() function will return a best quantized model during timeout constrain.
