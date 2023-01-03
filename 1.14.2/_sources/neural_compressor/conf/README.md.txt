Introduction
=========================================

User can specify yaml configuration file to control the entire tuning behavior.

Take peleenet model as an example, you will see many repeated but similar items in quantization.op_wise.

```yaml
quantization:
  calibration:
    sampling_size: 256
    dataloader:
      batch_size: 256
      dataset:
        ImageFolder:
          root: /path/to/calibration/dataset
      transform:
        RandomResizedCrop:
          size: 224
        RandomHorizontalFlip:
        ToTensor:
        Normalize:
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
  op_wise: {
             'module.features.stemblock.f_cat': {
               'weight':  {'dtype': ['fp32']},
             },
             'module.features.denseblock1.denselayer1.f_cat': {
               'weight':  {'dtype': ['fp32']},
             },
             'module.features.denseblock1.denselayer2.f_cat': {
               'weight':  {'dtype': ['fp32']},
             },
             'module.features.denseblock1.denselayer3.f_cat': {
               'weight':  {'dtype': ['fp32']},
             },
             'module.features.denseblock2.denselayer1.f_cat': {
               'weight':  {'dtype': ['fp32']},
             },
             'module.features.denseblock2.denselayer2.f_cat': {
               'weight':  {'dtype': ['fp32']},
             },
             'module.features.denseblock2.denselayer3.f_cat': {
               'weight':  {'dtype': ['fp32']},
             },
             'module.features.denseblock2.denselayer4.f_cat': {
               'weight':  {'dtype': ['fp32']},
             },
             'module.features.denseblock3.denselayer1.f_cat': {
               'weight':  {'dtype': ['fp32']},
             },
             'module.features.denseblock3.denselayer2.f_cat': {
               'weight':  {'dtype': ['fp32']},
             },
             'module.features.denseblock3.denselayer3.f_cat': {
               'weight':  {'dtype': ['fp32']},
             },
             'module.features.denseblock3.denselayer4.f_cat': {
               'weight':  {'dtype': ['fp32']},
             },
             'module.features.denseblock3.denselayer5.f_cat': {
               'weight':  {'dtype': ['fp32']},
             },
             'module.features.denseblock3.denselayer6.f_cat': {
               'weight':  {'dtype': ['fp32']},
             },
             'module.features.denseblock3.denselayer7.f_cat': {
               'weight':  {'dtype': ['fp32']},
             },
             'module.features.denseblock3.denselayer8.f_cat': {
               'weight':  {'dtype': ['fp32']},
             },
             'module.features.denseblock4.denselayer1.f_cat': {
               'weight':  {'dtype': ['fp32']},
             },
             'module.features.denseblock4.denselayer2.f_cat': {
               'weight':  {'dtype': ['fp32']},
             },
             'module.features.denseblock4.denselayer3.f_cat': {
               'weight':  {'dtype': ['fp32']},
             },
             'module.features.denseblock4.denselayer4.f_cat': {
               'weight':  {'dtype': ['fp32']},
             },
             'module.features.denseblock4.denselayer5.f_cat': {
               'weight':  {'dtype': ['fp32']},
             },
             'module.features.denseblock4.denselayer6.f_cat': {
               'weight':  {'dtype': ['fp32']},
             },
           }
```

Because config module supports parsing regular expression, so the above content can be simplified to:

```yaml
quantization:
  calibration:
    sampling_size: 256
    dataloader:
      batch_size: 256
      dataset:
        ImageFolder:
          root: /path/to/calibration/dataset
      transform:
        RandomResizedCrop:
          size: 224
        RandomHorizontalFlip:
        ToTensor:
        Normalize:
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
  op_wise: {
             'module.features.stemblock.f_cat': {
               'weight':  {'dtype': ['fp32']},
             },
             'module.features.denseblock1.denselayer[1-3].f_cat': {
               'weight':  {'dtype': ['fp32']},
             },
             'module.features.denseblock1.denselayer[1-4].f_cat': {
               'weight':  {'dtype': ['fp32']},
             },
             'module.features.denseblock1.denselayer[1-8].f_cat': {
               'weight':  {'dtype': ['fp32']},
             },
             'module.features.denseblock1.denselayer[1-6].f_cat': {
               'weight':  {'dtype': ['fp32']},
             }
           }
```

> Note that you can use other standard regular expression.
