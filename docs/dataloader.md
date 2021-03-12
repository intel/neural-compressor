DataLoader
=========================================

Deep Learning has been encountering larger and larger datasets which are so memory consuming. Before, working with large datasets requires loading them into memory all at once. It is impossible due to the lack of memory, we must figure out an efficient data generation scheme. This is not only about handle the lack of memory in large datasets, also about make the process of loading data faster enough using multi processing/thread. We call the data generation object as 'DataLoader'.

With the importance of DataLoader, different framework have their own DataLoadermodule, as for Intel® Low Precision Optimization Tool, it needs to calibrate the inputs/outputs of each layer of the model, framework specific DataLoader has different features and API that will make it hard to use them same way in the tool. Another request is, the tool also treat batch size as a tuning parameter, that means the tool can dynamically change the batch size to get accuracy target. The third reason is for easy of use, an unified DataLoader API can make it easy to config dataloader in yaml file without any code modification. Considering about all these advantages the tool has implemented an internal DataLoader.

DataLoader takes dataset as input parameter and loads data from dataset when needed.

Dataset is a container which holds all data that should be used by dataloader, and have the ability to be fetched by index or created as an iterator. One can implement a specific Dataset by inhereting from class Dataset with implementing `__iter__` method or `__getitem__` method, while implementing `__getitem__` method, `__len__` method is recommended.

Dataset use Transform as its data process component, Transform contains 3 different part, aimng at different part of the life cycle of data processing, it is:

  1. preprocessing

  2. postprocessing

  3. general

General Transform can be used in both preprocessing and postprocessing, one can also implement a specific transform by inheriting from class Transform with implementing `__call__` method. Usually, DataLoader will use Transform for preprocessing and postprocessing transform is used to give right processed data to metric to update. Transforms also support to compose together to be one and serially implement the transforms.

Transform for preprocessing will be launched in Dataset `__getitem__` or `__next__` method, that means transform is used after dataloader has loaded batched data  and before the data given to model for inference. That helps reduce the memory compared with load and process all data at once. Transform for postprocessing is used in evaluation function of internal lpot to process the inferenced data and the processed data is used by metric. 

# How to use it

## Config dataloader in yaml file
In this case dataloader will created after the Quantization object initialized. As calibration and evaluation may have different Transform and dataset, you can config different dataloader in yaml file.

```yaml
quantization:                                        # optional. tuning constraints on model-wise for advance user to reduce tuning space.
  calibration:
    sampling_size: 300                               # optional. default value is 100 samples. used to set how many samples in calibration dataset are used.
    dataloader:
      dataset:
        ImageFolder:
          root: /path/to/calibration/dataset
      transform:
        RandomResizedCrop:
          size: 224
        RandomHorizontalFlip: {}
        ToTensor: {}
        Normalize:
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]

evaluation:                                          # optional. required if user doesn't provide eval_func in lpot.Quantization.
  accuracy:                                          # optional. required if user doesn't provide eval_func in lpot.Quantization.
    metric:
      topk: 1 
    dataloader:
      batch_size: 30
      dataset:
        ImageFolder:
          root: /path/to/evaluation/dataset
      transform:
        Resize:
          size: 256
        CenterCrop:
          size: 224
        ToTensor: {}
        Normalize:
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
  performance:                                       # optional. used to benchmark performance of passing model.
    configs:
      cores_per_instance: 4
      num_of_instance: 7
    dataloader:
      batch_size: 1
      dataset:
        ImageFolder:
          root: /path/to/evaluation/dataset
      transform:
        Resize:
          size: 256
        CenterCrop:
          size: 224
        ToTensor: {}
        Normalize:
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
```

## Create user specific dataloader

```python
calib_data = mx.io.ImageRecordIter(path_imgrec=dataset,
                                   label_width=1,
                                   preprocess_threads=data_nthreads,
                                   batch_size=batch_size,
                                   data_shape=data_shape,
                                   label_name=label_name,
                                   rand_crop=False,
                                   rand_mirror=False,
                                   shuffle=args.shuffle_dataset,
                                   shuffle_chunk_seed=args.shuffle_chunk_seed,
                                   seed=args.shuffle_seed,
                                   dtype=data_layer_type,
                                   ctx=args.ctx,
                                   **combine_mean_std)

from lpot import Quantization, common
quantizer = Quantization('conf.yaml')
quantizer.model = common.Model(fp32_model)
quantizer.calib_dataloader = calib_data
quantizer.eval_dataloader = calib_data
q_model = quantizer()
```

