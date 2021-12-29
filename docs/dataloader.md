DataLoader
==========

Deep Learning often encounters large datasets that are memory-consuming. Previously, working with large datasets required loading them into memory all at once. The constant lack of memory resulted in the need for an efficient data generation scheme. This is not only about handling the lack of memory in large datasets, but also about making the process of loading data faster using a multi-processing thread. We call the data generation object a DataLoader.

With the importance of a dataloader, different frameworks can have their own DataLoadermodule. As for Neural Compressor, it needs to calibrate the inputs/outputs of each layer of the model; the framework-specific dataloader has different features and APIs that will make it hard to use them same way in the tool. Another request is that the tool also treat batch size as a tuning parameter  which means the tool can dynamically change the batch size to get the accuracy target. The third reason is for ease of use; a unified DataLoader API can make it easy to config dataloader in a yaml file without any code modification. Considering about all these advantages, the tool has implemented an internal dataloader.

The dataloader takes a dataset as the input parameter and loads data from the dataset when needed.

A dataset is a container which holds all data that can be used by the dataloader, and have the ability to be fetched by index or created as an iterator. One can implement a specific dataset by inheriting from the Dataset class by implementing `__iter__` method or `__getitem__` method, while implementing `__getitem__` method, `__len__` method is recommended.

A dataset uses transform as its data process component. Transform contains three parts, aiming at different parts of the life cycle of data processing:

* preprocessing

* postprocessing

* general

A general transform can be used in both preprocessing and postprocessing; one can also implement a specific transform by inheriting from the Transform class by implementing the `__call__` method. Usually, a dataloader will use the transform for preprocessing and the postprocessing transform is used to give the right processed data to the metric to update. Transforms also compose together to be one and serially implement the transforms.

Transform for preprocessing will be launched in the dataset `__getitem__` or `__next__` method; that means the transform will be used after the dataloader has loaded batched data and before the data given to the model for inference. That helps reduce the memory compared with load and process all data at once. Transform for postprocessing is used in evaluation function of the internal Neural Compressor to process the inference data and the processed data used by metric. 

# How to use it

## Config dataloader in a yaml file

In this case, the dataloader is created after the Quantization object is initialized. As calibrations and evaluations may have different transforms and datasets, you can config different dataloaders in a yaml file.

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

evaluation:                                          # optional. required if user doesn't provide eval_func in neural_compressor.Quantization.
  accuracy:                                          # optional. required if user doesn't provide eval_func in neural_compressor.Quantization.
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

## Create a user-specific dataloader

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

from neural_compressor import Quantization, common
quantizer = Quantization('conf.yaml')
quantizer.model = fp32_model
quantizer.calib_dataloader = calib_data
quantizer.eval_dataloader = calib_data
q_model = quantizer.fit()
```

