DataLoader and Metric
=========================================

Deep Learning has been encountering larger and larger datasets which are so memory consuming.Before, working with large datasets requires loading them into memory all at once. It is impossible due to the lack of memory, we must figure out an efficient data generation scheme. This is not only about handle the lack of memory in large datasets, also about make the process of loading data faster enough using multi processing/thread.

As to evaluate the performence of a specific model, we should have a general metric to measure the perfomance of different model.

Intel® Low Precision Optimization Tool has implemented an internal DataLoader and Metric, it have several advantages:
  a. an iterator for iLit tuning dynamic iterations across frameworks;
  b. dynamic batch size tuning
  c. help iLit to have an end to end tuning capability by easy configuration

# DataLoader & Metric internal design logic

<div align="left">
  <img src="imgs/dataloader.png" width="700px" />
</div>

<div align="left">
  <img src="imgs/metric.png" width="700px" />
</div>

Both DataLoader and Metric use Transform as its data process component, Transform contains 3 different part, aimng at different part of the life cycle of data processing, it is:
  a. preprocessing
  b. postprocessing
  c. general
general Transform can be used in both preprocessing and postprocessing, one can also implement a specific transform by inhereting from class Transform with implementing __call__ method. Usually, DataLoader will use Transform for preprocessing and Metric will use Transform for postprocessing. Transforms also support to compose together to be one and serially implement the transforms.

Transform will be launched in Dataset __getitem__ or __next__ method, that means only when dataloader will load batched data the transform will be implemented. That helps reduce the memory compared with load and process all data at once. 

Dataset is a container can be holding all data that should be used, and have the ability to be fetched by index or created as an iterator.one can implement a specific Dataset by inhereting from class Dataset with implementing __iter__ method or __getitem__ method, while implementing __getitem__ method, __len__ method is recommended.

DataLoader will take dataset as input parameter and load data from dataset when needed. Intel® Low Precision Optimization Tool support dynamic batching, you can easilly use .batch(batch_size). 

# How to use it

## config dataloader and metric in yaml file
In this case dataloader and metric will created when the tool's tuner initilized. As calibration and evaluation may have different Transform and dataset, you can config different dataloader in yaml file.
eg:

tuning:
    metric:
      topk: 1
calibration:
    iterations: 10
    dataloader:
      batch_size: 30
      dataset:
        - type: "ImageFolder"
        - root: /Path/to/imagenet/img/train
      transform:
        RandomResizedCrop:
          - size: 224
        RandomHorizontalFlip:
        ToTensor:
        Normalize:
          - mean: [0.485, 0.456, 0.406]
          - std: [0.229, 0.224, 0.225]

evaluation:
  dataloader:
    batch_size: 30
    dataset:
      - type: "ImageFolder"
      - root: /Path/to/imagenet/img/val
    transform:
      Resize:
        - size: 256
      CenterCrop:
        - size: 224
      ToTensor:
      Normalize:
        - mean: [0.485, 0.456, 0.406]
        - std: [0.229, 0.224, 0.225]

## create Intel® Low Precision Optimization Tool internal dataloader and metric and pass to tuner
eval_dataset = tuner.dataset('bert', dataset=eval_dataset, task=eval_task)
test_dataloader = tuner.dataloader(eval_dataset, batch_size=args.eval_batch_size)
tuner.tune(model, test_dataloader, eval_func=eval_func_for_ilit)

## use user specific dataloader and metric

calib_data = mx.io.ImageRecordIter(path_imgrec=dataset,label_width=1,preprocess_threads=data_nthreads,batch_size=batch_size,data_shape=data_shape,label_name=label_name,rand_crop=False,rand_mirror=False,shuffle=args.shuffle_dataset,shuffle_chunk_seed=args.shuffle_chunk_seed,seed=args.shuffle_seed,dtype=data_layer_type,ctx=args.ctx,**combine_mean_std)

ilit_model = cnn_tuner.tune(fp32_model, q_dataloader=calib_data, eval_dataloader=calib_data)
