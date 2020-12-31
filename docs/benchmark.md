Benchmarking
===============

Benchmakring measuring the model performance with the objective settings, user can get the performance of the models between float32 model and quantized low precision model in same scenarios that they configured in yaml. Benchmarking is always used after a quantization process.

# how to use it
## config evaluation filed in yaml file
'''
evaluation:                                          # optional. required if user doesn't provide eval_func in lpot.Quantization.
  accuracy:                                          # optional. required if user doesn't provide eval_func in lpot.Quantization.
    metric:
      topk: 1                                        # built-in metrics are topk, map, f1, allow user to register new metric.
    dataloader:
      batch_size: 30
      dataset:
        ImageFolder:
          root: /path/to/evaluation/dataset          # NOTE: modify to evaluation dataset location if needed
      transform:
        Resize:
          size: 256
        CenterCrop:
          size: 224
        ToTensor:
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
          root: /path/to/evaluation/dataset          # NOTE: modify to evaluation dataset location if needed
      transform:
        Resize:
          size: 256
        CenterCrop:
          size: 224
        ToTensor:
        Normalize:
          mean: [0.485, 0.456, 0.406]
'''

in this example config you can see there is 2 sub-fields named 'accuracy' and 'performance', benchmark module will get the accuracy and performance of the model. User can also remove the performance field to only get accuracy of the model or the opposite. It's flexible to configure the benchmark you want.
## use user specific dataloader to run benchmark
In this case, you should config your dataloader and lpot will construct an evaluation function to run the benchmarking.
'''python
dataset = Dataset() #  dataset class that implement __getitem__ method or __iter__ method
from lpot import Benchmark
evaluator = Benchmark(config.yaml)
evaluator.dataloader(dataset, batch_size=batch_size)
results = evaluator(model=input_model)

'''

###Examples
View [Benchamrk example of Tensorflow image recognition models](../examples/tensorflow/image_recognition/run_benchmarking.sh).

