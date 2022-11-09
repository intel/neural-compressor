Step-by-Step
============

This document is used to list steps of reproducing TensorFlow Intel® Neural Compressor tuning zoo result of bert large model on squad v1.1 task.
This example can run on Intel CPUs and GPUs.


## Prerequisite

### 1. Installation
```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```
### 2. Install Intel Tensorflow
```shell
pip install intel-tensorflow
```

### 3. Install Intel Extension for Tensorflow

#### Quantizing the model on Intel GPU
Intel Extension for Tensorflow is mandatory to be installed for quantizing the model on Intel GPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[gpu]
```
For any more details, please follow the procedure in [install-gpu-drivers](https://github.com/intel-innersource/frameworks.ai.infrastructure.intel-extension-for-tensorflow.intel-extension-for-tensorflow/blob/master/docs/install/install_for_gpu.md#install-gpu-drivers)

#### Quantizing the model on Intel CPU(Experimental)
Intel Extension for Tensorflow for Intel CPUs is experimental currently. It's not mandatory for quantizing the model on Intel CPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[cpu]
```

### 4. Prepare Dataset
```shell
wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
```

```shell
unzip wwm_uncased_L-24_H-1024_A-16.zip
```

```shell
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -P wwm_uncased_L-24_H-1024_A-16
```
wwm_uncased_L-24_H-1024_A-16 folder will be located on your data path.

#### Automatic dataset download
Run the `prepare_dataset.sh` script located in `examples/tensorflow/nlp/bert_large_squad/quantization/ptq`.

Usage:
```shell
cd examples/tensorflow/nlp/bert_large_squad/quantization/ptq
bash prepare_dataset.sh --output_dir=./data
```

Then create the tf_record file and you need to config the tf_record path in yaml file.
```shell
python create_tf_record.py --vocab_file=data/vocab.txt --predict_file=data/dev-v1.1.json --output_file=./eval.tf_record
```

### 5. Prepare Pretrained model

#### Manual approach

```shell
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/bert_large_checkpoints.zip
unzip bert_large_checkpoints.zip
```
#### Automatic model download
Run the `prepare_model.sh` script located in `examples/tensorflow/nlp/bert_large_squad/quantization/ptq`.

Usage:
```shell
cd examples/tensorflow/nlp/bert_large_squad/quantization/ptq
bash prepare_model.sh --output_dir=./model
```

## Prepare frozen pb from checkpoint
  ```shell
  python freeze_estimator_to_pb.py --input_model=./model --output_model=./bert_fp32.pb
  ```
## Run Command
  <b><font color='red'>Please make sure below command should be executed with the same Tensorflow runtime version as above step.</font></b>

  ```shell
  python tune_squad.py --config=./bert.yaml --input_model=./bert_fp32.pb --output_model=./int8.pb --tune
  ```

Now the tool will generate an int8 model with iterator inside the graph if you want the tuned int8 model to be raw input with 3 inputs you can use command like below:

  ```shell
  python tune_squad.py --config=./bert.yaml --input_model=./bert_fp32.pb --output_model=./int8.pb --tune --strip_iterator
  ```


Details of enabling Intel® Neural Compressor on bert model for Tensorflow.
=========================

This is a tutorial of how to enable bert model with Intel® Neural Compressor.
## User Code Analysis
1. User specifies fp32 *model*, calibration dataset *q_dataloader*, evaluation dataset *eval_dataloader* and metric in tuning.metric field of model-specific yaml config file.

2. User specifies fp32 *model*, calibration dataset *q_dataloader* and a custom *eval_func* which encapsulates the evaluation dataset and metric by itself.

For bert, we applied the first one as we  already have built-in dataset and metric for bert squad task. 

### Write Yaml config file
In examples directory, there is a bert.yaml for tuning the model on Intel CPUs. The 'framework' in the yaml is set to 'tensorflow'. If running this example on Intel GPUs, the 'framework' should be set to 'tensorflow_itex' and the device in yaml file should be set to 'gpu'. The bert_itex.yaml is prepared for the GPU case. We could remove most of items and only keep mandatory item for tuning. We also implement a calibration dataloader and have evaluation field for creation of evaluation function at internal neural_compressor.

```yaml
model: 
  name: bert
  framework: tensorflow
  inputs: input_file, batch_size
  outputs: IteratorGetNext:3, unstack:0, unstack:1

device: cpu                                          # optional. default value is cpu, other value is gpu.

evaluation:
  accuracy:
    metric:
      SquadF1:
    dataloader:
      dataset:
        bert:
          root: eval.tf_record
          label_file: dev-v1.1.json
      batch_size: 64
    postprocess:
      transform:
        SquadV1PostTransform:
          label_file: dev-v1.1.json
          vocab_file: vocab.txt
  performance:
    iteration: 50
    configs:
        num_of_instance: 7
        cores_per_instance: 4
    dataloader:
      dataset:
        bert:
          root: /path/to/eval.tf_record
          label_file: /path/to/dev-v1.1.json
      batch_size: 64

quantization:            
  calibration:
    sampling_size: 500
  model_wise:
    weight:
      granularity: per_channel
  op_wise: {
             'MatMul': {
               'activation':  {'dtype': ['fp32']},
               'weight':  {'dtype': ['fp32']},
             }
           }
tuning:
  accuracy_criterion:
    relative:  0.01   
  exit_policy:
    timeout: 0       
    max_trials: 100 
  random_seed: 9527

```
Here we set the input tensor and output tensors name into *inputs* and *outputs* field. In this case we calibrate and quantize the model, and use our calibration dataloader initialized from a 'Dataset' object.

### Code update

After prepare step is done, we add tune and benchmark code to generate quantized model and benchmark.

#### Tune
```python
        from neural_compressor.quantization import Quantization
        quantizer = Quantization('./bert.yaml')
        quantizer.model = FLAGS.input_model
        q_model = quantizer.fit()
        q_model.save(FLAGS.output_model)

```
#### Benchmark
```python
        from neural_compressor.experimental import Benchmark
        evaluator = Benchmark('./bert.yaml')
        evaluator.model = FLAGS.input_model
        results = evaluator()
        for mode, result in results.items():
            acc, batch_size, result_list = result
            latency = np.array(result_list).mean() / batch_size
            print('\n{} mode benchmark result:'.format(mode))
            print('Accuracy is {:.3f}'.format(acc))
            print('Batch size = {}'.format(batch_size))
            print('Latency: {:.3f} ms'.format(latency * 1000))
            print('Throughput: {:.3f} images/sec'.format(1./ latency))
```
The Intel® Neural Compressor quantizer.fit() function will return a best quantized model under time constraint.
