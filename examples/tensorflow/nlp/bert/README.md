Step-by-Step
============

This document is used to list steps of reproducing TensorFlow Intel® Low Precision Optimization Tool tuning zoo result of bert large model on squad v1.1 task.


## Prerequisite

### 1. Installation
```shell
# Install Intel® Low Precision Optimization Tool
pip instal lpot
```
### 2. Install Intel Tensorflow 1.15 up2
Check your python version and use pip install 1.15.0 up2 from links below:
https://storage.googleapis.com/intel-optimized-tensorflow/intel_tensorflow-1.15.0up2-cp36-cp36m-manylinux2010_x86_64.whl                
https://storage.googleapis.com/intel-optimized-tensorflow/intel_tensorflow-1.15.0up2-cp37-cp37m-manylinux2010_x86_64.whl
https://storage.googleapis.com/intel-optimized-tensorflow/intel_tensorflow-1.15.0up2-cp35-cp35m-manylinux2010_x86_64.whl

### 3. Prepare Dataset
wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
unzip wwm_uncased_L-24_H-1024_A-16.zip

wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -P wwm_uncased_L-24_H-1024_A-16

wwm_uncased_L-24_H-1024_A-16. will be your data path

#### Automatic dataset download
Run the `prepare_dataset.sh` script located in `examples/tensorflow/nlp/bert`.

Usage:
```shell
cd examples/tensorflow/nlp/bert
bash prepare_dataset.sh --output_dir=./data
```

Then create the tf_record file, you should config the tf_record path in yaml file.
```shell
python create_tf_record.py --vocab_file=data/vocab.txt --predict_file=data/dev-v1.1.json --output_file=./eval.tf_record
```

### 4. Prepare Pretrained model

#### Manual approach

```shell
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/bert_large_checkpoints.zip
unzip bert_large_checkpoints.zip
```
#### Automatic model download
Run the `prepare_model.sh` script located in `examples/tensorflow/nlp/bert`.

Usage:
```shell
cd examples/tensorflow/nlp/bert
bash prepare_model.sh --output_dir=./model
```

## Prepare frozen pb from checkpoint
  ```shell
  python freeze_estimator_to_pb.py --input_model=./model --output_model=./bert_fp32.pb
  ```
## Run Command
  ```shell
  python tune_squad.py --config=./bert.yaml --input_model=./bert_fp32.pb --output_model=./int8.pb --mode=tune
  ```

Details of enabling Intel® Low Precision Optimization Tool on bert model for Tensorflow.
=========================

This is a tutorial of how to enable bert model with Intel® Low Precision Optimization Tool.
## User Code Analysis
1. User specifies fp32 *model*, calibration dataset *q_dataloader*, evaluation dataset *eval_dataloader* and metric in tuning.metric field of model-specific yaml config file.

2. User specifies fp32 *model*, calibration dataset *q_dataloader* and a custom *eval_func* which encapsulates the evaluation dataset and metric by itself.

For bert, we applied the first one as we  already have built-in dataset and metric for bert squad task. 

### Write Yaml config file
In examples directory, there is a bert.yaml. We could remove most of items and only keep mandatory item for tuning. We also implement a calibration dataloader and have evaluation field for creation of evalation function at internal lpot.

```yaml
model: 
  name: bert
  framework: tensorflow
  inputs: input_file, batch_size
  outputs: IteratorGetNext:3, unstack:0, unstack:1

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
        from lpot.quantization import Quantization
        quantizer = Quantization('./bert.yaml')
        quantizer.model = FLAGS.input_model
        q_model = quantizer()
        q_model.save(FLAGS.output_model)

```
#### Benchmark
```python
        from lpot.experimental import Benchmark
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
The Intel® Low Precision Optimization Tool quantizer() function will return a best quantized model under time constraint.

