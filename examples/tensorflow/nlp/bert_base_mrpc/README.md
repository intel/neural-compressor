Step-by-Step
============

This document is used to list steps of reproducing TensorFlow Intel® Low Precision Optimization Tool tuning zoo result of bert base model on mrpc task.


## Prerequisite

### 1. Installation
```shell
# Install Intel® Low Precision Optimization Tool
pip install lpot
```
### 2. Install Intel Tensorflow 1.15 up2
Check your python version and use pip install 1.15.0 up2 from links below:
https://storage.googleapis.com/intel-optimized-tensorflow/intel_tensorflow-1.15.0up2-cp36-cp36m-manylinux2010_x86_64.whl                
https://storage.googleapis.com/intel-optimized-tensorflow/intel_tensorflow-1.15.0up2-cp37-cp37m-manylinux2010_x86_64.whl
https://storage.googleapis.com/intel-optimized-tensorflow/intel_tensorflow-1.15.0up2-cp35-cp35m-manylinux2010_x86_64.whl

### 3. Prepare Dataset

#### Automatic dataset download
Run the `prepare_dataset.sh` script located in `examples/tensorflow/nlp/bert_base_mrpc`.

Usage:
```shell
cd examples/tensorflow/nlp/bert_base_mrpc
python prepare_dataset.py --tasks='MRPC' --output_dir=./data
```

### 4. Prepare Pretrained model

#### Automatic model download
Run the `prepare_model.sh` script located in `examples/tensorflow/nlp/bert_base_mrpc`.
NOTICE: This will need you first prepare your dataset as mrpc task need do train for good accuracy.


Usage:
```shell
cd examples/tensorflow/nlp/bert_base_mrpc
bash prepare_model.sh --dataset_location=./data --output_dir=./model
```

## Run Command
Make sure the data and model has been generated successfully, located at ./data and ./model
And your output_model will located at ./output_model like the command below
  ```shell
    python run_classifier.py \
      --task_name=MRPC \
      --data_dir=data/MRPC \
      --vocab_file=model/vocab.txt \
      --bert_config_file=model/bert_config.json \
      --init_checkpoint=model/model.ckpt-343 \
      --max_seq_length=128 \
      --train_batch_size=32 \
      --learning_rate=2e-5 \
      --num_train_epochs=3.0 \
      --output_dir=model \
      --output_model=output_model \
      --config=mrpc.yaml \
      --tune \
  ```

Details of enabling Intel® Low Precision Optimization Tool on bert model for Tensorflow.
=========================

This is a tutorial of how to enable bert model with Intel® Low Precision Optimization Tool.
## User Code Analysis
1. User specifies fp32 *model*, calibration dataset *q_dataloader*, evaluation dataset *eval_dataloader* and metric in tuning.metric field of model-specific yaml config file.

2. User specifies fp32 *model*, calibration dataset *q_dataloader* and a custom *eval_func* which encapsulates the evaluation dataset and metric by itself.

For bert, we applied the first one as we  already have write dataset and metric for bert mrpc task. 

### Write Yaml config file
In examples directory, there is a mrpc.yaml. We could remove most of items and only keep mandatory item for tuning. We also implement a calibration dataloader and have evaluation field for creation of evaluation function at internal lpot.

```yaml
model:
  name: bert
  framework: tensorflow
  inputs: input_file, batch_size
  outputs: loss/Softmax:0, IteratorGetNext:3

evaluation:
  accuracy: {}
  performance:
    iteration: 20
    warmup: 5
    configs:
      num_of_instance: 1
      cores_per_instance: 28 
      kmp_blocktime: 1

quantization:            
  calibration:
    sampling_size: 500
  model_wise:
    weight:
      granularity: per_channel
  op_wise: {
             'loss/MatMul': {
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
      from lpot.experimental import Quantization, common
      quantizer = Quantization(FLAGS.config)
      dataset = Dataset(eval_file, FLAGS.eval_batch_size)
      quantizer.model = common.Model(estimator, input_fn=estimator_input_fn)
      quantizer.calib_dataloader = common.DataLoader(dataset, collate_fn=collate_fn)
      quantizer.eval_dataloader = common.DataLoader(dataset, collate_fn=collate_fn)
      quantizer.metric = common.Metric(metric_cls=Accuracy)
      q_model = quantizer()
      q_model.save(FLAGS.output_model)
```
#### Benchmark
```python
      from lpot.experimental import Benchmark, common
      from lpot.model.model import get_model_type
      evaluator = Benchmark(FLAGS.config)
      dataset = Dataset(eval_file, FLAGS.eval_batch_size)
      evaluator.b_dataloader = common.DataLoader(\
          dataset, batch_size=FLAGS.eval_batch_size, collate_fn=collate_fn)
      model_type = get_model_type(FLAGS.input_model)
      evaluator.metric = common.Metric(metric_cls=Accuracy)
      if model_type == 'frozen_pb':
          evaluator.model = FLAGS.input_model
      else:
          evaluator.model = common.Model(estimator, input_fn=estimator_input_fn)
      evaluator(FLAGS.mode)
```
The Intel® Low Precision Optimization Tool quantizer() function will return a best quantized model under time constraint.

