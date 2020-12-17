Step-by-Step
============

This document is used to list steps of reproducing TensorFlow Intel® Low Precision Optimization Tool tuning zoo result of bert large model on squad v1.1 task.


## Prerequisite

### 1. Installation
```Shell
# Install Intel® Low Precision Optimization Tool
pip instal lpot
```
### 2. Install Intel Tensorflow 1.15up2
```shell
pip intel-tensorflow==1.15up2
```

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

## Run Command
  ```Shell
  python style_tune.py --output_dir=./result --style_images_paths=./style_images --content_images_paths=./content_images --model_dir=./model --precision=quantized
  ```

Details of enabling Intel® Low Precision Optimization Tool on bert model for Tensorflow.
=========================

This is a tutorial of how to enable bert model with Intel® Low Precision Optimization Tool.
## User Code Analysis
1. User specifies fp32 *model*, calibration dataset *q_dataloader*, evaluation dataset *eval_dataloader* and metric in tuning.metric field of model-specific yaml config file.

2. User specifies fp32 *model*, calibration dataset *q_dataloader* and a custom *eval_func* which encapsulates the evaluation dataset and metric by itself.

For bert, we applied the latter one because we don't have metric for bert squad task. The task is to implement the q_dataloader and implement a *eval_func*. 

### Evaluation Part Adaption

For easy metric the result, we write a SquadF1 metric for squad task accuracy.

```python
def eval_func(graph, iteration=-1):
    print("gonna eval the model....")
    from lpot.adaptor.tf_utils.util import iterator_sess_run
    iter_op = graph.get_operation_by_name('MakeIterator')
    feed_dict = {'input_file:0': eval_writer.filename, \
        'batch_size:0': FLAGS.predict_batch_size}

    all_results = []
    output_tensor = {
        'unique_ids': graph.get_tensor_by_name('IteratorGetNext:3'),
        'start_logits': graph.get_tensor_by_name('unstack:0'),
        'end_logits': graph.get_tensor_by_name('unstack:1')
    }
    config = tf.compat.v1.ConfigProto()
    config.use_per_session_threads = 1
    config.inter_op_parallelism_threads = 1
    config.intra_op_parallelism_threads = 28
    sess = tf.compat.v1.Session(graph=graph, config=config)
    sess.run(iter_op, feed_dict)
    def result_producer(results):
      num_examples = results['unique_ids'].shape[0]
      for i in range(num_examples):
        yield {
            key: value[i]
            for key, value in six.iteritems(results)
        }
    import time
    time_list = []
    idx = 0
    while idx < iteration or iteration == -1:
        try:
            time_start = time.time() 
            results = sess.run(output_tensor)
            duration = time.time() - time_start
            time_list.append(duration)
            for result in result_producer(results):
              unique_id = int(result["unique_ids"])
              start_logits = [float(x) for x in result["start_logits"].flat]
              end_logits = [float(x) for x in result["end_logits"].flat]
              all_results.append(
                  RawResult(
                      unique_id=unique_id,
                      start_logits=start_logits,
                      end_logits=end_logits))
            idx += 1
        except tf.errors.OutOfRangeError:
            print("run out of data, exit....")
            break

    # all_predictions is the preds here, can caculate the accuracy
    label = parse_label_file(FLAGS.label_file)
    warmup = 5
    print('Latency is {}'.format(np.array(time_list[warmup:]).mean() / FLAGS.predict_batch_size))
    print('Batch size is {}'.format(FLAGS.predict_batch_size))
    # only calculate accuracy when running out all predictions
    if iteration == -1:
        squad_transform = SquadV1PostTransform(eval_examples, eval_features,
                          FLAGS.n_best_size, FLAGS.max_answer_length,
                          FLAGS.do_lower_case)

        preds, label = squad_transform((all_results, label))
        f1 = SquadF1()
        f1.update(preds, label)
        print('accuracy is F1: {}'.format(f1.result()))
        return f1.result()

```
We also right postprocess Transform to postprocess the predistion of the output, it's named *SquadV1PostTransform*, and metric for squad task named *SquadF1*, after these preparation, we can get the accuracy F1 from squad task

### Write Yaml config file
In examples directory, there is a bert.yaml. We could remove most of items and only keep mandatory item for tuning. We also implement a calibration dataloader

```yaml
model: 
  name: bert
  framework: tensorflow
  inputs: input_file, batch_size
  outputs: IteratorGetNext:3, unstack:0, unstack:1

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

After prepare step is done, we add tune code to generate quantized model.

```python
    from lpot.quantization import Quantization
    quantizer = Quantization('./bert.yaml')
    # we should change the dataloader to provide only once the file name and batch_size
    dataset = Dataset(eval_writer.filename, FLAGS.predict_batch_size)
    dataloader = quantizer.dataloader(dataset, collate_fn=collate_fn)
    q_model = quantizer(graph, q_dataloader=dataloader, eval_func=eval_func)
    # q_model = quantizer(graph, q_dataloader=dataloader)
    from lpot.adaptor.tf_utils.util import write_graph
    write_graph(q_model.as_graph_def(), FLAGS.output_model)
```

The Intel® Low Precision Optimization Tool quantizer() function will return a best quantized model under time constraint.
