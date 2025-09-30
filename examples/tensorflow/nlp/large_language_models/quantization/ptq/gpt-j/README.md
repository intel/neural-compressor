Step-by-Step
============

This document is used to list steps of reproducing TensorFlow Intel® Neural Compressor smooth quantization of language models gpt-j-6B.

# Prerequisite

## 1. Environment

### Installation
```shell
# Install Intel® Neural Compressor
pip install neural-compressor
pip install -r requirements.txt
```

## 2. Prepare Pretrained model
Run the follow script to download gpt-j-6B saved_model to ```./gpt-j-6B```: 
 ```
bash prepare_model.sh
 ```

## 3. Install TensorFlow 2.11.dev202242
Build a TensorFlow pip package from [intel-tensorflow spr_ww42 branch](https://github.com/Intel-tensorflow/tensorflow/tree/spr_ww42) and install it. How to build a TensorFlow pip package from source please refer to this [tutorial](https://www.tensorflow.org/install/source).

The performance of int8 gpt-j-6B would be better once intel-tensorflow for gnr is released.

## 4. Prepare Dataset
The dataset will be automatically loaded.

# Run

## Smooth Quantization

```shell
bash run_quant.sh --input_model=<FP32_MODEL_PATH> --output_model=<INT8_MODEL_PATH>
```

## Benchmark

### Evaluate Performance

```shell
bash run_benchmark.sh --input_model=<MODEL_PATH> --mode=benchmark
```

### Evaluate Accuracy

```shell
bash run_benchmark.sh --input_model=<MODEL_PATH> --mode=accuracy
```


Details of enabling Intel® Neural Compressor on gpt-j-6B for TensorFlow
=========================

This is a tutorial of how to enable gpt-j-6B model with Intel® Neural Compressor.
## User Code Analysis

User specifies fp32 *model*, calibration dataloader *q_dataloader* and a custom *eval_func* which encapsulates the evaluation dataloader and metric by itself.

### calib_dataloader Part Adaption
Below dataloader class uses generator function to provide the model with input.

```python
class MyDataloader:
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.length = math.ceil(len(dataset) / self.batch_size)

    def generate_data(self, data, pad_token_id=50256):
        input_ids = tf.convert_to_tensor([data[:-1]], dtype=tf.int32)
        cur_len = len(data)-1
        input_ids_padding = tf.ones((self.batch_size, 1), dtype=tf.int32) * (pad_token_id or 0)
        generated = tf.concat([input_ids, input_ids_padding], axis=-1)
        model_kwargs = {'attention_mask': prepare_attention_mask_for_generation(input_ids)}
        if model_kwargs.get("past_key_values") is None:
            input_ids = generated[:, :cur_len]
        else:
            input_ids = tf.expand_dims(generated[:, cur_len - 1], -1)
        return model_kwargs['attention_mask'], input_ids
    
    def __iter__(self):
        labels = None
        for _, data in enumerate(self.dataset):
            cur_input = self.generate_data(data)
            yield (cur_input, labels)

    def __len__(self):
        return self.length
```


### Code Update
After prepare step is done, we add the code for quantization tuning to generate quantized model.

Firstly, let's load a INC inner class model from the path of gpt-j-6B saved_model.
```python
    from neural_compressor import Model
    model = Model(run_args.input_model, modelType='llm_saved_model')
```

#### Tune

To apply quantization, the function that maps names from AutoTrackable variables to graph nodes must be defined to match names of nodes in different format.
```python
    def weight_name_mapping(name):
        """The function that maps name from AutoTrackable variables to graph nodes"""
        name = name.replace('tfgptj_for_causal_lm', 'StatefulPartitionedCall')
        name = name.replace('kernel:0', 'Tensordot/ReadVariableOp')
        return name
```

Please use the recipe to set smooth quantization.
```python
    from neural_compressor.tensorflow import StaticQuantConfig, SmoothQuantConfig, autotune
    from neural_compressor.tensorflow.quantization import TuningConfig
    from neural_compressor.tensorflow.utils import BaseDataLoader

    calib_dataloader = MyDataloader(mydata, batch_size=run_args.batch_size)  
    quant_config = [SmoothQuantConfig(alpha=0.52705), StaticQuantConfig(act_dtype="int8", weight_dtype="int8")]
    tune_config = TuningConfig(config_set=quant_config, max_trials=1)
    model.weight_name_mapping = weight_name_mapping
    q_model = autotune(model, 
                        tune_config, 
                        eval_fn=evaluate,
                        calib_dataloader=calib_dataloader)
    q_model.save(run_args.output_model)
```
#### Benchmark
```python
    if run_args.mode == "performance":
        evaluate(model.model)
    elif run_args.mode == "accuracy":
        acc_result = evaluate(model.model)
        print("Batch size = %d" % run_args.batch_size)
        print("Accuracy: %.5f" % acc_result)
```

The Intel® Neural Compressor quantization.fit() function will return a best quantized model under time constraint.