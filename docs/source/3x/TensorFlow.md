TensorFlow
===============


- [TensorFlow](#tensorflow)
  - [Introduction](#introduction)
  - [API for TensorFlow](#api-for-tensorflow)
    - [Support Matrix](#support-matrix)
      - [Quantization Scheme](#quantization-scheme)
      - [Quantization Approaches](#quantization-approaches)
        - [Post Training Static Quantization](#post-training-static-quantization)
        - [Smooth Quantization](#smooth-quantization)
        - [Mixed Precision](#mixed-precison)
      - [Backend and Device](#backend-and-device)

## Introduction

`neural_compressor.tensorflow` provides a integrated API for applying quantization on various TensorFlow model format, such as `pb`, `saved_model` and `keras`. The comprehensive range of supported models includes but not limited to CV models, NLP models, and large language models. 

In terms of ease of use, neural compressor is committed to providing flexible and scalable user interfaces. While `quantize_model` is designed to provide a fast and straightforward quantization experience, the `autotune` offers an advanced option of reducing accuracy loss during quantization.


## API for TensorFlow

Intel(R) Neural Compressor provides `quantize_model` and `autotune` as main interfaces for supported algorithms on TensorFlow framework.


**quantize_model**

The design philosophy of the `quantize_model` interface is easy-of-use. With minimal parameters requirement, including `model`, `quant_config`, `calib_dataloader`, `calib_iteration`, it offers a straightforward choice of quantizing TF model in one-shot.

```python
def quantize_model(
    model: Union[str, tf.keras.Model, BaseModel],
    quant_config: Union[BaseConfig, list],
    calib_dataloader: Callable = None,
    calib_iteration: int = 100,
    calib_func: Callable = None,
):
```
`model` should be a string of the model's location, the object of Keras model or INC TF model wrapper class.

`quant_config` is either the `StaticQuantConfig` object or a list contains `SmoothQuantConfig` and `StaticQuantConfig` to indicate what algorithm should be used and what specific quantization rules should be applied.

`calib_dataloader` is used to load the data samples for calibration phase. In most cases, it could be the partial samples of the evaluation dataset.

`calib_iteration` is used to decide how many iterations the calibration process will be run.

`calib_func` is a substitution for `calib_dataloader` when the built-in calibration function of INC does not work for model inference.


Here is a simple example of using `quantize_model` interface with a dummy calibration dataloader and the default `StaticQuantConfig`:
```python
from neural_compressor.tensorflow import StaticQuantConfig, quantize_model
from neural_compressor.tensorflow.utils import DummyDataset

dataset = DummyDataset(shape=(100, 32, 32, 3), label=True)
calib_dataloader = MyDataLoader(dataset=dataset)
quant_config = StaticQuantConfig()

qmodel = quantize_model("fp32_model.pb", quant_config, calib_dataloader)
```
**autotune**

The `autotune` interface, on the other hand, provides greater flexibility and power. It's particularly useful when accuracy is a critical factor. If the initial quantization doesn't meet the tolerance of accuracy loss, `autotune` will iteratively try quantization rules according to the `tune_config`. 

Just like `quantize_model`, `autotune` requires `model`, `calib_dataloader` and `calib_iteration`. And the `eval_fn`, `eval_args` are used to build evaluation process.



```python
def autotune(
    model: Union[str, tf.keras.Model, BaseModel],
    tune_config: TuningConfig,
    eval_fn: Callable,
    eval_args: Optional[Tuple[Any]] = None,
    calib_dataloader: Callable = None,
    calib_iteration: int = 100,
    calib_func: Callable = None,
) -> Optional[BaseModel]:
```
`model` should be a string of the model's location, the object of Keras model or INC TF model wrapper class.

`tune_config` is the `TuningConfig` object which contains multiple quantization rules.

`eval_fn` is the evaluation function that measures the accuracy of a model.

`eval_args` is the supplemental arguments required by the defined evaluation function.

`calib_dataloader` is used to load the data samples for calibration phase. In most cases, it could be the partial samples of the evaluation dataset.

`calib_iteration` is used to decide how many iterations the calibration process will be run.

`calib_func` is a substitution for `calib_dataloader` when the built-in calibration function of INC does not work for model inference.

Here is a simple example of using `autotune` interface with different quantization rules defined by a list of  `StaticQuantConfig`:
```python
from neural_compressor.common.base_tuning import TuningConfig
from neural_compressor.tensorflow import StaticQuantConfig, autotune

calib_dataloader = MyDataloader(dataset=Dataset())
custom_tune_config = TuningConfig(
    config_set=[
        StaticQuantConfig(weight_sym=True, act_sym=True),
        StaticQuantConfig(weight_sym=False, act_sym=False),
    ]
)
best_model = autotune(
    model="baseline_model",
    tune_config=custom_tune_config,
    eval_fn=eval_acc_fn,
    calib_dataloader=calib_dataloader,
)
```

### Support Matrix

#### Quantization Scheme

| Framework | Backend Library |  Symmetric Quantization | Asymmetric Quantization |
| :-------------- |:---------------:| ---------------:|---------------:|
| TensorFlow    | [oneDNN](https://github.com/oneapi-src/oneDNN) | Activation (int8/uint8), Weight (int8) | - |
| Keras         | [ITEX](https://github.com/intel/intel-extension-for-tensorflow) | Activation (int8/uint8), Weight (int8) | - |


+ Symmetric Quantization
    + int8: scale = 2 * max(abs(rmin), abs(rmax)) / (max(int8) - min(int8) - 1)
    + uint8: scale = max(rmin, rmax) / (max(uint8) - min(uint8))


+ oneDNN: [Lower Numerical Precision Deep Learning Inference and Training](https://software.intel.com/content/www/us/en/develop/articles/lower-numerical-precision-deep-learning-inference-and-training.html)

#### Quantization Approaches

The supported Quantization methods for TensorFlow and Keras are listed below:
<table class="center">
    <thead>
        <tr>
            <th>Types</th>
            <th>Quantization</th>
            <th>Dataset Requirements</th>
            <th>Framework</th>
            <th>Backend</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="2" align="center">Post-Training Static Quantization (PTQ)</td>
            <td rowspan="2" align="center">weights and activations</td>
            <td rowspan="2" align="center">calibration</td>
            <td align="center">Keras</td>
            <td align="center"><a href="https://github.com/intel/intel-extension-for-tensorflow">ITEX</a></td>
        </tr>
        <tr>
            <td align="center">TensorFlow</td>
            <td align="center"><a href="https://github.com/tensorflow/tensorflow">TensorFlow</a>/<a href="https://github.com/Intel-tensorflow/tensorflow">Intel TensorFlow</a></td>
        </tr>
        <tr>
            <td rowspan="1" align="center">Smooth Quantization(SQ)</td>
            <td rowspan="1" align="center">weights</td>
            <td rowspan="1" align="center">calibration</td>
            <td align="center">Tensorflow</td>
            <td align="center"><a href="https://github.com/tensorflow/tensorflow">TensorFlow</a>/<a href="https://github.com/Intel-tensorflow/tensorflow">Intel TensorFlow</a></td>
        </tr>
        <tr>
            <td rowspan="1" align="center">Mixed Precision(MP)</td>
            <td rowspan="1" align="center">weights and activations</td>
            <td rowspan="1" align="center">NA</td>
            <td align="center">Tensorflow</td>
            <td align="center"><a href="https://github.com/tensorflow/tensorflow">TensorFlow</a>/<a href="https://github.com/Intel-tensorflow/tensorflow">Intel TensorFlow</a></td>
        </tr>
    </tbody>
</table>
<br>
<br>

##### Post Training Static Quantization

The min/max range in weights and activations are collected offline on a so-called `calibration` dataset. This dataset should be able to represent the data distribution of those unseen inference dataset. The `calibration` process runs on the original fp32 model and dumps out all the tensor distributions for `Scale` and `ZeroPoint` calculations. Usually preparing 100 samples are enough for calibration.

Refer to the [PTQ Guide](./TF_Quant.md) for detailed information.

##### Smooth Quantization

Smooth Quantization (SQ) is an advanced quantization technique designed to optimize model performance while maintaining high accuracy. Unlike traditional quantization methods that can lead to significant accuracy loss, SQ focuses on a more refined approach by taking a balance between the scale of activations and weights. 

Refer to the [SQ Guide](./TF_SQ.md) for detailed information.

##### Mixed Precision
The Mixed Precision (MP) is enabled with Post Training Static Quantization. Once `BF16` is supported on machine, the matched operators will be automatically converted.


#### Backend and Device
Intel(R) Neural Compressor supports TF GPU with [ITEX-XPU](https://github.com/intel/intel-extension-for-tensorflow). We will automatically run model on GPU by checking if it has been installed.

<table class="center">
    <thead>
        <tr>
            <th>Framework</th>
            <th>Backend</th>
            <th>Backend Library</th>
            <th>Backend Value</th>
            <th>Support Device(cpu as default)</th> 
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="2" align="left">TensorFlow</td>
            <td align="left">TensorFlow</td>
            <td align="left">OneDNN</td>
            <td align="left">"default"</td>
            <td align="left">cpu</td>
        </tr>
        <tr>
            <td align="left">ITEX</td>
            <td align="left">OneDNN</td>
            <td align="left">"itex"</td>
            <td align="left">cpu | gpu</td>
        </tr>  
    </tbody>
</table>
<br>
<br>
