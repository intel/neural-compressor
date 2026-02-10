Keras ViT Image Classifier model quantization

============

This document describes quantization of Keras ViT models using Neural Compressor on Intel® Xeon® processors.


## 1. Create Environment
It is worth conducting experiments in a separate environment. For example, you can use the conda environment from [conda-forge](https://github.com/conda-forge/miniforge). The binary for your environment could be found here: [miniforge](https://github.com/conda-forge/miniforge/releases/latest)  

To see performance improvements from quantization, you have to enable some JAX/XLA features by setting an environment variable:
```bash
export XLA_FLAGS="\
    --xla_cpu_experimental_onednn_custom_call=true --xla_cpu_use_onednn=false \
    --xla_cpu_experimental_ynn_fusion_type=invalid --xla_cpu_use_xnnpack=false \
    --xla_backend_extra_options=xla_cpu_disable_new_fusion_emitter"
```
Without this flag, quantized model works in fake quantization mode (rounding tensors to a given fp8 format but later making calculations in 32-bit floating point format).


## 2. Install modules

```bash
pip install -r requirements.txt
```

## 3. Download model

The model can be downloaded using keras-hub preset loader:

```python
from keras_hub.models import ViTImageClassifier
import keras

model = ViTImageClassifier.from_preset("vit_base_patch16_224_imagenet")

keras.models.save_model(model, "/path/to/saved/model.keras")
```

## 4. Quantize model

To quantize the model you have to make 3 steps:

1. Load the original model:

```python
import keras

vit_orig = keras.models.load_model("/path/to/original/model.keras")
```

2. Calibrate the model using a dataset similar to the one that will be used later. In our example - we use just one image. We can choose which floating point format will be used in quantized model.

```python
import tensorflow as tf


def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224], method=tf.image.ResizeMethod.BILINEAR)
    return tf.cast(img, tf.uint8)


img = load_image("./colva_beach_sq.jpg")
img = tf.expand_dims(img, axis=0)

from neural_compressor.jax.quantization.config import StaticQuantConfig

config = StaticQuantConfig(weight_dtype="fp8_e4m3", activation_dtype="fp8_e4m3")


def calib_function(model):
    model.predict(img)


from neural_compressor.jax import quantize_model

vit_quantized = quantize_model(vit_orig, config, calib_function)
```

3. Use the quantized model

```python
output = vit_quantized.predict(img)

from keras.applications.imagenet_utils import decode_predictions


def print_predictions(preds):
    preds = decode_predictions(preds, top=4)
    for i, sample_preds in enumerate(preds):
        print(f"Predictions for sample {i}:")
        for k, pred in enumerate(sample_preds):
            _, class_name, score = pred
            print(f"    top-{k+1}: class={class_name}, score={score:.4f}")


print("\nOutput after quantization:")
print_predictions(output)
```

You can run this by running prepared [quantization.py](quantization.py) example.
```bash
python quantization.py -m /path/to/original/model.keras
```
When we run the example, we can notice slightly different top-4 label probabilities. For example:

```bash
Output before quantization:
Predictions for sample 0:
    top-1: class=seashore, score=12.2238
    top-2: class=sandbar, score=8.7158
    top-3: class=lakeside, score=6.5203
    top-4: class=promontory, score=5.9983

Output after quantization:
Predictions for sample 0:
    top-1: class=seashore, score=12.0924
    top-2: class=sandbar, score=8.6914
    top-3: class=lakeside, score=6.3961
    top-4: class=promontory, score=5.9991
```

## 5. Save and load quantized model

Calibration costs time, so we can calibrate once on representative data sets and later reuse it many times. To achieve it saving model functionality is supported.
You can run [prepare_static.py](prepare_static.py) script:
```bash
export XLA_FLAGS="\
    --xla_cpu_experimental_onednn_custom_call=true --xla_cpu_use_onednn=false \
    --xla_cpu_experimental_ynn_fusion_type=invalid --xla_cpu_use_xnnpack=false \
    --xla_backend_extra_options=xla_cpu_disable_new_fusion_emitter"
python prepare_static.py -m /path/to/original/model.keras -q /path/to/quantized/model.keras -p fp8_e4m3
```
or, if default parameters work for you, just:
```bash
export XLA_FLAGS="\
    --xla_cpu_experimental_onednn_custom_call=true --xla_cpu_use_onednn=false \
    --xla_cpu_experimental_ynn_fusion_type=invalid --xla_cpu_use_xnnpack=false \
    --xla_backend_extra_options=xla_cpu_disable_new_fusion_emitter"
python prepare_static.py
```

After this step we have saved model stored in the /path/to/quantized/model.keras file. You can load and use it with [use_static.py](use_static.py)

```python
import neural_compressor.jax.quantization

vit_quantized = keras.models.load_model("/path/to/quantized/model.keras")
output = vit_quantized.predict(img)
```

You can notice that even though the script does not directly use neural_compressor, it requires importing neural_compressor.jax.quantization to load the quantized model.

## 6. Some debug

If you are interested how your model looks like after quantization you can set environment variable:
export LOGLEVEL=DEBUG
and then use print_model() function in your script. To see how it works just uncomment lines 39 and 64 in [quantization.py](quantization.py)

Part of the vit model could look like:

```
1970-01-01 03:34:46 [DEBUG][utility.py:169] ------------------------- internal representation:
1970-01-01 03:34:46 [DEBUG][utility.py:185] ViTImageClassifier                                                                           
1970-01-01 03:34:46 [DEBUG][utility.py:185] InputLayer                .images                                                            
1970-01-01 03:34:46 [DEBUG][utility.py:185] ViTBackbone               .vi_t_backbone                                                     
1970-01-01 03:34:46 [DEBUG][utility.py:185] InputLayer                .vi_t_backbone.images                                              
1970-01-01 03:34:46 [DEBUG][utility.py:185] ViTPatchingAndEmbedding   .vi_t_backbone.vit_patching_and_embedding                          
1970-01-01 03:34:46 [DEBUG][utility.py:185] Conv2D                    .vi_t_backbone.vit_patching_and_embedding.patch_embedding          
1970-01-01 03:34:46 [DEBUG][utility.py:185] Embedding                 .vi_t_backbone.vit_patching_and_embedding.position_embedding       
1970-01-01 03:34:46 [DEBUG][utility.py:185] ViTEncoder                .vi_t_backbone.vit_encoder                                         
1970-01-01 03:34:46 [DEBUG][utility.py:185] ViTEncoderBlock           .vi_t_backbone.vit_encoder.tranformer_block_1                      
1970-01-01 03:34:46 [DEBUG][utility.py:185] LayerNormalization        .vi_t_backbone.vit_encoder.tranformer_block_1.ln_1                 
1970-01-01 03:34:46 [DEBUG][utility.py:185] QStaticMultiHeadAttention .vi_t_backbone.vit_encoder.tranformer_block_1.mha                  
1970-01-01 03:34:46 [DEBUG][utility.py:185] QStaticEinsumDense        .vi_t_backbone.vit_encoder.tranformer_block_1.mha.query             a_scale=[0.00189279] w_scale=[0.00415615]
1970-01-01 03:34:46 [DEBUG][utility.py:185] QStaticEinsumDense        .vi_t_backbone.vit_encoder.tranformer_block_1.mha.key               a_scale=[0.00189279] w_scale=[0.00370309]
1970-01-01 03:34:46 [DEBUG][utility.py:185] QStaticEinsumDense        .vi_t_backbone.vit_encoder.tranformer_block_1.mha.value             a_scale=[0.00189279] w_scale=[0.0013797]
1970-01-01 03:34:46 [DEBUG][utility.py:185] Softmax                   .vi_t_backbone.vit_encoder.tranformer_block_1.mha.softmax          
1970-01-01 03:34:46 [DEBUG][utility.py:185] Dropout                   .vi_t_backbone.vit_encoder.tranformer_block_1.mha.dropout          
1970-01-01 03:34:46 [DEBUG][utility.py:185] QStaticEinsumDense        .vi_t_backbone.vit_encoder.tranformer_block_1.mha.attention_output  a_scale=[0.00718404] w_scale=[0.00568582]
1970-01-01 03:34:46 [DEBUG][utility.py:185] QDQLayer                  .vi_t_backbone.vit_encoder.tranformer_block_1.mha.q_qdq             a_scale=[0.00305243]
1970-01-01 03:34:46 [DEBUG][utility.py:185] QDQLayer                  .vi_t_backbone.vit_encoder.tranformer_block_1.mha.k_qdq             a_scale=[0.02323515]
1970-01-01 03:34:46 [DEBUG][utility.py:185] QDQLayer                  .vi_t_backbone.vit_encoder.tranformer_block_1.mha.a_qdq             a_scale=[0.00223214]
1970-01-01 03:34:46 [DEBUG][utility.py:185] QDQLayer                  .vi_t_backbone.vit_encoder.tranformer_block_1.mha.v_qdq             a_scale=[0.00732226]
1970-01-01 03:34:46 [DEBUG][utility.py:185] Dropout                   .vi_t_backbone.vit_encoder.tranformer_block_1.dropout              
1970-01-01 03:34:46 [DEBUG][utility.py:185] LayerNormalization        .vi_t_backbone.vit_encoder.tranformer_block_1.ln_2                 
1970-01-01 03:34:46 [DEBUG][utility.py:185] MLP                       .vi_t_backbone.vit_encoder.tranformer_block_1.mlp                  
1970-01-01 03:34:46 [DEBUG][utility.py:185] QStaticDense              .vi_t_backbone.vit_encoder.tranformer_block_1.mlp.dense_1           a_scale=[0.01193694] w_scale=[0.00350214]
1970-01-01 03:34:46 [DEBUG][utility.py:185] QStaticDense              .vi_t_backbone.vit_encoder.tranformer_block_1.mlp.dense_2           a_scale=[0.01522314] w_scale=[0.00583878]
1970-01-01 03:34:46 [DEBUG][utility.py:185] Dropout                   .vi_t_backbone.vit_encoder.tranformer_block_1.mlp.dropout          
1970-01-01 03:34:46 [DEBUG][utility.py:185] ViTEncoderBlock           .vi_t_backbone.vit_encoder.tranformer_block_2                      
```
