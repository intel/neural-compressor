Keras Gemma3 model quantization

============

This document describes quantization of Keras Gemma models using Neural Compressor on Intel® Xeon® processors.


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

The model can be downloaded manually from the [Kaggle website](https://www.kaggle.com/models/keras/gemma3/keras/gemma3_instruct_270m) or you can register yourself to Kaggle and setup your authentication according to https://www.kaggle.com/docs/api#authentication. Once you have configured Kaggle authentication in your environment, simply specify the name of the registered model you intend to use ([List of gemma3 models](https://keras.io/keras_hub/api/models/gemma3/gemma3_backbone/)). As default examples use gemma3_instruct_270m model.

## 4. Quantize model

To quantize the model you have to make 3 steps:

1. Load the original model:
```python
gemma_lm = Gemma3CausalLM.from_preset("model_name or path_to_the_model")
```
2. Calibrate the model using a dataset similar to the one that will be used later. In our example - we use just one prompt. We can choose which floating point format will be used in quantized model.

```python
config = StaticQuantConfig(weight_dtype="fp8_e4m3", activation_dtype="fp8_e4m3")


def calib_function(model):
    model.generate({"prompts": "Describe the city of Moscow"}, max_length=100)


gemma_lm = quantize_model(gemma_lm, config, calib_function)
```

3. Use quantized model
```python
output = gemma_lm.generate({"prompts": "Describe the city of Berlin?"}, max_length=100)
print("\nOutput after quantization:\n", output)
```

You can simply run this by running prepared [quantization.py](quantization.py) example.
When we run example, we can notice different answers of both models for the same prompt, but both should be reasonable. For example:  
Original gemma3_instruct_270m model: 
``` 
Describe the city of Berlin?

Berlin is a vibrant and diverse city with a rich history, a thriving arts scene, and a strong sense of community. It's a place where you can find a wide variety of experiences, from exploring historical landmarks to enjoying the vibrant nightlife. Berlin is known for its iconic landmarks like the Brandenburg Gate, the Reichstag, and the Wall. The city is also known for its diverse cultural scene, with museums, theaters, and music venues. Berlin is a
```
Quantized gemma3_instruct_270m model:
```
Describe the city of Berlin?

Berlin is a vibrant, diverse, and historically rich city with a unique blend of cultures and traditions. It's a place where history is palpable, art, and music, and the very alive. The city's architecture is stunning, with a mix of modern and a unique blend of styles. The streets are lined with charming, bustling with people, and the atmosphere. The food is a delight, with a variety of cuisines, from the most delicious and
```
## 5. Save and load quantized model

Calibration costs time, so we can calibrate once on representative data sets and later reuse it many times. To achieve it saving model functionality is supported.
You can run [prepare_static.py](prepare_static.py) script:
```bash
export XLA_FLAGS="\
    --xla_cpu_experimental_onednn_custom_call=true --xla_cpu_use_onednn=false \
    --xla_cpu_experimental_ynn_fusion_type=invalid --xla_cpu_use_xnnpack=false \
    --xla_backend_extra_options=xla_cpu_disable_new_fusion_emitter"
python prepare_static.py -m /path_to_your_gemma_model/gemma3_instruct_270m -q /path_to_store_your_quantized_model/gemma3_instruct_270m -p fp8_e4m3
```
or, if default parameters works for you, just:
```bash
export XLA_FLAGS="\
    --xla_cpu_experimental_onednn_custom_call=true --xla_cpu_use_onednn=false \
    --xla_cpu_experimental_ynn_fusion_type=invalid --xla_cpu_use_xnnpack=false \
    --xla_backend_extra_options=xla_cpu_disable_new_fusion_emitter"
python prepare_static.py
```

After this step, the saved model is stored in the `/path_to_store_your_quantized_model/gemma3_instruct_270m` file. You can load and use it with [use_static.py](use_static.py)

```
from keras_hub.models import Gemma3CausalLM
import neural_compressor.jax.quantization
gemma_lm = Gemma3CausalLM.from_preset("/path_to_store_your_quantized_model/gemma3_instruct_270m")
output = gemma_lm.generate({"prompts": "Describe the city of Berlin?"}, max_length=100)
```

You can notice that even script does not directly use neural_compressor function, but requires importing neural_compressor.jax.quantization to load quantized model.

## 6. Some debug

If you are interested how your model looks like after quantization you can set environment variable:
export LOGLEVEL=DEBUG
and then use print_model() function in your script. To see how it works just uncomment line 4 in [quantization.py](quantization.py)

Part of the gemma3_instruct_270m model could look like:

```
1970-01-01 00:01:17 [DEBUG][utility.py:156] ---------------------------- internal representation:
1970-01-01 00:01:17 [DEBUG][utility.py:172] KerasQuantizedModelWrapper
1970-01-01 00:01:17 [DEBUG][utility.py:172] InputLayer                   .padding_mask
1970-01-01 00:01:17 [DEBUG][utility.py:172] InputLayer                   .token_ids
1970-01-01 00:01:17 [DEBUG][utility.py:172] Gemma3Backbone               .gemma3_backbone
1970-01-01 00:01:17 [DEBUG][utility.py:172] InputLayer                   .gemma3_backbone.token_ids
1970-01-01 00:01:17 [DEBUG][utility.py:172] QStaticReversibleEmbedding   .gemma3_backbone.token_embedding
1970-01-01 00:01:17 [DEBUG][utility.py:172] QDQLayer                     .gemma3_backbone.token_embedding.inputs_qdq                                      a_scale=[0.26609924]
1970-01-01 00:01:17 [DEBUG][utility.py:172] QDQLayer                     .gemma3_backbone.token_embedding.kernel_qdq                                      a_scale=[0.00209263]
1970-01-01 00:01:17 [DEBUG][utility.py:172] InputLayer                   .gemma3_backbone.padding_mask
1970-01-01 00:01:17 [DEBUG][utility.py:172] Gemma3DecoderBlock           .gemma3_backbone.decoder_block_0
1970-01-01 00:01:17 [DEBUG][utility.py:172] RMSNormalization             .gemma3_backbone.decoder_block_0.pre_attention_norm
1970-01-01 00:01:17 [DEBUG][utility.py:172] RMSNormalization             .gemma3_backbone.decoder_block_0.post_attention_norm
1970-01-01 00:01:17 [DEBUG][utility.py:172] QStaticCachedGemma3Attention .gemma3_backbone.decoder_block_0.attention
1970-01-01 00:01:17 [DEBUG][utility.py:172] QStaticEinsumDense           .gemma3_backbone.decoder_block_0.attention.query                                 a_scale=[0.61645776] w_scale=[0.00193569]
1970-01-01 00:01:17 [DEBUG][utility.py:172] QStaticEinsumDense           .gemma3_backbone.decoder_block_0.attention.key                                   a_scale=[0.61645776] w_scale=[0.00083269]
1970-01-01 00:01:17 [DEBUG][utility.py:172] QStaticEinsumDense           .gemma3_backbone.decoder_block_0.attention.value                                 a_scale=[0.61645776] w_scale=[0.00063651]
1970-01-01 00:01:17 [DEBUG][utility.py:172] RMSNormalization             .gemma3_backbone.decoder_block_0.attention.query_norm
1970-01-01 00:01:17 [DEBUG][utility.py:172] RMSNormalization             .gemma3_backbone.decoder_block_0.attention.key_norm
1970-01-01 00:01:17 [DEBUG][utility.py:172] Dropout                      .gemma3_backbone.decoder_block_0.attention.dropout
1970-01-01 00:01:17 [DEBUG][utility.py:172] QStaticEinsumDense           .gemma3_backbone.decoder_block_0.attention.attention_output                      a_scale=[0.15646777] w_scale=[0.00125558]
1970-01-01 00:01:17 [DEBUG][utility.py:172] Softmax                      .gemma3_backbone.decoder_block_0.attention.softmax
1970-01-01 00:01:17 [DEBUG][utility.py:172] QStaticRotaryEmbedding       .gemma3_backbone.decoder_block_0.attention.rotary_embedding
1970-01-01 00:01:17 [DEBUG][utility.py:172] QDQLayer                     .gemma3_backbone.decoder_block_0.attention.rotary_embedding.positions_qdq        a_scale=[0.22098215]
1970-01-01 00:01:17 [DEBUG][utility.py:172] QDQLayer                     .gemma3_backbone.decoder_block_0.attention.rotary_embedding.inverse_freq_qdq     a_scale=[0.00223214]
1970-01-01 00:01:17 [DEBUG][utility.py:172] QDQLayer                     .gemma3_backbone.decoder_block_0.attention.q_qdq                                 a_scale=[0.00163109]
1970-01-01 00:01:17 [DEBUG][utility.py:172] QDQLayer                     .gemma3_backbone.decoder_block_0.attention.k_qdq                                 a_scale=[0.03631029]
1970-01-01 00:01:17 [DEBUG][utility.py:172] QDQLayer                     .gemma3_backbone.decoder_block_0.attention.attention_softmax_qdq                 a_scale=[0.00223214]
1970-01-01 00:01:17 [DEBUG][utility.py:172] QDQLayer                     .gemma3_backbone.decoder_block_0.attention.v_qdq                                 a_scale=[0.20262058]
```

