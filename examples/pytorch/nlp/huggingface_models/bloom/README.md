## BLOOM

Finetuned BLOOM models can be found in BLOOMZ's [model card](https://huggingface.co/bigscience/bloomz). ```--model_name_or_path``` argument can be changed to any one of the finetuned models listed in the model card. ```--task_hub_name``` can either be ```glue``` or ```super_glue```, or the name of any other HuggingFace dataset that is compatible with BLOOM benchmark, with minor modifications to the script.

### Run the script
```
python -m neural_coder run_bloom.py
    --model_name_or_path bloomz-7b1
    --task_name rte
    --task_hub_name super_glue 
    --do_eval 
    --output_dir result
```

By default it runs dynamic quantization. To run static quantization, run with ```python -m neural_coder --approach static```.

### Try text generation with quantized model

For example:

```
python bloom_prediction.py --prompt "Translate to English: Je t’aime."
```

Result:

```
Prediction of the original model:  Translate to English: Je t’aime. I love you.</s>
Prediction of the quantized model:  Translate to English: Je t’aime. I love you.</s>
```