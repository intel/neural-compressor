Step-by-Step
============

This document is used to list steps of reproducing PyTorch BERT tuning zoo result.

# Prerequisite

## Environment

The dependent packages are all in requirements, please install as following.

```
pip install -r requirements.txt
```

# Quantization

## Command
If the automatic download from modelhub fails, you can download [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B?text=My+name+is+Clara+and+I+am) offline.

```shell

python run_clm.py \
  --model_name_or_path EleutherAI/gpt-j-6B \
  --dataset_name wikitext\
  --dataset_config_name wikitext-2-raw-v1 \
  --do_train \
  --do_eval \
  --tune \
  --output_dir /path/to/checkpoint/dir
```
## Language-Modeling Task
```bash
sh run_tuning.sh --topology=topology_name --input_model=/path/to/checkpoint/dir
```
> NOTE
>
> topology_name can be one of {"gpt_j_wikitext", "dialogpt_wikitext", "reformer_crime_and_punishment", "ctrl_WikiText"}

# Saving and Loading Model
## Saving model:
```
from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig
from neural_compressor import quantization
accuracy_criterion = AccuracyCriterion(higher_is_better=False, tolerable_loss=0.5)
conf = PostTrainingQuantConfig(accuracy_criterion=accuracy_criterion)
q_model = quantization.fit(model,
                           conf,
                           calib_dataloader=dataloader(),
                           eval_func=eval_func)
q_model.save("output_dir")
```

Here, `q_model` is the Neural Compressor model class, so it has "save" API:

```python
q_model.save("Path_to_save_quantized_model")
```

## Loading model:

```python
from neural_compressor.utils.pytorch import load
quantized_model = load(tuned_checkpoint,
                       model)
```

Please refer to [Sample code](./run_clm.py).

