# Step by step example how to dump weights data for PyTorch model with Neural Insights
1. [Introduction](#introduction)
2. [Preparation](#preparation)
3. [Running the quantization](#running-the-quantization)

# Introduction
In this instruction weight data will be dumped using Neural Insights. PyTorch GPT-J-6B model will be used as an example.

# Preparation
## Source
First you need to install Intel® Neural Compressor.
```shell
# Install Neural Compressor
git clone https://github.com/intel/neural-compressor.git
cd neural-compressor 
pip install -r requirements.txt 
python setup.py install

# Install Neural Insights
pip install -r neural_insights/requirements.txt
python setup.py install neural_insights
```

## Requirements
```shell
cd /examples/pytorch/nlp/huggingface_models/language-modeling/quantization/ptq_static/fx
pip install -r requirements.txt
```

# Running the quantization
Before applying quantization, modify some code in `run_clm.py` file to enable Neural Insights:
1. Set the argument `diagnosis` to be `True` in `PostTrainingQuantConfig` so that Neural Insights will dump weights of quantizable Ops in this model.

```python
conf = PostTrainingQuantConfig(
    accuracy_criterion=accuracy_criterion,
    diagnosis=True,
)
```
2. Quantize the model with following command:
```shell
python run_clm.py \
  --model_name_or_path EleutherAI/gpt-j-6B \
  --dataset_name wikitext\
  --dataset_config_name wikitext-2-raw-v1 \
  --do_train \
  --do_eval \
  --tune \
  --output_dir saved_results
```

Results would be dumped into `nc_workspace` directory in similar structure:
```
├── history.snapshot
├── input_model.pt
├── inspect_saved
│   ├── fp32
│   │   └── inspect_result.pkl
│   └── quan
│       └── inspect_result.pkl
├── model_summary.txt
└── weights_table.csv
```
