# Run GPTQ tasks on GPT-j-6B model for summary task

# Step by Step

## Step 1 Prepare datasets and models
Use the following link to get
[**CNN Daily Mail** datasets](https://github.com/intel-innersource/frameworks.ai.benchmarking.mlperf.submission.inference-submission-v3-1/tree/master/closed/Intel/code/gpt-j/pytorch-cpu#download-and-prepare-dataset)
and [gpt-j-6B mlperf model](https://github.com/mlcommons/inference/tree/master/language/gpt-j#download-gpt-j-model)

## Step 2 Run GPTQ quantization
```shell
sh run-gptq-gptj-sym.sh
```
