Step-by-Step
============

This document is used to list steps of reproducing PyTorch DLRM tuning zoo result. and original DLRM README is in [DLRM README](https://github.com/facebookresearch/dlrm/blob/master/README.md)

> **Note**
>
> Please  ensure your PC have >370G memory to run DLRM
> IPEX version >= 1.11

# Prerequisite

### 1. Environment

PyTorch 1.11 or higher version is needed with pytorch_fx backend.

  ```shell
  # Install dependency
  cd examples/pytorch/recommendation/dlrm/quantization/ptq/ipex
  pip install -r requirements.txt
  ```
> Note: Validated PyTorch [Version](/docs/source/installation_guide.md#validated-software-environment).

### 2. Prepare Dataset

  The code supports interface with the [Criteo Terabyte Dataset](https://labs.criteo.com/2013/12/download-terabyte-click-logs/)

  1. download the raw data files day_0.gz, ...,day_23.gz and unzip them.
  2. Specify the location of the unzipped text files day_0, ...,day_23, using --raw-data-file=<path/day> (the day number will be appended automatically), please refer "Run" command.

### 3. Prepare pretrained model

  Download the DLRM PyTorch weights (`tb00_40M.pt`, 90GB) from the
[MLPerf repo](https://github.com/mlcommons/inference/tree/master/recommendation/dlrm/pytorch#more-information-about-the-model-weights)

# Run
### tune with INC
  ```shell
  cd examples/pytorch/recommendation/dlrm/quantization/ptq/ipex
  bash run_quant.sh --input_model="/path/of/pretrained/model" --dataset_location="/path/of/dataset"
  ```

### benchmark
```shell
bash run_benchmark.sh --input_model="/path/of/pretrained/model" --dataset_location="/path/of/dataset" --mode=accuracy --int8=true
```


Examples of enabling Intel® Neural Compressor
=========================

This is a tutorial of how to enable DLRM model with Intel® Neural Compressor.


### Code update

We need update dlrm_s_pytorch.py like below

```python
# evaluation
def eval_func(model):
	args.int8 = model.is_quantized
	with torch.no_grad():
		return inference(
			args,
			model,
			best_acc_test,
			best_auc_test,
			test_ld,
			trace=args.int8
		)

# calibration
def calib_fn(model):
	calib_number = 0
	for X_test, lS_o_test, lS_i_test, T in train_ld:
		if calib_number < 102400:
			model(X_test, lS_o_test, lS_i_test)
			calib_number += 1

from neural_compressor.torch.quantization import SmoothQuantConfig, autotune, TuningConfig
tune_config = TuningConfig(config_set=SmoothQuantConfig.get_config_set_for_tuning())
dlrm = autotune(
	dlrm, 
	tune_config=tune_config,
	eval_fn=eval_func,
	run_fn=calib_fn,
)
dlrm.save("saved_results")
```
