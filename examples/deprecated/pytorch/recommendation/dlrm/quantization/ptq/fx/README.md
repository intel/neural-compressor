Step-by-Step
============

This document is used to list steps of reproducing PyTorch DLRM tuning zoo result. and original DLRM README is in [DLRM README](https://github.com/facebookresearch/dlrm/blob/master/README.md)

> **Note**
>
> Please  ensure your PC have >370G memory to run DLRM

# Prerequisite

### 1. Environment

PyTorch 1.10 or higher version is needed with pytorch_fx backend.

  ```shell
  # Install dependency
  cd examples/pytorch/recommendation/dlrm/quantization/ptq/fx
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
  cd examples/pytorch/recommendation/dlrm/quantization/ptq/fx
  bash run_quant.sh --input_model="/path/of/pretrained/model" --dataset_location="/path/of/dataset"
  ```

### benchmark
```shell
bash run_benchmark.sh --input_model="/path/of/pretrained/model" --dataset_location="/path/of/dataset" --mode=accuracy --int8=true
```


### code update

We need update dlrm_s_pytorch_tune.py like below

```python
class DLRM_DataLoader(object):
    def __init__(self, loader=None):
        self.loader = loader
        self.batch_size = loader.dataset.batch_size
    def __iter__(self):
        for X_test, lS_o_test, lS_i_test, T in self.loader:
            yield (X_test, lS_o_test, lS_i_test), T
```

```python
    def eval_func(model):
        batch_time = AverageMeter('Time', ':6.3f')
        scores = []
        targets = []
        for j, (X_test, lS_o_test, lS_i_test, T) in enumerate(test_ld):
            if j >= args.warmup_iter:
                start = time_wrap(False)
            if not lS_i_test.is_contiguous():
                lS_i_test = lS_i_test.contiguous()
            Z = model(X_test, lS_o_test, lS_i_test)
            S = Z.detach().cpu().numpy()  # numpy array
            T = T.detach().cpu().numpy()  # numpy array
            scores.append(S)
            targets.append(T)
            if j >= args.warmup_iter:
                batch_time.update(time_wrap(False) - start)
            if args.iters > 0 and j >= args.warmup_iter + args.iters - 1:
                break

        scores = np.concatenate(scores, axis=0)
        targets = np.concatenate(targets, axis=0)
        roc_auc = sklearn.metrics.roc_auc_score(targets, scores)

        return roc_auc

    eval_dataloader = DLRM_DataLoader(test_ld)
	dlrm.eval()
	from neural_compressor import PostTrainingQuantConfig, quantization
	conf = PostTrainingQuantConfig()
	q_model = quantization.fit(
						dlrm,
						conf=conf,
						calib_dataloader=eval_dataloader
						)
	q_model.save("saved_results")
	exit(0)
```
