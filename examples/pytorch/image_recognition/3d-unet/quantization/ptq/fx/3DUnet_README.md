# MLPerf Inference Benchmarks for Medical Image 3D Segmentation

The chosen model is 3D-Unet in [nnUnet](https://github.com/MIC-DKFZ/nnUNet) performing [BraTS 2019](https://www.med.upenn.edu/cbica/brats2019/data.html) brain tumor segmentation task.

## Prerequisites

If you would like to run on NVIDIA GPU, you will need:

- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
- Any NVIDIA GPU supported by TensorFlow or PyTorch

## Supported Models

| model | framework | accuracy | dataset | model link | model source | precision | notes |
| ----- | --------- | -------- | ------- | ---------- | ------------ | --------- | ----- |
| 3D-Unet | PyTorch | **mean = 0.85300** (whole tumor = 0.9141, tumor core = 0.8679, enhancing tumor = 0.7770) | [Fold 1](folds/fold1_validation.txt) of [BraTS 2019](https://www.med.upenn.edu/cbica/brats2019/data.html) Training Dataset | [from zenodo](https://zenodo.org/record/3904106) | Trained in PyTorch using codes from [nnUnet](https://github.com/MIC-DKFZ/nnUNet) on [Fold 0](folds/fold0_validation.txt), [Fold 2](folds/fold2_validation.txt), [Fold 3](folds/fold3_validation.txt), and [Fold 4](folds/fold4_validation.txt) of [BraTS 2019](https://www.med.upenn.edu/cbica/brats2019/data.html) Training Dataset. | fp32 | |
| 3D-Unet | ONNX | **mean = 0.85300** (whole tumor = 0.9141, tumor core = 0.8679, enhancing tumor = 0.7770) | [Fold 1](folds/fold1_validation.txt) of [BraTS 2019](https://www.med.upenn.edu/cbica/brats2019/data.html) Training Dataset | [from zenodo](https://zenodo.org/record/3928973) | Converted from the PyTorch model using [script](unet_pytorch_to_onnx.py). | fp32 | |
| 3D-Unet | Tensorflow | **mean = 0.85300** (whole tumor = 0.9141, tumor core = 0.8679, enhancing tumor = 0.7770) | [Fold 1](folds/fold1_validation.txt) of [BraTS 2019](https://www.med.upenn.edu/cbica/brats2019/data.html) Training Dataset | [from zenodo](https://zenodo.org/record/3928991) | Converted from the ONNX model using [script](unet_onnx_to_tf.py). | fp32 | |
| 3D-Unet | OpenVINO | **mean = 0.85300** (whole tumor = 0.9141, tumor core = 0.8679, enhancing tumor = 0.7770) | [Fold 1](folds/fold1_validation.txt) of [BraTS 2019](https://www.med.upenn.edu/cbica/brats2019/data.html) Training Dataset | [from zenodo](https://zenodo.org/record/3929002) | Converted from the ONNX model. | fp32 | |


## Disclaimer
This benchmark app is a reference implementation that is not meant to be the fastest implementation possible.

## Commands

Please download [BraTS 2019](https://www.med.upenn.edu/cbica/brats2019/data.html) separately and unzip the dataset.

Please run the following commands:

- `export DOWNLOAD_DATA_DIR=<path/to/MICCAI_BraTS_2019_Data_Training>`: point to location of downloaded BraTS 2019 Training dataset.
- `make setup`: initialize submodule and download models.
- `make build_docker`: build docker image.
- `make launch_docker`: launch docker container with an interaction session.
- `make preprocess_data`: preprocess the BraTS 2019 dataset.
- `python3 run.py --backend=[tf|pytorch|onnxruntime|ov] --scenario=[Offline|SingleStream|MultiStream|Server] [--accuracy] --model=[path/to/model_file(tf/onnx/OpenVINO only)]`: run the harness inside the docker container. Performance or Accuracy results will be printed in console.
- `python3 accuracy-brats.py --log_file=<LOADGEN_LOG> --output_dtype=<DTYPE>`: compute accuracy from a LoadGen accuracy JSON log file. 

## Details

- SUT implementations are in [ov_SUT.py](ov_SUT.py), [pytorch_SUT.py](pytorch_SUT.py), [onnxruntime_SUT.py](onnxruntime_SUT.py), and [tf_SUT.py](tf_SUT.py). QSL implementation is in [brats_QSL.py](brats_QSL.py).
- The script [accuracy-brats.py](accuracy-brats.py) parses LoadGen accuracy log, post-processes it, and computes the accuracy.
- Preprocessing and evaluation (including post-processing) are not included in the timed path.
- The input to the SUT is a volume of size `[4, 224, 224, 160]`. The output from SUT is a volume of size `[4, 224, 224, 160]` with predicted label logits for each voxel.

## Calibration Set

The calibration set is the forty images listed in [brats_cal_images_list.txt](../../../calibration/BraTS/brats_cal_images_list.txt). They are randomly selected from [Fold 0](folds/fold0_validation.txt), [Fold 2](folds/fold2_validation.txt), [Fold 3](folds/fold3_validation.txt), and [Fold 4](folds/fold4_validation.txt) of [BraTS 2019](https://www.med.upenn.edu/cbica/brats2019/data.html) Training Dataset.

## License

Apache License 2.0
