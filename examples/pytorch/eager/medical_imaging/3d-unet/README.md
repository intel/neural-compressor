# Introduction

This example is used to demostrate 3D-Unet int8 accuracy by tuning with LPOT on PyTorch FBGEMM path.

The 3D-Unet source code comes from [mlperf](https://github.com/mlcommons/inference/tree/master/vision/medical_imaging/3d-unet), commit SHA is **b7e8f0da170a421161410d18e5d2a05d75d6bccf**. [nnUnet](https://github.com/MIC-DKFZ/nnaUNet) commit SHA is **b38c69b345b2f60cd0d053039669e8f988b0c0af**. User could diff them with this example to know which changes are made to integrate with LPOT.

The model is performing [BraTS 2019](https://www.med.upenn.edu/cbica/brats2019/data.html) brain tumor segmentation task.

## Prerequisites

> **note**
>
> PyTorch 1.6.0 and above version has bug on layernorm int8 implementation, which causes 3D-Unet int8 model get ~0.0 accuracy.
> so this example takes PyTorch 1.5.0 as requirments.

```shell
  # install PyTorch 1.5.0+cpu
  pip install torch==1.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

  # download BraTS 2019 from https://www.med.upenn.edu/cbica/brats2019/data.html
  export DOWNLOAD_DATA_DIR=<path/to/MICCAI_BraTS_2019_Data_Training> # point to location of downloaded BraTS 2019 Training dataset.

  # install dependency required by data preprocessing script
  cd ./nnUnet
  python setup.py install
  cd ..

  # download pytorch model
  make download_pytorch_model

  # generate preprocessed data
  make preprocess_data

  # create postprocess dir
  make mkdir_postprocessed_data

  # generate calibration preprocessed data
  python preprocess.py --preprocessed_data_dir=./build/calib_preprocess/ --validation_fold_file=./brats_cal_images_list.txt

  # install mlperf loadgen required by tuning script
  git clone https://github.com/mlcommons/inference.git --recursive
  cd inference
  git checkout b7e8f0da170a421161410d18e5d2a05d75d6bccf
  cd loadgen
  pip install absl-py
  python setup.py install
  cd ..
```

## running cmd

```shell
  make run_pytorch_LPOT_tuning
  
  or

  python run.py --model_dir=build/result/nnUNet/3d_fullres/Task043_BraTS2019/nnUNetTrainerV2__nnUNetPlansv2.mlperf.1 --backend=pytorch --accuracy --preprocessed_data_dir=build/preprocessed_data/ --mlperf_conf=./mlperf.conf --tune

```

## Model Baseline


| model | framework | accuracy | dataset | model link | model source | precision | notes |
| - | - | - | - | - | - | - | - |
| 3D-Unet | PyTorch | **mean = 0.85300** (whole tumor = 0.9141, tumor core = 0.8679, enhancing tumor = 0.7770) | [Fold 1](folds/fold1_validation.txt) of [BraTS 2019](https://www.med.upenn.edu/cbica/brats2019/data.html) Training Dataset | [from zenodo](https://zenodo.org/record/3904106) | Trained in PyTorch using codes from[nnUnet](https://github.com/MIC-DKFZ/nnUNet) on [Fold 0](folds/fold0_validation.txt), [Fold 2](folds/fold2_validation.txt), [Fold 3](folds/fold3_validation.txt), and [Fold 4](folds/fold4_validation.txt) of [BraTS 2019](https://www.med.upenn.edu/cbica/brats2019/data.html) Training Dataset. | fp32 |   |
