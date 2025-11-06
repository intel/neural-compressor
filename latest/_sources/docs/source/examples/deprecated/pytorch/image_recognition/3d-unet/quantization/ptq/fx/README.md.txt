Step-by-Step
============

This example is used to demonstrate the steps of reproducing quantization and benchmarking results with Intel® Neural Compressor.

The 3D-Unet source code comes from [mlperf](https://github.com/mlcommons/inference/tree/v1.0.1/vision/medical_imaging/3d-unet), commit SHA is **b7e8f0da170a421161410d18e5d2a05d75d6bccf**; [nnUnet](https://github.com/MIC-DKFZ/nnUNet) commit SHA is **b38c69b345b2f60cd0d053039669e8f988b0c0af**. Users could diff them with this example to know which changes have been made to integrate with Intel® Neural Compressor..

The model is performing on [BraTS 2019](https://www.med.upenn.edu/cbica/brats2019/data.html) brain tumor segmentation task.

# Prerequisite
## 1. Environment
Python 3.6 or higher version is recommended.
The dependent packages are all in requirements, please install as following.
```shell
cd examples/pytorch/image_recognition/3d-unet/quantization/ptq/fx
pip install -r requirements.txt
```
## 2. Preprocess Dataset
```shell
  # download BraTS 2019 from https://www.med.upenn.edu/cbica/brats2019/data.html
  export DOWNLOAD_DATA_DIR=<path/to/MICCAI_BraTS_2019_Data_Training> # point to location of downloaded BraTS 2019 Training dataset.

  # install dependency required by data preprocessing script
  git clone https://github.com/MIC-DKFZ/nnUNet.git --recursive
  cd nnUNet/
  git checkout b38c69b345b2f60cd0d053039669e8f988b0c0af
  # replace sklearn in the older version with scikit-learn
  sed -i 's/sklearn/scikit-learn/g' setup.py
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
  cd ../..
```

# Run
## 1. Quantization

```shell
  make run_pytorch_NC_tuning
```
  
  or

```shell
  python run.py --model_dir=build/result/nnUNet/3d_fullres/Task043_BraTS2019/nnUNetTrainerV2__nnUNetPlansv2.mlperf.1 --backend=pytorch --accuracy --preprocessed_data_dir=build/preprocessed_data/ --mlperf_conf=./mlperf.conf --tune
```
## 2. Benchmark
```bash
# int8
sh run_benchmark.sh --int8=true --input_model=build/result/nnUNet/3d_fullres/Task043_BraTS2019/nnUNetTrainerV2__nnUNetPlansv2.mlperf.1 --dataset_location=build/preprocessed_data/
# fp32
sh run_benchmark.sh --input_model=build/result/nnUNet/3d_fullres/Task043_BraTS2019/nnUNetTrainerV2__nnUNetPlansv2.mlperf.1 --dataset_location=build/preprocessed_data/
```
## 3. Model Baseline
| model | framework | accuracy | dataset | model link | model source | precision |
| - | - | - | - | - | - | - |
| 3D-Unet | PyTorch | **mean = 0.85300** (whole tumor = 0.9141, tumor core = 0.8679, enhancing tumor = 0.7770) | [Fold 1](folds/fold1_validation.txt) of [BraTS 2019](https://www.med.upenn.edu/cbica/brats2019/data.html) Training Dataset | [from zenodo](https://zenodo.org/record/3904106) | Trained in PyTorch using codes from[nnUnet](https://github.com/MIC-DKFZ/nnUNet) on [Fold 0](folds/fold0_validation.txt), [Fold 2](folds/fold2_validation.txt), [Fold 3](folds/fold3_validation.txt), and [Fold 4](folds/fold4_validation.txt) of [BraTS 2019](https://www.med.upenn.edu/cbica/brats2019/data.html) Training Dataset. | fp32 |
