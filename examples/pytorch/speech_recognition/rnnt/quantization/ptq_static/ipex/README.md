# RNNT MLPerf Inference v1.1

> Note: not support IPEX 1.10, 1.11

## SW requirements
###
| SW |configuration |
|--|--|
| GCC | GCC 9.3 |

## Steps to run RNNT

### 1. Install anaconda 3.0
```
  wget https://repo.continuum.io/archive/Anaconda3-5.0.0-Linux-x86_64.sh -O anaconda3.sh
  chmod +x anaconda3.sh
  ~/anaconda3.sh -b -p ~/anaconda3
  ~/anaconda3/bin/conda create -n rnnt python=3.7

  export PATH=~/anaconda3/bin:$PATH
  source ~/anaconda3/bin/activate rnnt
```
### 2. Prepare code and environment
```
  cd examples/pytorch/speech_recognition/rnnt/quantization/ptq_static/ipex
  bash prepare_env.sh
```

### 3. Install IPEX
refer [intel/intel-extension-for-pytorch at mlperf/inference-1.1 (github.com)](https://github.com/intel/intel-extension-for-pytorch/tree/mlperf/inference-1.1)

1. install PyTorch1.8 and TorchVision0.9

   refer [PyTorch install](https://pytorch.org/get-started/locally/)
   ```shell position-relative
   pip3 install torch==1.8.0+cpu torchvision==0.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
   ```
2. Get Intel PyTorch Extension source and install
    > **Note**
    >
    > GCC9 compiler is recommended
    >

   ```shell position-relative
   git clone https://github.com/intel/intel-extension-for-pytorch
   cd intel-extension-for-pytorch
   git checkout mlperf/inference-1.1
   git submodule sync
   git submodule update --init --recursive
   pip install lark-parser hypothesis

   python setup.py install
   ```

### 4. Prepare model and dataset
```
  work_dir=mlperf-rnnt-librispeech
  local_data_dir=$work_dir/local_data
  mkdir -p $local_data_dir
  librispeech_download_dir=.
  # prepare model
  wget https://zenodo.org/record/3662521/files/DistributedDataParallel_1576581068.9962234-epoch-100.pt?download=1 -O $work_dir/rnnt.pt

  # prepare inference dataset
  wget https://www.openslr.org/resources/12/dev-clean.tar.gz
  # suggest you check run.sh to locate the dataset
  python pytorch/utils/download_librispeech.py \
         pytorch/utils/librispeech-inference.csv \
         $librispeech_download_dir \
         -e $local_data_dir --skip_download
  python pytorch/utils/convert_librispeech.py \
         --input_dir $local_data_dir/LibriSpeech/dev-clean \
         --dest_dir $local_data_dir/dev-clean-wav \
         --output_json $local_data_dir/dev-clean-wav.json
```

### 5. tune RNN-T with Neural Compressor
  Please update the setup_env_offline.sh or setup_env_server.sh and user.conf according to your platform resource.
```
  # offline
  ./run_tuning.sh --dataset_location=$local_data_dir --input_model=$work_dir/rnnt.pt
  # server scenario
  ./run_tuning.sh --dataset_location=$local_data_dir --input_model=$work_dir/rnnt.pt --server
```

### 6. benchmark
```
# fp32 benchmark
bash ./run_benchmark.sh --dataset_location=/path/to/RNN-T/dataset/LibriSpeech --input_model=rnnt.pt --mode=benchmark
# int8+bf16 benchmark
bash ./run_benchmark.sh --dataset_location=/path/to/RNN-T/dataset/LibriSpeech --input_model=rnnt.pt --mode=benchmark --int8=true
# fp32 accuracy
bash ./run_benchmark.sh --dataset_location=/path/to/RNN-T/dataset/LibriSpeech --input_model=rnnt.pt --mode=accuracy
# int8+bf16 benchmark
bash ./run_benchmark.sh --dataset_location=/path/to/RNN-T/dataset/LibriSpeech --input_model=rnnt.pt --mode=accuracy --int8=true

```

### Note on Server scenario

* Only quantized encoder and decoder is bf16 ops. 
* For server scenario, we exploit the fact that incoming data have different sequence lengths (and inference times) by bucketing according to sequence length 
and specifying batch size for each bucket such that latency can be satisfied. The settings are specified in machine.conf file and required fields 
are cores_per_instance, num_instances, waveform_len_cutoff, batch_size.

