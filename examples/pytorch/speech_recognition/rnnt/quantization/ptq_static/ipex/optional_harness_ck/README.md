# MLPerf Inference - Speech Recognition - RNN-T

We describe an automated and reproducible workflow for the [RNN-T
workload](https://github.com/mlperf/inference/tree/master/v0.7/speech_recognition/rnnt)
implemented using the [Collective Knowledge](http://cknowledge.org) technology. It automatically
downloads the model and the dataset, preprocesses the dataset, builds the LoadGen API, etc.
For any questions or questions, please email info@dividiti.com or simply [open an issue](https://github.com/mlperf/inference/issues) on GitHub.

**NB:** Below we give an _essential_ sequence of steps that should result in a successful setup
of the RNN-T workflow on a minimally configured Linux system.

The steps are extracted from a [minimalistic Amazon Linux
2](https://github.com/ctuning/ck-mlperf/blob/master/docker/speech-recognition.rnnt/Dockerfile.amazonlinux.min)
Docker image, which is derived from a more verbose [Amazon Linux
2](https://github.com/ctuning/ck-mlperf/blob/master/docker/speech-recognition.rnnt/Dockerfile.amazonlinux)
Docker image by omitting steps that the [Collective Knowledge
framework](https://github.com/ctuning/ck) performs automatically.

For example, installing the preprocessed dataset is explicit in the verbose image:
```
#-----------------------------------------------------------------------------#
# Step 3. Download the official MLPerf Inference RNNT dataset (LibriSpeech
# dev-clean) and preprocess it to wav.
#-----------------------------------------------------------------------------#
RUN ck install package --tags=dataset,speech-recognition,dev-clean,original
# NB: Can ignore the lzma related warning.
RUN ck install package --tags=dataset,speech-recognition,dev-clean,preprocessed
#-----------------------------------------------------------------------------#
```
but is implicit in the minimalistic image:
```
#- #-----------------------------------------------------------------------------#
#- # Step 3. Download the official MLPerf Inference RNNT dataset (LibriSpeech
#- # dev-clean) and preprocess it to wav.
#- #-----------------------------------------------------------------------------#
#- RUN ck install package --tags=dataset,speech-recognition,dev-clean,original
#- # NB: Can ignore the  lzma related warning.
#- RUN ck install package --tags=dataset,speech-recognition,dev-clean,preprocessed
#- #-----------------------------------------------------------------------------#
```
because it's going to be triggered by a test performance run:
```
#+ #-----------------------------------------------------------------------------#
#+ # Step 6. Pull all the implicit dependencies commented out in Steps 1-5.
#+ #-----------------------------------------------------------------------------#
RUN ck run program:speech-recognition-pytorch-loadgen --cmd_key=performance --skip_print_timers
#+ #-----------------------------------------------------------------------------#
```
(Omitted steps are commented out with `#- `. Added steps are commented with `#+ `.)

For other possible variations and workarounds see the [complete
collection](https://github.com/ctuning/ck-mlperf/blob/master/docker/speech-recognition.rnnt/README.md)
of Docker images for this workflow including Ubuntu, Debian and CentOS.

# Table of Contents

1. [Installation](#install)
    1. Install [system-wide prerequisites](#install_system)
        1. [Ubuntu 20.04 or similar](#install_system_ubuntu)
        1. [CentOS 7 or similar](#install_system_centos_7)
        1. [CentOS 8 or similar](#install_system_centos_8)
    1. Install [Collective Knowledge](#install_ck) (CK) and its repositories
    1. Detect [GCC](#detect_gcc)
    1. Detect [Python](#detect_python)
    1. Install [Python dependencies](#install_python_deps)
    1. Install a branch of the [MLPerf Inference](#install_inference_repo) repo
1. [Usage](#usage)
    1. [Performance](#usage_performance)
    1. [Accuracy](#usage_performance)

<a name="install"></a>
## Installation

<a name="install_system"></a>
### Install system-wide prerequisites

**NB:** Run the below commands for your Linux system with `sudo` or as superuser.

<a name="install_system_ubuntu"></a>
#### Ubuntu 20.04 or similar
```bash
$ sudo apt update -y
$ sudo apt install -y apt-utils
$ sudo apt upgrade -y
$ sudo apt install -y\
 python3 python3-pip\
 gcc g++\
 make patch vim\
 git wget zip libz-dev\
 libsndfile1-dev
$ sudo apt clean
```

<a name="install_system_centos_7"></a>
#### CentOS 7 or similar
```bash
$ sudo yum upgrade -y
$ sudo yum install -y\
 python3 python3-pip python3-devel\
 gcc gcc-c++\
 make which patch vim\
 git wget zip unzip\
 tar xz\
 libsndfile-devel
$ sudo yum clean all
```

<a name="install_system_centos_8"></a>
#### CentOS 8 or similar
```bash
$ sudo yum upgrade -y
$ sudo yum install -y\
 gcc gcc-c++\
 make which patch vim\
 git wget zip unzip\
 openssl-devel bzip2-devel libffi-devel\
$ sudo yum clean all
$ sudo dnf install -y python3 python3-pip python3-devel
$ sudo dnf --enablerepo=PowerTools install -y libsndfile-devel
```


<a name="install_ck"></a>
### Install [Collective Knowledge](http://cknowledge.org/) (CK) and its repositories

```bash
$ export CK_PYTHON=/usr/bin/python3
$ $CK_PYTHON -m pip install --ignore-installed pip setuptools --user
$ $CK_PYTHON -m pip install ck
$ ck version
V1.15.0
$ ck pull repo:ck-mlperf
$ ck pull repo:ck-pytorch
```

<a name="detect_gcc"></a>
### Detect (system) GCC
```
$ export CK_CC=/usr/bin/gcc
$ ck detect soft:compiler.gcc --full_path=$CK_CC
$ ck show env --tags=compiler,gcc
Env UID:         Target OS: Bits: Name:          Version: Tags:

b8bd7b49f72f9794   linux-64    64 GNU C compiler 7.3.1    64bits,compiler,gcc,host-os-linux-64,lang-c,lang-cpp,target-os-linux-64,v7,v7.3,v7.3.1
```
**NB:** Required to build the FLAC and SoX dependencies of preprocessing. CK can normally detect compilers automatically, but we are playing safe here.

<a name="detect_python"></a>
### Detect (system) Python
```
$ export CK_PYTHON=/usr/bin/python3
$ ck detect soft:compiler.python --full_path=$CK_PYTHON
$ ck show env --tags=compiler,python
Env UID:         Target OS: Bits: Name:  Version: Tags:

633a6b22205eb07f   linux-64    64 python 3.7.6    64bits,compiler,host-os-linux-64,lang-python,python,target-os-linux-64,v3,v3.7,v3.7.6
```
**NB:** CK can normally detect available Python interpreters automatically, but we are playing safe here.

<a name="install_python_deps"></a>
### Install Python dependencies (in userspace)

#### Install implicit dependencies via pip
```bash
$ export CK_PYTHON=/usr/bin/python3
$ $CK_PYTHON -m pip install --user --upgrade \
  tqdm wheel toml unidecode inflect sndfile librosa numba==0.48
...
Successfully installed inflect-4.1.0 librosa-0.7.2 llvmlite-0.31.0 numba-0.48.0 sndfile-0.2.0 unidecode-1.1.1 wheel-0.34.2
```
**NB:** These dependencies are _implicit_, i.e. CK will not try to satisfy them. If they are not installed, however, the workflow will fail.


#### Install explicit dependencies via CK (also via `pip`, but register with CK at the same time)
```bash
$ ck install package --tags=python-package,torch
$ ck install package --tags=python-package,pandas
$ ck install package --tags=python-package,sox
$ ck install package --tags=python-package,absl
```
**NB:** These dependencies are _explicit_, i.e. CK will try to satisfy them automatically. On a machine with multiple versions of Python, things can get messy, so we are playing safe here.

<a name="install_inference_repo"></a>
### Install an MLPerf Inference [branch](https://github.com/dividiti/inference/tree/dvdt-rnnt) with [dividiti](http://dividiti.com)'s tweaks for RNN-T
```bash
$ ck install package --tags=mlperf,inference,source,dividiti.rnnt
```
**NB:** This source will be used for building LoadGen as well.


<a name="usage"></a>
## Usage

<a name="usage_performance"></a>
### Running a performance test

The first run will end up resolving all the remaining explicit dependencies:
- preprocessing the LibriSpeech Dev-Clean dataset to wav;
- building the LoadGen API;
- downloading the PyTorch model.

It's a performance run which should print something like:
```
$ ck run program:speech-recognition-pytorch-loadgen --cmd_key=performance --skip_print_timers
...
Dataset loaded with 4.36 hours. Filtered 1.02 hours. Number of samples: 2513
Running Loadgen test...
Average latency (ms) per query:
7335.167247106061
Median latency (ms):
7391.662108
90 percentile latency (ms):
13347.925176
================================================
MLPerf Results Summary
================================================
SUT name : PySUT
Scenario : Offline
Mode     : Performance
Samples per second: 4.63626
Result is : INVALID
  Min duration satisfied : NO
  Min queries satisfied : Yes
Recommendations:
 * Increase expected QPS so the loadgen pre-generates a larger (coalesced) query.

================================================
Additional Stats
================================================
Min latency (ns)                : 278432559
Max latency (ns)                : 14235613054
Mean latency (ns)               : 7335167247
50.00 percentile latency (ns)   : 7521181269
90.00 percentile latency (ns)   : 13402430910
95.00 percentile latency (ns)   : 13723706550
97.00 percentile latency (ns)   : 14054764438
99.00 percentile latency (ns)   : 14235613054
99.90 percentile latency (ns)   : 14235613054

================================================
Test Parameters Used
================================================
samples_per_query : 66
target_qps : 1
target_latency (ns): 0
max_async_queries : 1
min_duration (ms): 60000
max_duration (ms): 0
min_query_count : 1
max_query_count : 0
qsl_rng_seed : 3133965575612453542
sample_index_rng_seed : 665484352860916858
schedule_rng_seed : 3622009729038561421
accuracy_log_rng_seed : 0
accuracy_log_probability : 0
print_timestamps : false
performance_issue_unique : false
performance_issue_same : false
performance_issue_same_index : 0
performance_sample_count : 2513

No warnings encountered during test.

No errors encountered during test.
Done!

Execution time: 38.735 sec.
```

The above output is the contents of `mlperf_log_summary.txt`, one of the log files generated by LoadGen. All LoadGen log files can be located in the program's temporary directory:
```bash
$ cd `ck find program:speech-recognition-pytorch-loadgen`/tmp && ls -la mlperf_log_*
-rw-r--r-- 1 anton eng      4 Jul  3 18:06 mlperf_log_accuracy.json
-rw-r--r-- 1 anton eng  20289 Jul  3 18:06 mlperf_log_detail.txt
-rw-r--r-- 1 anton eng   1603 Jul  3 18:06 mlperf_log_summary.txt
-rw-r--r-- 1 anton eng 860442 Jul  3 18:06 mlperf_log_trace.json
```

<a name="usage_accuracy"></a>
### Running an accuracy test

```
$ ck run program:speech-recognition-pytorch-loadgen --cmd_key=accuracy --skip_print_timers
...
Dataset loaded with 4.36 hours. Filtered 1.02 hours. Number of samples: 2513
Running Loadgen test...

No warnings encountered during test.

No errors encountered during test.
Running accuracy script: /usr/bin/python3 /disk1/homes/anton/CK-TOOLS/mlperf-inference-dividiti.rnnt/inference/v0.7/speech_recognition/rnnt/accuracy_eval.py --log_dir /disk1/homes/anton/CK/ck-mlperf/program/speech-recognition-pytorch-loadgen/tmp --dataset_dir /homes/anton/CK-TOOLS/dataset-librispeech-preprocessed-to-wav-dev-clean/../ --manifest /homes/anton/CK-TOOLS/dataset-librispeech-preprocessed-to-wav-dev-clean/wav-list.json
Dataset loaded with 4.36 hours. Filtered 1.02 hours. Number of samples: 2513
Word Error Rate: 0.07452253714852645
Done!

Execution time: 502.197 sec.

$ cd `ck find program:speech-recognition-pytorch-loadgen`/tmp && ls -la mlperf_log_*
-rw-r--r-- 1 anton eng  3862427 Jul  3 18:00 mlperf_log_accuracy.json
-rw-r--r-- 1 anton eng    20126 Jul  3 18:00 mlperf_log_detail.txt
-rw-r--r-- 1 anton eng       74 Jul  3 18:00 mlperf_log_summary.txt
-rw-r--r-- 1 anton eng 29738248 Jul  3 18:00 mlperf_log_trace.json
```
