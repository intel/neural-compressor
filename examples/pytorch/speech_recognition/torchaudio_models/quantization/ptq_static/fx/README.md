Step-by-Step
============

This document describes the step-by-step instructions for reproducing torchaudio pretrained model wav2vec2.0 and Hubert tuning results with Intel® Neural Compressor.

## Prerequisite

### 1. Environment
```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```
```shell
cd examples/pytorch/speech_recognition/torchaudio_models/quantization/ptq_static/fx
pip install -r requirements.txt
```
> Note: Validated PyTorch [Version](/docs/source/installation_guide.md#validated-software-environment).

### 2. Prepare Dataset

Download [LibriSpeech](https://www.openslr.org/resources/12/) Raw audio to dir: /path/to/speech_dataset.  The dir include below folder:

```bash
cd /path/to/speech_dataset
wget http://us.openslr.org/resources/12/test-clean.tar.gz
wget http://us.openslr.org/resources/12/test-other.tar.gz
tar -zxvf test-clean.tar.gz -C ./
tar -zxvf test-other.tar.gz -C ./

ls /path/to/speech_dataset
LibriSpeech  #folder_in_archive
cd LibriSpeech
BOOKS.TXT  CHAPTERS.TXT  dataset_infos.json  LICENSE.TXT  README.TXT  SPEAKERS.TXT  test-clean  test-other
```
# Run

## Hubert

```bash
bash run_quant.sh --dataset_location=/path/to/speech_dataset/LibriSpeech/test-clean --input_model=hubert --output_model=./saved_results
```
```bash
bash run_benchmark.sh --dataset_location=/path/to/speech_dataset/LibriSpeech/test-clean --input_model=hubert --output_model=./saved_results --mode=performance
```


