# Evaluate performance of ONNX Runtime(MobileBERT) 
>ONNX runtime quantization is under active development. please use 1.6.0+ to get more quantization support. 

This example load a language translation model and confirm its accuracy and speed based on [SQuAD]((https://rajpurkar.github.io/SQuAD-explorer/)) task. 

### Environment
onnx: 1.7.0
onnxruntime: 1.6.0+

### Prepare dataset
Download pretrained bert model. We will refer to `vocab.txt` file.

```bash
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
```

Download MLPerf mobilebert model and convert it to onnx model with [tf2onnx](https://github.com/onnx/tensorflow-onnx) tool.

```bash
wget https://github.com/fatihcakirs/mobile_models/blob/main/v0_7/tflite/mobilebert_float_384_20200602.tflite

python -m tf2onnx.convert --opset 11 --tflite mobilebert_float_384_20200602.tflite --output mobilebert_SQuAD.onnx
```


### Evaluating
To evaluate the model, run `main.py` with the path to the model:

```bash
bash run_tuning.sh --input_model=/path/to/model \ # model path as *.onnx
                   --output_model=/path/to/model_tune \
                   --dataset_location=/path/to/SQuAD/dataset \
                   --config=mobilebert.yaml
```


