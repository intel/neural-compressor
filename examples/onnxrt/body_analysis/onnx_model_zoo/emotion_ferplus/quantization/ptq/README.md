# Evaluate performance of ONNX Runtime(Emotion FERPlus) 
>ONNX runtime quantization is under active development. please use 1.6.0+ to get more quantization support. 

This example load an model converted from [ONNX Model Zoo](https://github.com/onnx/models) and confirm its accuracy and speed based on [Emotion FER dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data). You need to download this dataset yourself.

### Environment
onnx: 1.11.0
onnxruntime: 1.10.0

### Prepare model
Download model from [ONNX Model Zoo](https://github.com/onnx/models)

```shell

wget https://github.com/onnx/models/raw/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-12.onnx
```

### Quantization

```bash
bash run_tuning.sh --input_model=path/to/model  \ # model path as *.onnx
                   --config=emotion_ferplus.yaml \ 
                   --data_path=/path/to/data \
                   --output_model=path/to/save
```

### Performance

```bash
bash run_benchmark.sh --input_model=path/to/model \  # model path as *.onnx
                      --config=emotion_ferplus.yaml \
                      --data_path=/path/to/data \
                      --mode=performance
```
