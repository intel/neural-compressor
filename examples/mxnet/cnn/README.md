# ImageNet Classification

From [MXNet example](https://github.com/apache/incubator-mxnet/blob/v1.x/example/quantization/imagenet_inference.py)


# Quantization with iLiT
```
# ResNet50_v1
python -u imagenet_inference.py \
        --symbol-file=./model/resnet50_v1-symbol.json \
        --param-file=./model/resnet50_v1-0000.params \
        --rgb-mean=123.68,116.779,103.939 \
        --rgb-std=58.393,57.12,57.375 \
        --batch-size=64 \
        --num-inference-batches=500 \
        --dataset=./data/val_256_q90.rec \
        --ctx=cpu 

# Squeezenet1.0
python -u imagenet_inference.py \
        --symbol-file=./model/squeezenet1.0-symbol.json \
        --param-file=./model/squeezenet1.0-0000.params \
        --rgb-mean=123.68,116.779,103.939 \
        --rgb-std=58.393,57.12,57.375 \
        --batch-size=64 \
        --num-inference-batches=500 \
        --dataset=./data/val_256_q90.rec \
        --ctx=cpu 

# MobileNet1.0
python -u imagenet_inference.py \
        --symbol-file=./model/mobilenet1.0-symbol.json \
        --param-file=./model/mobilenet1.0-0000.params \
        --rgb-mean=123.68,116.779,103.939 \
        --rgb-std=58.393,57.12,57.375 \
        --batch-size=64 \
        --num-inference-batches=500 \
        --dataset=./data/val_256_q90.rec \
        --ctx=cpu 

# MobileNetv2_1.0
python -u imagenet_inference.py \
        --symbol-file=./model/mobilenetv2_1.0-symbol.json \
        --param-file=./model/mobilenetv2_1.0-0000.params \
        --rgb-mean=123.68,116.779,103.939 \
        --rgb-std=58.393,57.12,57.375 \
        --batch-size=64 \
        --num-inference-batches=500 \
        --dataset=./data/val_256_q90.rec \
        --ctx=cpu \

# Inceptionv3
python -u imagenet_inference.py \
        --symbol-file=./model/inceptionv3-symbol.json \
        --param-file=./model/inceptionv3-0000.params \
        --rgb-mean=123.68,116.779,103.939 \
        --rgb-std=58.393,57.12,57.375 \
        --batch-size=64 \
        --image-shape 3,299,299 \
        --num-inference-batches=500 \
        --dataset=./data/val_256_q90.rec \
        --ctx=cpu

```
 

# Dependency

```
pip install mxnet-mkl gluoncv


```
