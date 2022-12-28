# Quantization-aware Training

## Design

Quantization-aware training (QAT) simulates low-precision inference-time computation in the forward pass of the training process. With QAT, all weights and activations are "fake quantized" during both the forward and backward passes of training: that is, float values are rounded to mimic int8 values, but all computations are still done with floating point numbers. Thus, all the weight adjustments during training are made while "aware" of the fact that the model will ultimately be quantized; after quantizing, therefore, this method will usually yield higher accuracy than either dynamic quantization or post-training static quantization.

<img src="./imgs/fake_quant.png" width=700 height=433 alt="fake quantize">

## Usage

First, define a training function as below.
accuracy is in the 

```python
def training_func_for_nc(model):
    epochs = 8
    iters = 30
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    for nepoch in range(epochs):
        model.train()
        cnt = 0
        for image, target in train_loader:
            print('.', end='')
            cnt += 1
            output = model(image)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if cnt >= iters:
                break
        if nepoch > 3:
            # Freeze quantizer parameters
            model.apply(torch.quantization.disable_observer)
        if nepoch > 2:
            # Freeze batch norm mean and variance estimates
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    return model
```
Fuse modules:
```python
model.fuse_model()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001)
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
```
Finally, prepare_qat performs the "fake quantization", preparing the model for quantization-aware training, this function already be implemented as a hook :
```python
torch.quantization.prepare_qat(model, inplace=True)
```
Training a quantized model with high accuracy requires accurate modeling of numerics at inference. INC does the training loop by following:
* Switch batch norm to use running mean and variance towards the end of training to better match inference numerics.
* Freeze the quantizer parameters (scale and zero-point) and fine tune the weights.

```python
num_train_batches = 20
# Train and check accuracy after each epoch
for nepoch in range(8):
    train_one_epoch(qat_model, criterion, optimizer, data_loader, torch.device('cpu'), num_train_batches)
    if nepoch > 3:
        # Freeze quantizer parameters
        qat_model.apply(torch.quantization.disable_observer)
    if nepoch > 2:
        # Freeze batch norm mean and variance estimates
        qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    # Check the accuracy after each epoch
    quantized_model = torch.quantization.convert(qat_model.eval(), inplace=False)
    quantized_model.eval()
    top1, top5 = evaluate(quantized_model,criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Epoch %d :Evaluation accuracy on %d images, %2.2f'%(nepoch, num_eval_batches * eval_batch_size, top1.avg))
```

When using QAT in INC, you just need to use these APIs: 
```python
from neural_compressor.experimental import Quantization, common
quantizer = Quantization("./conf.yaml")
quantizer.model = common.Model(model)
quantizer.q_func = training_func_for_nc
quantizer.eval_dataloader = val_loader
q_model = quantizer.fit()
```

The quantizer.fit() function will return a best quantized model during timeout constrain.
<br>
The yaml define example: [The yaml example](/examples/pytorch/image_recognition/torchvision_models/quantization/qat/fx)

Here, we just perform quantization-aware training for a small number of epochs. Nevertheless, quantization-aware training yields an accuracy of over 71% on the entire imagenet dataset, which is close to the floating point accuracy of 71.9%.

More on quantization-aware training:

* QAT is a super-set of post-training quantization techniques that allows for more debugging. For example, we can analyze if the accuracy of the model is limited by weight or activation quantization.
* We can simulate the accuracy of a quantized model in floating points since we are using fake-quantization to model the numerics of actual quantized arithmetic.
* We can easily mimic post-training quantization.

### Examples
For related examples, please refer to the [QAT models](/examples/README.md).

