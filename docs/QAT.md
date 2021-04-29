# QAT

## Design

At its core, QAT simulates low-precision inference-time computation in the forward pass of the training process. With QAT, all weights and activations are "fake quantized" during both the forward and backward passes of training: that is, float values are rounded to mimic int8 values, but all computations are still done with floating point numbers. Thus, all the weight adjustments during training are made while "aware" of the fact that the model will ultimately be quantized; after quantizing, therefore, this method will usually yield higher accuracy than either dynamic quantization or post-training static quantization.

The overall workflow for actually performing QAT is very similar to Post-training static quantization (PTQ):

* We can use the same model as PTQ; no additional preparation is needed for quantization-aware training.
* We need to use a qconfig specifying what kind of fake-quantization is to be inserted after weights and activations, instead of specifying observers.

## Usage

### MobileNetV2 Model Architecture

Refer to the [PTQ Model Usage](PTQ.md#mobilenetv2-model-architecture).

### Helper Functions

Refer to [PTQ Helper Functions](PTQ.md#helper-functions).

### QAT

First, define a training function:

```python
def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
    model.train()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')

    cnt = 0
    for image, target in data_loader:
        start_time = time.time()
        print('.', end = '')
        cnt += 1
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        avgloss.update(loss, image.size(0))
        if cnt >= ntrain_batches:
            print('Loss', avgloss.avg)

            print('Training: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
            return

    print('Full imagenet train set:  * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=top1, top5=top5))
    return
```
Fuse modules as PTQ:
```python
model.fuse_model()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001)
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
```
Finally, prepare_qat performs the "fake quantization", preparing the model for quantization-aware training:
```python
torch.quantization.prepare_qat(model, inplace=True)
```
Training a quantized model with high accuracy requires accurate modeling of numerics at inference. For quantization-aware training, therefore, modify the training loop by doing the following:

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

Here, we just perform quantization-aware training for a small number of epochs. Nevertheless, quantization-aware training yields an accuracy of over 71% on the entire imagenet dataset, which is close to the floating point accuracy of 71.9%.

More on quantization-aware training:

* QAT is a super-set of post-training quantization techniques that allows for more debugging. For example, we can analyze if the accuracy of the model is limited by weight or activation quantization.
* We can simulate the accuracy of a quantized model in floating points since we are using fake-quantization to model the numerics of actual quantized arithmetic.
* We can easily mimic post-training quantization.

Intel® Low Precision Optimization Tool can support QAT calibration for
PyTorch models. Refer to the [QAT model](https://github.com/intel/lpot/tree/master/examples/pytorch/image_recognition/imagenet/cpu/qat/README.md) for step-by-step tuning.

### Example
View a [QAT example of PyTorch resnet50](/examples/pytorch/image_recognition/imagenet/cpu/qat).


