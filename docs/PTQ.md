# PTQ

## Design

Post-training static quantization (PTQ) involves not just converting the weights from **float** to **int**, but also first feeding batches of data through the network and computing the resulting distributions of the different activations (specifically, this is done by inserting observer modules at different points that record this data). These distributions are then used to determine specifically how the different activations should be quantized at inference time (a simple technique would be to simply divide the entire range of activations into 256 levels, but we support more sophisticated methods as well). This additional step allows us to pass quantized values between operations instead of converting these values to floats - and then back to ints - between every operation, resulting in a significant speed-up.

## Usage
```
num_calibration_batches = 10

myModel = load_model(saved_model_dir + float_model_file).to('cpu')
myModel.eval()

# Fuse Conv, bn and relu
myModel.fuse_model()

# Specify quantization configuration
# Start with simple min/max range estimation and per-tensor quantization of weights
myModel.qconfig = torch.quantization.default_qconfig
print(myModel.qconfig)
torch.quantization.prepare(myModel, inplace=True)

# Calibrate first
print('Post Training Quantization Prepare: Inserting Observers')
print('\n Inverted Residual Block:After observer insertion \n\n', myModel.features[1].conv)

# Calibrate with the training set
evaluate(myModel, criterion, data_loader, neval_batches=num_calibration_batches)
print('Post Training Quantization: Calibration done')

# Convert to quantized model
torch.quantization.convert(myModel, inplace=True)
print('Post Training Quantization: Convert done')
print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',myModel.features[1].conv)

print("Size of model after quantization")
print_size_of_model(myModel)

top1, top5 = evaluate(myModel, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
```
Output:
```
QConfig(activation=functools.partial(<class 'torch.quantization.observer.MinMaxObserver'>, reduce_range=True), weight=functools.partial(<class 'torch.quantization.observer.MinMaxObserver'>, dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))
Post Training Quantization Prepare: Inserting Observers

 Inverted Residual Block:After observer insertion

 Sequential(
  (0): ConvBNReLU(
    (0): ConvReLU2d(
      (0): Conv2d(
        32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32
        (activation_post_process): MinMaxObserver(min_val=tensor([]), max_val=tensor([]))
      )
      (1): ReLU(
        (activation_post_process): MinMaxObserver(min_val=tensor([]), max_val=tensor([]))
      )
    )
    (1): Identity()
    (2): Identity()
  )
  (1): Conv2d(
    32, 16, kernel_size=(1, 1), stride=(1, 1)
    (activation_post_process): MinMaxObserver(min_val=tensor([]), max_val=tensor([]))
  )
  (2): Identity()
)
..........Post Training Quantization: Calibration done
Post Training Quantization: Convert done

Inverted Residual Block: After fusion and quantization, note fused modules:
Sequential(
 (0): ConvBNReLU(
   (0): QuantizedConvReLU2d(32, 32, kernel_size=(3, 3), stride=(1, 1), scale=0.15583468973636627, zero_point=0, padding=(1, 1), groups=32)
    (1): Identity()
    (2): Identity()
  )
  (1): QuantizedConv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), scale=0.19358506798744202, zero_point=74)
  (2): Identity()
)
Size of model after quantization
Size (MB): 3.631847
..........Evaluation accuracy on 300 images, 67.67
```
For this quantized model, we see a significantly lower accuracy of just ~62% on these same 300 images. Nevertheless, we did reduce the size of our model down to just under 3.6 MB, almost a 4x decrease.

In addition, we can significantly improve on the accuracy simply by using a different quantization configuration. We repeat the same exercise with the recommended configuration for quantizing for x86 architectures. This configuration does the following:

* Quantizes weights on a per-channel basis
* Uses a histogram observer that collects a histogram of activations and then picks quantization parameters in an optimal manner.
```
per_channel_quantized_model = load_model(saved_model_dir + float_model_file)
per_channel_quantized_model.eval()
per_channel_quantized_model.fuse_model()
per_channel_quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
print(per_channel_quantized_model.qconfig)

torch.quantization.prepare(per_channel_quantized_model, inplace=True)
evaluate(per_channel_quantized_model,criterion, data_loader, num_calibration_batches)
torch.quantization.convert(per_channel_quantized_model, inplace=True)
top1, top5 = evaluate(per_channel_quantized_model, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
torch.jit.save(torch.jit.script(per_channel_quantized_model), saved_model_dir + scripted_quantized_model_file)
```
Output:
```
QConfig(activation=functools.partial(<class 'torch.quantization.observer.HistogramObserver'>, reduce_range=True), weight=functools.partial(<class 'torch.quantization.observer.PerChannelMinMaxObserver'>, dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
....................Evaluation accuracy on 300 images, 76.67
```
Changing just this quantization configuration method resulted in an increase of the accuracy to over 76%!

## Example
View a [PTQ example of PyTorch resnet50](../examples/pytorch/image_recognition/imagenet/cpu/ptq/README.md).


