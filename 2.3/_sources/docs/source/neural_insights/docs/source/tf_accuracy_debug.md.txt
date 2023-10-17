# Step by step example how to debug accuracy with Neural Insights
1. [Introduction](#introduction)
2. [Preparation](#preparation)
3. [Running the quantization](#running-the-quantization)
4. [Analyzing the result of quantization](#-analyzing-the-result-of-quantization)
5. [Analyzing weight histograms](#-analyzing-weight-histograms)

# Introduction
In this instruction accuracy issue will be debugged using Neural Insights. TensorFlow Inception_v3 model will be used as an example. It will be quantized and the results will be analyzed to find the cause of the accuracy loss.

# Preparation
## Source
First you need to install Intel® Neural Compressor.
```shell
# Install Neural Compressor
git clone https://github.com/intel/neural-compressor.git
cd neural-compressor 
pip install -r requirements.txt 
python setup.py install

# Install Neural Insights
pip install -r neural_insights/requirements.txt
python setup.py install neural_insights
```

## Requirements
```shell
cd examples/tensorflow/image_recognition/tensorflow_models/inception_v3/quantization/ptq
pip install -r requirements.txt
```

## Model
Download pre-trained PB model file.
```shell
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/inceptionv3_fp32_pretrained_model.pb
```

## Prepare the dataset
Download dataset from ImageNet and process the data to TensorFlow Record format.
```shell
cd examples/tensorflow/image_recognition/tensorflow_models/
bash prepare_dataset.sh --output_dir=./inception_v3/quantization/ptq/data --raw_dir=/PATH/TO/img_raw/val/ --subset=validation
bash prepare_dataset.sh --output_dir=./inception_v3/quantization/ptq/data --raw_dir=/PATH/TO/img_raw/train/ --subset=train
```

# Running the quantization
Before applying quantization, modify some code to enable Neural Insights:
1. Set the argument `diagnosis` to be `True` in `PostTrainingQuantConfig` so that Neural Insights will dump weights and activations of quantizable Ops in this model.
2. Delete the `op_name_dict` argument because that’s the answer of our investigation.
```python
conf = PostTrainingQuantConfig(calibration_sampling_size=[50, 100], diagnosis=True)
```
3. Quantize the model with following command:
```shell
bash run_tuning.sh --input_model=/PATH/TO/inceptionv3_fp32_pretrained_model.pb --output_model=./nc_inception_v3.pb --dataset_location=/path/to/ImageNet/
```

The accuracy of this model will decrease a lot if all Ops are quantized to int8 as default strategy:

![accuracy_decrease](./imgs/accuracy_decrease.png)

# Analyzing the result of quantization
Then, if you run quantization, you will find the following table:

![activations_summary](./imgs/activations_summary.png)

The MSE (Mean Square Error) of the Ops’ activation are listed from high to low, there are also min-max values.
Usually, MSE can be referred as one of a typical indexes leading to accuracy loss.

![ops_weights](./imgs/ops_weights.png)

There are also relevant information about Ops’ weights.
Often Op with highest MSE will cause the highest accuracy loss, but it is not always the case.

Experiment with disabling the quantization of some of the Ops with top 5 highest MSE in both tables is not satisfactory, as results show in this example:

![tune_result](./imgs/tune_result.png)

Then weights histograms can be analyzed to find the reason of the accuracy loss.

# Analyzing weight histograms
## Open Neural Insights
```shell
neural_insights
```

Then you will get a webpage address with Neural insights GUI mode. You can find there histograms of weights and activations.
```
Neural Insights Server started.
Open address [...]
```

The weights of Ops are usually distributed in one spike like the following graph:

![weights_histograms](./imgs/weights_histograms.png)

When you click on the Op in the Op list, you can get weight and activation histograms at the bottom of the page.
One of the weights histograms looks different than the examples above.

![weights_histogram](./imgs/weights_histogram.png)

As is shown in the chart, the distribution of weights often concentrates in a small range of min-max values, when the accuracy loss of an Op is tolerable. But in this Op the min-max values of weights are significantly high (range is bigger than [-20, 20]) because of some outliers. The values near zero point, which are the majority, will be mapped to a very small range in int8, leading to a huge accuracy loss. Besides, since the min-max values vary in different channels, the accuracy will decrease without using channel-wise quantization.

Therefore, you can disable this Op:
```python
op_name_dict = {"v0/cg/conv0/conv2d/Conv2D": {"activation": {"dtype": ["fp32"]}}}
conf = PostTrainingQuantConfig(calibration_sampling_size=[50, 100], op_name_dict=op_name_dict)
```

After running quantization again, you can see that accuracy result has increased. The Op that caused accuracy loss was found.

![tune_result2](./imgs/tune_result2.png)
