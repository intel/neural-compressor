Dataset
==================

User can use LPOT builtin datasets as well as register their own datasets.

## Builtin dataset support list

LPOT supports builtin dataloader on popular industry dataset. Please refer to 'examples/helloworld/tf_example1' about how to config a builtin dataloader.

#### TensorFlow

| Type | Parameters |
| :------ | :------ |
| MNIST | root (str): Root directory of dataset <br> train(bool, default=False): If True, creates dataset from train subset, otherwise from validation subset <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions <br> download(bool, default=True): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again. <br> |
| FashionMNIST | root (str): Root directory of dataset <br> train(bool, default=False): If True, creates dataset from train subset, otherwise from validation subset <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions <br> download(bool, default=True): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again. |
| CIFAR10 | root (str): Root directory of dataset <br> train(bool, default=False): If True, creates dataset from train subset, otherwise from validation subset <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions <br> download(bool, default=True): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again. |
| CIFAR100 | root (str): Root directory of dataset <br> train(bool, default=False): If True, creates dataset from train subset, otherwise from validation subset <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions <br> download(bool, default=True): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again. |
| ImageRecord | root(str): Root directory of dataset <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions |
| ImageFolder | root(str): Root directory of dataset <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions |
| ImagenetRaw | data_path(str): Root directory of dataset <br> image_list(str): data file, record image_names and their labels <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions |
| COCORecord | root(str): Root directory of dataset <br> num_cores(int, default=28):The number of input Datasets to interleave from in parallel <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions |
| COCORaw | root(str): Root directory of dataset <br> img_dir(str, default='val2017'): image file directory <br> anno_dir(str, default='annotations/instances_val2017.json'): annotation file directory <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions |
| dummy | shape(list or tuple):support create multi shape tensors, use list of tuples for each tuple in the list, will create a such size tensor. <br> low(list or float, default=-128.):low out the tensor value range from[0, 1] to [0, low] or [low, 0] if low < 0, if float, will implement all tensors with same low value. <br> high(list or float, default=127.):high the tensor value by add all tensor element value high. If list, length of list should be same with shape list <br> dtype(list or str, default='float32'):support multi tensor dtype setting. If list, length of list should be same with shape list, if str, all tensors will use same dtype. dtype support 'float32', 'float16', 'uint8', 'int8', 'int32', 'int64', 'bool' <br> label(bool, default=False):whether to return 0 as label <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions |
| style_transfer | content_folder(str):Root directory of content images <br> style_folder(str):Root directory of style images <br> crop_ratio(float, default=0.1):cropped ratio to each side <br> resize_shape(tuple, default=(256, 256)):target size of image <br> image_format(str, default='jpg'): target image format <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions |
| TFRecordDataset | root(str): filename of dataset <br> compression_type(str, default=None):compression type, support "" (no compression), "ZLIB", or "GZIP". <br> buffer_size(int, default=None): the number of bytes in the read buffer <br> num_parallel_reads(tint, default=None): the number of files to read in parallel <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions |
| bert | root(str): path of dataset <br> label_file(str): path of label file <br> task(str, default='squad'): task type of model <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions |

#### PyTorch

| Type | Parameters |
| :------ | :------ |
| MNIST | root (str): Root directory of dataset <br> train(bool, default=False): If True, creates dataset from train subset, otherwise from validation subset <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions <br> download(bool, default=True): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again. <br> |
| FashionMNIST | root (str): Root directory of dataset <br> train(bool, default=False): If True, creates dataset from train subset, otherwise from validation subset <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions <br> download(bool, default=True): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again. |
| CIFAR10 | root (str): Root directory of dataset <br> train(bool, default=False): If True, creates dataset from train subset, otherwise from validation subset <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions <br> download(bool, default=True): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again. |
| CIFAR100 | root (str): Root directory of dataset <br> train(bool, default=False): If True, creates dataset from train subset, otherwise from validation subset <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions <br> download(bool, default=True): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again. |
| ImageFolder | root(str): Root directory of dataset <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions |
| ImagenetRaw | data_path(str): Root directory of dataset <br> image_list(str): data file, record image_names and their labels <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions |
| COCORaw | root(str): Root directory of dataset <br> img_dir(str, default='val2017'): image file directory <br> anno_dir(str, default='annotations/instances_val2017.json'): annotation file directory <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions |
| dummy | shape(list or tuple):support create multi shape tensors, use list of tuples for each tuple in the list, will create a such size tensor. <br> low(list or float, default=-128.):low out the tensor value range from[0, 1] to [0, low] or [low, 0] if low < 0, if float, will implement all tensors with same low value. <br> high(list or float, default=127.):high the tensor value by add all tensor element value high. If list, length of list should be same with shape list <br> dtype(list or str, default='float32'):support multi tensor dtype setting. If list, length of list should be same with shape list, if str, all tensors will use same dtype. dtype support 'float32', 'float16', 'uint8', 'int8', 'int32', 'int64', 'bool' <br> label(bool, default=False):whether to return 0 as label <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions |
| bert | dataset(list): list of data <br> task(str): the task of the model, support "classifier", "squad" <br> model_type(str, default='bert'): model type, support 'distilbert', 'bert', 'xlnet', 'xlm' <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions |

#### MXNet

| Type | Parameters |
| :------ | :------ |
| MNIST | root (str): Root directory of dataset <br> train(bool, default=False): If True, creates dataset from train subset, otherwise from validation subset <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions <br> download(bool, default=True): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again. <br> |
| FashionMNIST | root (str): Root directory of dataset <br> train(bool, default=False): If True, creates dataset from train subset, otherwise from validation subset <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions <br> download(bool, default=True): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again. |
| CIFAR10 | root (str): Root directory of dataset <br> train(bool, default=False): If True, creates dataset from train subset, otherwise from validation subset <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions <br> download(bool, default=True): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again. |
| CIFAR100 | root (str): Root directory of dataset <br> train(bool, default=False): If True, creates dataset from train subset, otherwise from validation subset <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions <br> download(bool, default=True): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again. |
| ImageFolder | root(str): Root directory of dataset <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions |
| ImagenetRaw | data_path(str): Root directory of dataset <br> image_list(str): data file, record image_names and their labels <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions |
| COCORaw | root(str): Root directory of dataset <br> img_dir(str, default='val2017'): image file directory <br> anno_dir(str, default='annotations/instances_val2017.json'): annotation file directory <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions |
| dummy | shape(list or tuple):support create multi shape tensors, use list of tuples for each tuple in the list, will create a such size tensor. <br> low(list or float, default=-128.):low out the tensor value range from[0, 1] to [0, low] or [low, 0] if low < 0, if float, will implement all tensors with same low value. <br> high(list or float, default=127.):high the tensor value by add all tensor element value high. If list, length of list should be same with shape list <br> dtype(list or str, default='float32'):support multi tensor dtype setting. If list, length of list should be same with shape list, if str, all tensors will use same dtype. dtype support 'float32', 'float16', 'uint8', 'int8', 'int32', 'int64', 'bool' <br> label(bool, default=False):whether to return 0 as label <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions |


#### ONNXRT

| Type | Parameters |
| :------ | :------ |
| MNIST | root (str): Root directory of dataset <br> train(bool, default=False): If True, creates dataset from train subset, otherwise from validation subset <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions <br> download(bool, default=True): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again. <br> |
| FashionMNIST | root (str): Root directory of dataset <br> train(bool, default=False): If True, creates dataset from train subset, otherwise from validation subset <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions <br> download(bool, default=True): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again. |
| CIFAR10 | root (str): Root directory of dataset <br> train(bool, default=False): If True, creates dataset from train subset, otherwise from validation subset <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions <br> download(bool, default=True): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again. |
| CIFAR100 | root (str): Root directory of dataset <br> train(bool, default=False): If True, creates dataset from train subset, otherwise from validation subset <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions <br> download(bool, default=True): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again. |
| ImageFolder | root(str): Root directory of dataset <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions |
| ImagenetRaw | data_path(str): Root directory of dataset <br> image_list(str): data file, record image_names and their labels <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions |
| COCORaw | root(str): Root directory of dataset <br> img_dir(str, default='val2017'): image file directory <br> anno_dir(str, default='annotations/instances_val2017.json'): annotation file directory <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions |
| dummy | shape(list or tuple):support create multi shape tensors, use list of tuples for each tuple in the list, will create a such size tensor. <br> low(list or float, default=-128.):low out the tensor value range from[0, 1] to [0, low] or [low, 0] if low < 0, if float, will implement all tensors with same low value. <br> high(list or float, default=127.):high the tensor value by add all tensor element value high. If list, length of list should be same with shape list <br> dtype(list or str, default='float32'):support multi tensor dtype setting. If list, length of list should be same with shape list, if str, all tensors will use same dtype. dtype support 'float32', 'float16', 'uint8', 'int8', 'int32', 'int64', 'bool' <br> label(bool, default=False):whether to return 0 as label <br> transform(transform object, default=None):  transform to process input data <br> filter(Filter objects, default=None): filter out examples according to specific conditions |

## User specific dataset

User can register their own dataset as follows:

```python
class Dataset(object):
    def __init__(self, args):
        # init code here

    def __getitem__(self, idx):
        # use idx to get data and label
        return data, label

    def __len__(self):
        return len

```

After defining the dataset class, user can pass it to quantizer.

```python
from lpot.quantization import Quantization
quantizer = Quantization(yaml_file)
dataloader = quantizer.dataloader(dataset) # user can pass more optional args to dataloader such as batch_size and collate_fn
q_model = quantizer(graph, 
                    q_dataloader=dataloader, 
                    eval_func=eval_func)

```
