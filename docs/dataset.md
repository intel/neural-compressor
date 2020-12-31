Dataset
==================

User can use LPOT builtin datasets as well as register their own datasets.

## Builtin dataset support list

LPOT supports builtin dataloader on popular industry dataset. Pleaes refer to 'examples/helloworld/tf_example1' about how to config a builtin dataloader.

#### TensorFlow

| Type                  | Parameters                                    |
| :------               | :------                                       |
| dummy                 | shape (list or tuple)                         |
| Imagenet              | root (str)<br>subset (str)<br>                |
| TFRecordDataset       | filenames (str)                               |
| COCORecord            | root (str)                                    |
| style_transfer        | content_path (str)<br>style_path (str)        |


#### PyTorch

| Type                  | Parameters                                      |
| :------               | :------                                         |
| dummy                 | shape (list or tuple)                           |
| ImageNet              | root (str)                                      |
| ImageFolder           | root (str)                                      |
| DatasetFolder         | root (str)                                      |
| Bert                  | dataset (list)<br>task ('classifier' or 'squad')|


#### MXNet

| Type                  | Parameters                                     |
| :------               | :------                                        |
| dummy                 | shape (list or tuple)                          |
| ImageRecordDataset    | root (str)                                     |
| ImageFolderDataset    | root (str)                                     |

#### ONNX Runtime

| Type                  | Parameters                                     |
| :------               | :------                                        |
| dummy                 | shape (list or tuple)                          |
| Imagenet              | root (str)                                     |

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
