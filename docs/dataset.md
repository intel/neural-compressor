Dataset
==================
LPOT supports builtin dataloader on popular industry dataset. Pleaes refer to 'examples/helloworld/tf_example1' about how to config a builtin dataloader.

## Dataset support list

### Tensorflow

| Type                  | Parameters                                    |
| :------               | :------                                       |
| dummy                 | shape (list or tuple)                         |
| Imagenet              | root (str)<br>subset (str)<br>                |
| TFRecordDataset       | filenames (str)                               |
| COCORecord            | root (str)                                    |
| style_transfer        | content_path (str)<br>style_path (str)        |


### Pytorch

| Type                  | Parameters                                      |
| :------               | :------                                         |
| dummy                 | shape (list or tuple)                           |
| ImageNet              | root (str)                                      |
| ImageFolder           | root (str)                                      |
| DatasetFolder         | root (str)                                      |
| Bert                  | dataset (list)<br>task ('classifier' or 'squad')|


### Mxnet

| Type                  | Parameters                                     |
| :------               | :------                                        |
| dummy                 | shape (list or tuple)                          |
| ImageRecordDataset    | root (str)                                     |
| ImageFolderDataset    | root (str)                                     |

### Onnxrt

| Type                  | Parameter                                      |
| :------               | :------                                        |
| dummy                 | shape (list or tuple)                          |
| Imagenet              | root (str)                                     |
