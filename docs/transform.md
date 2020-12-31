Transform
====================
LPOT supports builtin preprocessing methods on diffrent framework backend. Pleaes refer to 'examples/helloworld/tf_example1' about how to config a transform in dataloader.

## Transform support list

### Tensorflow

| Type                | Parameters                            | Inputs        | Outputs      |
| :------             | :------                               | :------       | :------      |
| Resize              | size (list)                            | image, label  | image, label |
| CenterCrop          | central_fraction (float)               | image, label  | image, label |
| RandomResizedCrop   | size (list)                            | image, label  | image, label |
| Normalize           | mean (list)<br>std (list)               | image, label  | image, label |
| RandomCrop          | size (list)                            | image, label  | image, label |
| Compose             | None                                  | image, label  | image, label |
| CropAndResize       | boxes (list)<br>box_indices (list)<br>crop_size (list)| image, label  | image, label |
| RandomHorizontalFlip| None                                  | image, label  | image, label |
| RandomVerticalFlip  | None                                  | image, label  | image, label |
| DecodeImage         | None                                  | contents(str), label| image, label |
| EncodeJped          | None                                  | image, label  | Tensor(str), label|
| Transpose           | perm (list)                            | image, label  | image, label |
| CropToBoundingBox   | offset_height (int)<br>offset_width (int)<br>target_height (int)<br>target_width (int)| image, label  | image, label |
| ConvertImageDtype   | dtype (Dtype)                          | image, label  | image, label |


### Pytorch

| Type                  | Parameters                | Inputs        | Outputs      |
| ------                | :------                   | :------       | :------      |
| Resize                | size (list or int)         | image, label  | image, label |
| CenterCrop            | size (list or int)         | image, label  | image, label |
| RandomResizedCrop     | size (list or int)         | image, label  | image, label |
| Normalize             | mean (list)<br>std (list)   | image, label  | image, label |
| RandomCrop            | size (list or int)         | image, label  | image, label |
| Compose               | None                      | transform_list| None         |
| RandomHorizontalFlip  | None                      | image, label  | image, label |
| RandomVerticalFlip    | None                      | image, label  | image, label |
| ToTensor              | None                      | image, label  | image, label |
| ToPILImage            | None                      | image, label  | image, label |
| Pad                   | padding (int or tuple or list)| image, label  | image, label |
| ColorJitter           | brightness (float or tuple)<br>contrast (float or tuple)<br>saturation (float or tuple)<br>hue (float or tuple)| image, label  | image, label |

### Mxnet

| Type                  | Parameters             | Inputs        | Outputs      |
| ------                | :------                | :------       | :------      |
| Resize                | size (tuple or int)     | image, label  | image, label |
| CenterCrop            | size (tuple or int)     | image, label  | image, label |
| RandomResizedCrop     | size (tuple or int)     | image, label  | image, label |
| Normalize             | mean (tuple or float)<br>std (tuple or float) | image, label | image, label |
| RandomCrop            | size (tuple or int)     | image, label  | image, label |
| Compose               | None                   | transform_list| None         |
| CropResize            | x (int)<br>y (int)<br>w (int)<br>h (int)   | image, label | image, label |
| RandomHorizontalFlip  | None                   | image, label  | image, label |
| RandomVerticalFlip    | None                   | image, label  | image, label |
| ToTensor              | None                   | image, label  | image, label |
| Cast                  | dtype (str)             | image, label  | image, label |

### Onnxrt

| Type                  | Parameters              | Inputs        | Outputs      |
| ------                | :------                 | :------       | :------      |
| Resize                | size (list)              | image, label  | image, label |
| CenterCrop            | size (list)              | image, label  | image, label |
| RandomResizedCrop     | size (list)              | image, label  | image, label |
| Normalize             | mean (list)<br>std (list) | image, label  | image, label |
| RandomCrop            | size (list)              | image, label  | image, label |
| Compose               | None                    | transform_list| None         |
| ImageTypeParse        | None                    | image, label  | image, label |
