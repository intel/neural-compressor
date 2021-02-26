Transform
====================
LPOT supports builtin preprocessing methods on diffrent framework backend. Pleaes refer to 'examples/helloworld/tf_example1' about how to config a transform in dataloader.

## Transform support list

### TensorFlow

| Type | Parameters |
| :------ | :------ |
| Resize | size (list or int): Size of the result <br> interpolation(str, default='bilinear'):Desired interpolation type, support 'bilinear', 'nearest', 'bicubic' |
| CenterCrop | size (list or int): Size of the result |
| RandomResizedCrop | size (list or int): Size of the result <br> scale (tuple, default=(0.08, 1.0)):range of size of the origin size cropped <br> ratio (tuple, default=(3. / 4., 4. / 3.)): range of aspect ratio of the origin aspect ratio cropped <br> interpolation (str, default='bilinear'):Desired interpolation type, support 'bilinear', 'nearest' |
| Normalize | mean (list, default=[0.0]):means for each channel, if len(mean)=1, mean will be broadcasted to each channel, otherwise its length should be same with the length of image shape <br> std (list, default=[1.0]):stds for each channel, if len(std)=1, std will be broadcasted to each channel, otherwise its length should be same with the length of image shape |
| RandomCrop | size (list or int): Size of the result |
| Compose | transform_list (list of Transform objects):  list of transforms to compose |
| CropResize | x (int):Left boundary of the cropping area <br> y (int):Top boundary of the cropping area <br> width (int):Width of the cropping area <br> height (int):Height of the cropping area <br> size (list or int): resize to new size after cropping <br> interpolation (str, default='bilinear'):Desired interpolation type, support 'bilinear', 'nearest' |
| RandomHorizontalFlip | None |
| RandomVerticalFlip | None |
| DecodeImage | None |
| EncodeJped | None |
| Transpose | perm (list): A permutation of the dimensions of input image |
| CropToBoundingBox | offset_height (int): Vertical coordinate of the top-left corner of the result in the input <br> offset_width (int): Horizontal coordinate of the top-left corner of the result in the input <br> target_height (int): Height of the result <br> target_width (int): Width of the result |
| Cast | dtype(str, default='float32'): A dtype to convert image to |
| ToArray | None |
| Rescale | None |
| AlignImageChannel | dim (int): The channel number of result image |
| ParseDecodeImagenet | None | 
| ResizeCropImagenet | height: Height of the result <br> width: Width of the result <br> random_crop(bool, default=False): whether to random crop <br> resize_side(int, default=256):desired shape after resize operation <br> random_flip_left_right(bool, default=False): whether to random flip left and right <br> mean_value(list, default=[0.0,0.0,0.0]):means for each channel <br> scale(float, default=1.0):std value |
| QuantizedInput | dtype(str): desired image dtype, support 'uint8', 'int8' <br> scale(float, default=None):scaling ratio of each point in image | 
| LabelShift | label_shift(int, default=0): number of label shift |
| BilinearImagenet | height: Height of the result <br> width:Width of the result <br> central_fraction(float, default=0.875):fraction of size to crop <br> mean_value(list, default=[0.0,0.0,0.0]):means for each channel <br> scale(float, default=1.0):std value |
| ParseDecodeCoco | None|
| SquadV1 | label_file(str): path of label file <br> vocab_file(str): path of vocabulary file <br> n_best_size(int, default=20): The total number of n-best predictions to generate in the nbest_predictions.json output file <br> max_seq_length(int, default=384): The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter, than this will be padded <br> max_query_length(int, default=64): The maximum number of tokens for the question. Questions longer than this will be truncated to this length <br> max_answer_length(int, default=30): The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another <br> do_lower_case(bool, default=True): Whether to lower case the input text. Should be True for uncased models and False for cased models <br> doc_stride(int, default=128): When splitting up a long document into chunks, how much stride to take between chunks |

### Pytorch

| Type | Parameters |
| :------ | :------ |
| Resize | size (list or int): Size of the result <br> interpolation(str, default='bilinear'):Desired interpolation type, support 'bilinear', 'nearest', 'bicubic' |
| CenterCrop | size (list or int): Size of the result|
| RandomResizedCrop | size (list or int): Size of the result <br> scale (tuple, default=(0.08, 1.0)):range of size of the origin size cropped <br> ratio (tuple, default=(3. / 4., 4. / 3.)): range of aspect ratio of the origin aspect ratio cropped <br> interpolation (str, default='bilinear'):Desired interpolation type, support 'bilinear', 'nearest', 'bicubic' |
| Normalize | mean (list, default=[0.0]):means for each channel, if len(mean)=1, mean will be broadcasted to each channel, otherwise its length should be same with the length of image shape <br> std (list, default=[1.0]):stds for each channel, if len(std)=1, std will be broadcasted to each channel, otherwise its length should be same with the length of image shape |
| RandomCrop | size (list or int): Size of the result |
| Compose | transform_list (list of Transform objects):  list of transforms to compose |
| RandomHorizontalFlip | None |
| RandomVerticalFlip | None |
| ToTensor | None |
| ToPILImage | None |
| Pad | padding  (int or tuple or list): Padding on each border <br> fill (int or str or tuple): Pixel fill value for constant fill. Default is 0 <br> padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant |
| ColorJitter | brightness (float or tuple of python:float (min, max)): How much to jitter brightness. Default is 0 <br> contrast (float or tuple of python:float (min, max)): How much to jitter contrast. Default is 0 <br> saturation (float or tuple of python:float (min, max)): How much to jitter saturation. Default is 0 <br> hue (float or tuple of python:float (min, max)): How much to jitter hue. Default is 0 |
| ToArray | None |
| CropResize | x (int):Left boundary of the cropping area <br> y (int):Top boundary of the cropping area <br> width (int):Width of the cropping area <br> height (int):Height of the cropping area <br> size (list or int): resize to new size after cropping <br> interpolation (str, default='bilinear'):Desired interpolation type, support 'bilinear', 'nearest', 'bicubic' |

### MXNet

| Type | Parameters |
| :------ | :------ |
| Resize | size (list or int): Size of the result <br> interpolation(str, default='bilinear'):Desired interpolation type, support 'bilinear', 'nearest', 'bicubic' |
| CenterCrop | size (list or int): Size of the result|
| RandomResizedCrop | size (list or int): Size of the result <br> scale (tuple, default=(0.08, 1.0)):range of size of the origin size cropped <br> ratio (tuple, default=(3. / 4., 4. / 3.)): range of aspect ratio of the origin aspect ratio cropped <br> interpolation (str, default='bilinear'):Desired interpolation type, support 'bilinear', 'nearest', 'bicubic' |
| Normalize | mean (list, default=[0.0]):means for each channel, if len(mean)=1, mean will be broadcasted to each channel, otherwise its length should be same with the length of image shape <br> std (list, default=[1.0]):stds for each channel, if len(std)=1, std will be broadcasted to each channel, otherwise its length should be same with the length of image shape |
| RandomCrop | size (list or int): Size of the result |
| Compose | transform_list (list of Transform objects):  list of transforms to compose |
| CropResize | x (int):Left boundary of the cropping area <br> y (int):Top boundary of the cropping area <br> width (int):Width of the cropping area <br> height (int):Height of the cropping area <br> size (list or int): resize to new size after cropping <br> interpolation (str, default='bilinear'):Desired interpolation type, support 'bilinear', 'nearest', 'bicubic' |
| RandomHorizontalFlip | None |
| RandomVerticalFlip | None |
| ToTensor | None |
| Cast | dtype (str, default ='float32') :The target data type |
| Transpose | perm (list): A permutation of the dimensions of input image |

### ONNXRT

| Type | Parameters | 
| :------ | :------ |
| Resize | size (list or int): Size of the result <br> interpolation(str, default='bilinear'):Desired interpolation type, support 'bilinear', 'nearest', 'bicubic' |
| CenterCrop | size (list or int): Size of the result|
| RandomResizedCrop | size (list or int): Size of the result <br> scale (tuple, default=(0.08, 1.0)):range of size of the origin size cropped <br> ratio (tuple, default=(3. / 4., 4. / 3.)): range of aspect ratio of the origin aspect ratio cropped <br> interpolation (str, default='bilinear'):Desired interpolation type, support 'bilinear', 'nearest' |
| Normalize | mean (list, default=[0.0]):means for each channel, if len(mean)=1, mean will be broadcasted to each channel, otherwise its length should be same with the length of image shape <br> std (list, default=[1.0]):stds for each channel, if len(std)=1, std will be broadcasted to each channel, otherwise its length should be same with the length of image shape |
| RandomCrop | size (list or int): Size of the result |
| Compose | transform_list (list of Transform objects):  list of transforms to compose |
| CropResize | x (int):Left boundary of the cropping area <br> y (int):Top boundary of the cropping area <br> width (int):Width of the cropping area <br> height (int):Height of the cropping area <br> size (list or int): resize to new size after cropping <br> interpolation (str, default='bilinear'):Desired interpolation type, support 'bilinear', 'nearest' |
| RandomHorizontalFlip | None |
| RandomVerticalFlip | None |
| ToArray | None |
| Rescale | None |
| AlignImageChannel | dim (int): The channel number of result image |
| ResizeCropImagenet | height: Height of the result <br> width: Width of the result <br> random_crop(bool, default=False): whether to random crop <br> resize_side(int, default=256):desired shape after resize operation <br> random_flip_left_right(bool, default=False): whether to random flip left and right <br> mean_value(list, default=[0.0,0.0,0.0]):means for each channel <br> scale(float, default=1.0):std value |
