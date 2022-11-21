transform
==================================================================

.. py:module:: neural_compressor.experimental.data.transforms.transform

.. autoapi-nested-parse::

   Neural Compressor built-in Transforms on multiple framework backends.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   neural_compressor.experimental.data.transforms.transform.Transforms
   neural_compressor.experimental.data.transforms.transform.TensorflowTransforms
   neural_compressor.experimental.data.transforms.transform.MXNetTransforms
   neural_compressor.experimental.data.transforms.transform.PyTorchTransforms
   neural_compressor.experimental.data.transforms.transform.ONNXRTQLTransforms
   neural_compressor.experimental.data.transforms.transform.ONNXRTITTransforms
   neural_compressor.experimental.data.transforms.transform.TRANSFORMS
   neural_compressor.experimental.data.transforms.transform.BaseTransform
   neural_compressor.experimental.data.transforms.transform.TensorflowWrapFunction
   neural_compressor.experimental.data.transforms.transform.TensorflowTransform
   neural_compressor.experimental.data.transforms.transform.PytorchMxnetWrapFunction
   neural_compressor.experimental.data.transforms.transform.PytorchMxnetTransform
   neural_compressor.experimental.data.transforms.transform.ComposeTransform
   neural_compressor.experimental.data.transforms.transform.CropToBoundingBox
   neural_compressor.experimental.data.transforms.transform.MXNetCropToBoundingBox
   neural_compressor.experimental.data.transforms.transform.ONNXRTCropToBoundingBox
   neural_compressor.experimental.data.transforms.transform.TensorflowCropToBoundingBox
   neural_compressor.experimental.data.transforms.transform.ResizeWithRatio
   neural_compressor.experimental.data.transforms.transform.TensorflowResizeWithRatio
   neural_compressor.experimental.data.transforms.transform.Transpose
   neural_compressor.experimental.data.transforms.transform.TensorflowTranspose
   neural_compressor.experimental.data.transforms.transform.MXNetTranspose
   neural_compressor.experimental.data.transforms.transform.PyTorchTranspose
   neural_compressor.experimental.data.transforms.transform.RandomVerticalFlip
   neural_compressor.experimental.data.transforms.transform.TensorflowRandomVerticalFlip
   neural_compressor.experimental.data.transforms.transform.RandomHorizontalFlip
   neural_compressor.experimental.data.transforms.transform.TensorflowRandomHorizontalFlip
   neural_compressor.experimental.data.transforms.transform.ToArray
   neural_compressor.experimental.data.transforms.transform.CastTFTransform
   neural_compressor.experimental.data.transforms.transform.CastONNXTransform
   neural_compressor.experimental.data.transforms.transform.CastPyTorchTransform
   neural_compressor.experimental.data.transforms.transform.CenterCropTFTransform
   neural_compressor.experimental.data.transforms.transform.PaddedCenterCropTransform
   neural_compressor.experimental.data.transforms.transform.ResizeTFTransform
   neural_compressor.experimental.data.transforms.transform.ResizePytorchTransform
   neural_compressor.experimental.data.transforms.transform.RandomCropTFTransform
   neural_compressor.experimental.data.transforms.transform.RandomResizedCropPytorchTransform
   neural_compressor.experimental.data.transforms.transform.RandomResizedCropMXNetTransform
   neural_compressor.experimental.data.transforms.transform.RandomResizedCropTFTransform
   neural_compressor.experimental.data.transforms.transform.NormalizeTFTransform
   neural_compressor.experimental.data.transforms.transform.RescaleKerasPretrainTransform
   neural_compressor.experimental.data.transforms.transform.RescaleTFTransform
   neural_compressor.experimental.data.transforms.transform.RescaleTransform
   neural_compressor.experimental.data.transforms.transform.AlignImageChannelTransform
   neural_compressor.experimental.data.transforms.transform.PyTorchAlignImageChannel
   neural_compressor.experimental.data.transforms.transform.ToNDArrayTransform
   neural_compressor.experimental.data.transforms.transform.ResizeMXNetTransform
   neural_compressor.experimental.data.transforms.transform.ResizeTransform
   neural_compressor.experimental.data.transforms.transform.CropResizeTFTransform
   neural_compressor.experimental.data.transforms.transform.PyTorchCropResizeTransform
   neural_compressor.experimental.data.transforms.transform.MXNetCropResizeTransform
   neural_compressor.experimental.data.transforms.transform.CropResizeTransform
   neural_compressor.experimental.data.transforms.transform.CenterCropTransform
   neural_compressor.experimental.data.transforms.transform.MXNetNormalizeTransform
   neural_compressor.experimental.data.transforms.transform.PyTorchNormalizeTransform
   neural_compressor.experimental.data.transforms.transform.NormalizeTransform
   neural_compressor.experimental.data.transforms.transform.RandomCropTransform
   neural_compressor.experimental.data.transforms.transform.RandomResizedCropTransform
   neural_compressor.experimental.data.transforms.transform.SquadExample
   neural_compressor.experimental.data.transforms.transform.InputFeatures
   neural_compressor.experimental.data.transforms.transform.CollectTransform
   neural_compressor.experimental.data.transforms.transform.TFSquadV1PostTransform
   neural_compressor.experimental.data.transforms.transform.TFModelZooCollectTransform
   neural_compressor.experimental.data.transforms.transform.TFSquadV1ModelZooPostTransform
   neural_compressor.experimental.data.transforms.transform.ParseDecodeVocTransform



Functions
~~~~~~~~~

.. autoapisummary::

   neural_compressor.experimental.data.transforms.transform.transform_registry
   neural_compressor.experimental.data.transforms.transform.get_torchvision_map
   neural_compressor.experimental.data.transforms.transform._compute_softmax
   neural_compressor.experimental.data.transforms.transform._get_best_indexes
   neural_compressor.experimental.data.transforms.transform.get_final_text
   neural_compressor.experimental.data.transforms.transform.read_squad_examples
   neural_compressor.experimental.data.transforms.transform._check_is_max_context
   neural_compressor.experimental.data.transforms.transform.convert_examples_to_features



Attributes
~~~~~~~~~~

.. autoapisummary::

   neural_compressor.experimental.data.transforms.transform.torchvision
   neural_compressor.experimental.data.transforms.transform.torch
   neural_compressor.experimental.data.transforms.transform.tf
   neural_compressor.experimental.data.transforms.transform.mx
   neural_compressor.experimental.data.transforms.transform.cv2
   neural_compressor.experimental.data.transforms.transform.framework_transforms
   neural_compressor.experimental.data.transforms.transform.TENSORFLOW_TRANSFORMS
   neural_compressor.experimental.data.transforms.transform.TENSORFLOW_ITEX_TRANSFORMS
   neural_compressor.experimental.data.transforms.transform.MXNET_TRANSFORMS
   neural_compressor.experimental.data.transforms.transform.PYTORCH_TRANSFORMS
   neural_compressor.experimental.data.transforms.transform.ONNXRT_QL_TRANSFORMS
   neural_compressor.experimental.data.transforms.transform.ONNXRT_IT_TRANSFORMS
   neural_compressor.experimental.data.transforms.transform.registry_transforms
   neural_compressor.experimental.data.transforms.transform.interpolation_map
   neural_compressor.experimental.data.transforms.transform.interpolation_pytorch_map
   neural_compressor.experimental.data.transforms.transform.interpolation_mxnet_map
   neural_compressor.experimental.data.transforms.transform.np_dtype_map


.. py:data:: torchvision
   

   

.. py:data:: torch
   

   

.. py:data:: tf
   

   

.. py:data:: mx
   

   

.. py:data:: cv2
   

   

.. py:class:: Transforms(process, concat_general=True)

   Bases: :py:obj:`object`

   INC supports built-in preprocessing, postprocessing and general methods on different framework backends.

   Transforms base class provides the abstract methods.
   Users can also register their own Transforms classes by inheriting this base class.

   .. py:method:: _get_preprocess()
      :abstractmethod:

      Abstract method to get preprocessing method.


   .. py:method:: _get_postprocess()
      :abstractmethod:

      Abstract method to get postprocess method.


   .. py:method:: _get_general()
      :abstractmethod:

      Abstract method to get general method.



.. py:class:: TensorflowTransforms(process, concat_general=True)

   Bases: :py:obj:`Transforms`

   Tensorflow Transforms subclass.

   .. py:method:: _get_preprocess()

      Tensorflow get preprocess method.

      :returns: a dict including all the registered preprocess methods
      :rtype: preprocess


   .. py:method:: _get_postprocess()

      Tensorflow get postprocess method.

      :returns: a dict including all the registered postprocess methods
      :rtype: postprocess


   .. py:method:: _get_general()

      Tensorflow get general method.

      :returns: a dict including all the registered general methods
      :rtype: general



.. py:class:: MXNetTransforms(process, concat_general=True)

   Bases: :py:obj:`Transforms`

   Mxnet Transforms subclass.

   .. py:method:: _get_preprocess()

      Mxnet get preprocess method.

      :returns: a dict including all the registered preprocess methods
      :rtype: preprocess


   .. py:method:: _get_postprocess()

      Mxnet get postprocess method.

      :returns: a dict including all the registered postprocess methods
      :rtype: postprocess


   .. py:method:: _get_general()

      Mxnet get general method.

      :returns: a dict including all the registered general methods
      :rtype: general



.. py:class:: PyTorchTransforms(process, concat_general=True)

   Bases: :py:obj:`Transforms`

   Pytorch Transforms subclass.

   .. py:method:: _get_preprocess()

      Pytorch get preprocessing method.

      :returns: a dict including all the registered preprocess methods
      :rtype: preprocess


   .. py:method:: _get_postprocess()

      Pytorch get postprocess method.

      :returns: a dict including all the registered postprocess methods
      :rtype: postprocess


   .. py:method:: _get_general()

      Pytorch get general method.

      :returns: a dict including all the registered general methods
      :rtype: general



.. py:class:: ONNXRTQLTransforms(process, concat_general=True)

   Bases: :py:obj:`Transforms`

   Onnxrt_qlinearops Transforms subclass.

   .. py:method:: _get_preprocess()

      Onnxrt_qlinearops get preprocessing method.

      :returns: a dict including all the registered preprocess methods
      :rtype: preprocess


   .. py:method:: _get_postprocess()

      Onnxrt_qlinearops get postprocess method.

      :returns: a dict including all the registered postprocess methods
      :rtype: postprocess


   .. py:method:: _get_general()

      Onnxrt_qlinearops get general method.

      :returns: a dict including all the registered general methods
      :rtype: general



.. py:class:: ONNXRTITTransforms(process, concat_general=True)

   Bases: :py:obj:`Transforms`

   Onnxrt_integerops Transforms subclass.

   .. py:method:: _get_preprocess()

      Onnxrt_integerops get preprocessing method.

      :returns: a dict including all the registered preprocess methods
      :rtype: preprocess


   .. py:method:: _get_postprocess()

      Onnxrt_integerops get postprocess method.

      :returns: a dict including all the registered postprocess methods
      :rtype: postprocess


   .. py:method:: _get_general()

      Onnxrt_integerops get general method.

      :returns: a dict including all the registered general methods
      :rtype: general



.. py:data:: framework_transforms
   

   

.. py:data:: TENSORFLOW_TRANSFORMS
   

   

.. py:data:: TENSORFLOW_ITEX_TRANSFORMS
   

   

.. py:data:: MXNET_TRANSFORMS
   

   

.. py:data:: PYTORCH_TRANSFORMS
   

   

.. py:data:: ONNXRT_QL_TRANSFORMS
   

   

.. py:data:: ONNXRT_IT_TRANSFORMS
   

   

.. py:data:: registry_transforms
   

   

.. py:class:: TRANSFORMS(framework, process)

   Bases: :py:obj:`object`

   Transforms collection class.

   Provide register method to register new Transforms
   and provide __getitem__ method to get Transforms according to Transforms type.

   .. py:method:: __getitem__(transform_type)

      Get Transform according to Transforms type.

      :param transform_type: the value can be preprocess, postprocess or general
      :type transform_type: str

      :returns: the registered Transforms
      :rtype: Transforms


   .. py:method:: register(name, transform_cls)

      Register new Transform according to Transforms type.

      :param name: process name
      :type name: str
      :param transform_cls: process function wrapper class
      :type transform_cls: class



.. py:function:: transform_registry(transform_type, process, framework)

   Class decorator used to register all transform subclasses.

   :param transform_type: Transform registration name
   :type transform_type: str
   :param process: support 3 process including 'preprocess', 'postprocess', 'general'
   :type process: str
   :param framework: support 4 framework including 'tensorflow', 'pytorch', 'mxnet', 'onnxrt'
   :type framework: str
   :param cls: The class of register.
   :type cls: class

   :returns: The class of register.
   :rtype: cls


.. py:class:: BaseTransform

   Bases: :py:obj:`object`

   The base class for transform.

   .. py:method:: __call__(*args, **kwargs)
      :abstractmethod:

      __call__ method is needed when write user specific transform.



.. py:class:: TensorflowWrapFunction(transform_func)

   Bases: :py:obj:`object`

   Tensorflow wrapper function class.

   .. py:method:: __call__(**kwargs)

      __call__ method.

      :returns: TensorflowTransform class



.. py:class:: TensorflowTransform(transform_func, **kwargs)

   Bases: :py:obj:`BaseTransform`

   Tensorflow transform class, the subclass of BaseTransform.

   .. py:method:: __call__(sample)

      __call__ method.

      :returns: a tuple of image and lable which get from tensorflow tranform processing



.. py:class:: PytorchMxnetWrapFunction(transform_func)

   Bases: :py:obj:`object`

   Pytorch and MXNet wrapper function class.

   .. py:method:: __call__(**args)

      __call__ method.

      :returns: PytorchMxnetTransform class



.. py:class:: PytorchMxnetTransform(transform_func)

   Bases: :py:obj:`BaseTransform`

   Pytorch and Mxnet transform class, the subclass of BaseTransform.

   .. py:method:: __call__(sample)

      __call__ method.

      :returns: a tuple of image and lable which get from pytorch or mxnet tranform processing



.. py:data:: interpolation_map
   

   

.. py:data:: interpolation_pytorch_map
   

   

.. py:data:: interpolation_mxnet_map
   

   

.. py:function:: get_torchvision_map(interpolation)

   Get torchvision interpolation map.


.. py:class:: ComposeTransform(transform_list)

   Bases: :py:obj:`BaseTransform`

   Composes several transforms together.

   :param transform_list: list of transforms to compose
   :type transform_list: list of Transform objects

   :returns: tuple of processed image and label
   :rtype: sample (tuple)

   .. py:method:: __call__(sample)

      Call transforms in transform_list.



.. py:class:: CropToBoundingBox(offset_height, offset_width, target_height, target_width)

   Bases: :py:obj:`BaseTransform`

   Crops an image to a specified bounding box.

   :param offset_height: Vertical coordinate of the top-left corner of the result in the input
   :type offset_height: int
   :param offset_width: Horizontal coordinate of the top-left corner of the result in the input
   :type offset_width: int
   :param target_height: Height of the result
   :type target_height: int
   :param target_width: Width of the result
   :type target_width: int

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Call torchvision.transforms.functional.crop.



.. py:class:: MXNetCropToBoundingBox(offset_height, offset_width, target_height, target_width)

   Bases: :py:obj:`CropToBoundingBox`

   Crops an image to a specified bounding box.

   :param offset_height: Vertical coordinate of the top-left corner of the result in the input
   :type offset_height: int
   :param offset_width: Horizontal coordinate of the top-left corner of the result in the input
   :type offset_width: int
   :param target_height: Height of the result
   :type target_height: int
   :param target_width: Width of the result
   :type target_width: int

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Call mx.image.fixed_crop.



.. py:class:: ONNXRTCropToBoundingBox(offset_height, offset_width, target_height, target_width)

   Bases: :py:obj:`CropToBoundingBox`

   Crops an image to a specified bounding box.

   :param offset_height: Vertical coordinate of the top-left corner of the result in the input
   :type offset_height: int
   :param offset_width: Horizontal coordinate of the top-left corner of the result in the input
   :type offset_width: int
   :param target_height: Height of the result
   :type target_height: int
   :param target_width: Width of the result
   :type target_width: int

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Crop the image in sample.



.. py:class:: TensorflowCropToBoundingBox(offset_height, offset_width, target_height, target_width)

   Bases: :py:obj:`CropToBoundingBox`

   Crops an image to a specified bounding box.

   :param offset_height: Vertical coordinate of the top-left corner of the result in the input
   :type offset_height: int
   :param offset_width: Horizontal coordinate of the top-left corner of the result in the input
   :type offset_width: int
   :param target_height: Height of the result
   :type target_height: int
   :param target_width: Width of the result
   :type target_width: int

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Crop the image in sample.



.. py:class:: ResizeWithRatio(min_dim=800, max_dim=1365, padding=False, constant_value=0)

   Bases: :py:obj:`BaseTransform`

   Resize image with aspect ratio and pad it to max shape(optional).

   If the image is padded, the label will be processed at the same time.
   The input image should be np.array.

   :param min_dim: Resizes the image such that its smaller dimension == min_dim
   :type min_dim: int, default=800
   :param max_dim: Ensures that the image longest side doesn't exceed this value
   :type max_dim: int, default=1365
   :param padding: If true, pads image with zeros so its size is max_dim x max_dim
   :type padding: bool, default=False

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Resize the image with ratio in sample.



.. py:class:: TensorflowResizeWithRatio(min_dim=800, max_dim=1365, padding=False, constant_value=0)

   Bases: :py:obj:`BaseTransform`

   Resize image with aspect ratio and pad it to max shape(optional).

   If the image is padded, the label will be processed at the same time.
   The input image should be np.array or tf.Tensor.

   :param min_dim: Resizes the image such that its smaller dimension == min_dim
   :type min_dim: int, default=800
   :param max_dim: Ensures that the image longest side doesn't exceed this value
   :type max_dim: int, default=1365
   :param padding: If true, pads image with zeros so its size is max_dim x max_dim
   :type padding: bool, default=False

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Resize the image with ratio in sample.



.. py:class:: Transpose(perm)

   Bases: :py:obj:`BaseTransform`

   Transpose image according to perm.

   :param perm: A permutation of the dimensions of input image
   :type perm: list

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Transpose the image according to perm in sample.



.. py:class:: TensorflowTranspose(perm)

   Bases: :py:obj:`Transpose`

   Transpose image according to perm.

   :param perm: A permutation of the dimensions of input image
   :type perm: list

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Transpose the image according to perm in sample.



.. py:class:: MXNetTranspose(perm)

   Bases: :py:obj:`Transpose`

   Transpose image according to perm.

   :param perm: A permutation of the dimensions of input image
   :type perm: list

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Transpose the image according to perm in sample.



.. py:class:: PyTorchTranspose(perm)

   Bases: :py:obj:`Transpose`

   Transpose image according to perm.

   :param perm: A permutation of the dimensions of input image
   :type perm: list

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Transpose the image according to perm in sample.



.. py:class:: RandomVerticalFlip

   Bases: :py:obj:`BaseTransform`

   Vertically flip the given image randomly.

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Vertically flip the image in sample.



.. py:class:: TensorflowRandomVerticalFlip

   Bases: :py:obj:`BaseTransform`

   Vertically flip the given image randomly.

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Vertically flip the image in sample.



.. py:class:: RandomHorizontalFlip

   Bases: :py:obj:`BaseTransform`

   Horizontally flip the given image randomly.

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Horizontally flip the image in sample.



.. py:class:: TensorflowRandomHorizontalFlip

   Bases: :py:obj:`BaseTransform`

   Horizontally flip the given image randomly.

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Horizontally flip the image in sample.



.. py:class:: ToArray

   Bases: :py:obj:`BaseTransform`

   Convert PIL Image or NDArray to numpy array.

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Convert image in sample to numpy array.



.. py:data:: np_dtype_map
   

   

.. py:class:: CastTFTransform(dtype='float32')

   Bases: :py:obj:`BaseTransform`

   Convert image to given dtype.

   :param dtype: A dtype to convert image to
   :type dtype: str, default='float32'

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Convert image in sample to given dtype.



.. py:class:: CastONNXTransform(dtype='float32')

   Bases: :py:obj:`BaseTransform`

   Convert image to given dtype.

   :param dtype: A dtype to convert image to
   :type dtype: str, default='float32'

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Convert image in sample to given dtype.



.. py:class:: CastPyTorchTransform(dtype='float32')

   Bases: :py:obj:`BaseTransform`

   Convert image to given dtype.

   :param dtype: A dtype to convert image to
   :type dtype: str, default='float32'

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Convert image in sample to given dtype.



.. py:class:: CenterCropTFTransform(size)

   Bases: :py:obj:`BaseTransform`

   Crops the given image at the center to the given size.

   :param size: Size of the result
   :type size: list or int

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Crops image in sample to the given size.



.. py:class:: PaddedCenterCropTransform(size, crop_padding=0)

   Bases: :py:obj:`BaseTransform`

   Crops the given image at the center to the given size with padding.

   :param size: Size of the result
   :type size: list or int
   :param crop_padding: crop padding number
   :type crop_padding: int

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Crops image in sample to the given size with padding.



.. py:class:: ResizeTFTransform(size, interpolation='bilinear')

   Bases: :py:obj:`BaseTransform`

   Resize the input image to the given size.

   :param size: Size of the result
   :type size: list or int
   :param interpolation: Desired interpolation type,
                         support 'bilinear', 'nearest', 'bicubic'
   :type interpolation: str, default='bilinear'

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Resize the input image in sample to the given size.



.. py:class:: ResizePytorchTransform(size, interpolation='bilinear')

   Bases: :py:obj:`BaseTransform`

   Resize the input image to the given size.

   :param size: Size of the result
   :type size: list or int
   :param interpolation: Desired interpolation type,
                         support 'bilinear', 'nearest', 'bicubic'
   :type interpolation: str, default='bilinear'

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Resize the input image in sample to the given size.



.. py:class:: RandomCropTFTransform(size)

   Bases: :py:obj:`BaseTransform`

   Crop the image at a random location to the given size.

   :param size: Size of the result
   :type size: list or tuple or int

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Crop the image in sample to the given size.



.. py:class:: RandomResizedCropPytorchTransform(size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation='bilinear')

   Bases: :py:obj:`BaseTransform`

   Crop the given image to random size and aspect ratio.

   :param size: Size of the result
   :type size: list or int
   :param scale: range of size of the origin size cropped
   :type scale: tuple or list, default=(0.08, 1.0)
   :param ratio: range of aspect ratio of the origin aspect ratio cropped
   :type ratio: tuple or list, default=(3. / 4., 4. / 3.)
   :param interpolation: Desired interpolation type, support 'bilinear', 'nearest', 'bicubic'
   :type interpolation: str, default='bilinear'

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Crop the image in sample to the random size.



.. py:class:: RandomResizedCropMXNetTransform(size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation='bilinear')

   Bases: :py:obj:`BaseTransform`

   Crop the given image to random size and aspect ratio.

   :param size: Size of the result
   :type size: list or int
   :param scale: range of size of the origin size cropped
   :type scale: tuple or list, default=(0.08, 1.0)
   :param ratio: range of aspect ratio of the origin aspect ratio cropped
   :type ratio: tuple or list, default=(3. / 4., 4. / 3.)
   :param interpolation: Desired interpolation type, support 'bilinear', 'nearest', 'bicubic'
   :type interpolation: str, default='bilinear'

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Crop the image in sample to the random size.



.. py:class:: RandomResizedCropTFTransform(size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation='bilinear')

   Bases: :py:obj:`BaseTransform`

   Crop the given image to random size and aspect ratio.

   :param size: Size of the result
   :type size: list or int
   :param scale: range of size of the origin size cropped
   :type scale: tuple or list, default=(0.08, 1.0)
   :param ratio: range of aspect ratio of the origin aspect ratio cropped
   :type ratio: tuple or list, default=(3. / 4., 4. / 3.)
   :param interpolation: Desired interpolation type, support 'bilinear', 'nearest'
   :type interpolation: str, default='bilinear'

   :returns: tuple of processed image and label

   .. py:method:: get_params(image, scale, ratio)

      Get the image prameters: position, height and width.


   .. py:method:: __call__(sample)

      Crop the image in sample to the random size.



.. py:class:: NormalizeTFTransform(mean=[0.0], std=[1.0], rescale=None)

   Bases: :py:obj:`BaseTransform`

   Normalize a image with mean and standard deviation.

   :param mean: means for each channel, if len(mean)=1, mean will be broadcasted to each channel,
                otherwise its length should be same with the length of image shape
   :type mean: list, default=[0.0]
   :param std: stds for each channel, if len(std)=1, std will be broadcasted to each channel,
               otherwise its length should be same with the length of image shape
   :type std: list, default=[1.0]

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Normalize the image in sample.



.. py:class:: RescaleKerasPretrainTransform(rescale=None)

   Bases: :py:obj:`BaseTransform`

   Scale the values of image to [0,1].

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Scale the values of the image in sample.



.. py:class:: RescaleTFTransform

   Bases: :py:obj:`BaseTransform`

   Scale the values of image to [0,1].

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Scale the values of the image in sample.



.. py:class:: RescaleTransform

   Bases: :py:obj:`BaseTransform`

   Scale the values of image to [0,1].

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Scale the values of the image in sample.



.. py:class:: AlignImageChannelTransform(dim=3)

   Bases: :py:obj:`BaseTransform`

   Align image channel, now just support [H,W]->[H,W,dim], [H,W,4]->[H,W,3] and [H,W,3]->[H,W].

   Input image must be np.ndarray.

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Align channel of the image in sample.



.. py:class:: PyTorchAlignImageChannel(dim=3)

   Bases: :py:obj:`BaseTransform`

   Align image channel, now just support [H,W,4]->[H,W,3] and [H,W,3]->[H,W].

   Input image must be PIL Image.

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Align channel of the image in sample.



.. py:class:: ToNDArrayTransform

   Bases: :py:obj:`BaseTransform`

   Convert np.array to NDArray.

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Convert np.array of the image in sample.



.. py:class:: ResizeMXNetTransform(size, interpolation='bilinear')

   Bases: :py:obj:`BaseTransform`

   Resize the input image to the given size.

   :param size: Size of the result
   :type size: list or int
   :param interpolation: Desired interpolation type,
                         support 'bilinear', 'nearest', 'bicubic'
   :type interpolation: str, default='bilinear'

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Resize the input image in sample to the given size.



.. py:class:: ResizeTransform(size, interpolation='bilinear')

   Bases: :py:obj:`BaseTransform`

   Resize the input image to the given size.

   :param size: Size of the result
   :type size: list or int
   :param interpolation: Desired interpolation type,
   :type interpolation: str, default='bilinear'
   :param support 'bilinear':
   :param 'nearest':
   :param 'bicubic'.:

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Resize the input image in sample to the given size.



.. py:class:: CropResizeTFTransform(x, y, width, height, size, interpolation='bilinear')

   Bases: :py:obj:`BaseTransform`

   Crop the input image with given location and resize it.

   :param x: Left boundary of the cropping area
   :type x: int
   :param y: Top boundary of the cropping area
   :type y: int
   :param width: Width of the cropping area
   :type width: int
   :param height: Height of the cropping area
   :type height: int
   :param size: resize to new size after cropping
   :type size: list or int
   :param interpolation: Desired interpolation type,
                         support 'bilinear', 'nearest', 'bicubic'
   :type interpolation: str, default='bilinear'

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Resize the input image in sample with given location.



.. py:class:: PyTorchCropResizeTransform(x, y, width, height, size, interpolation='bilinear')

   Bases: :py:obj:`BaseTransform`

   Crop the input image with given location and resize it.

   :param x: Left boundary of the cropping area
   :type x: int
   :param y: Top boundary of the cropping area
   :type y: int
   :param width: Width of the cropping area
   :type width: int
   :param height: Height of the cropping area
   :type height: int
   :param size: resize to new size after cropping
   :type size: list or int
   :param interpolation: Desired interpolation type,
                         support 'bilinear', 'nearest', 'bicubic'
   :type interpolation: str, default='bilinear'

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Resize the input image in sample with given location.



.. py:class:: MXNetCropResizeTransform(x, y, width, height, size, interpolation='bilinear')

   Bases: :py:obj:`BaseTransform`

   Crop the input image with given location and resize it.

   :param x: Left boundary of the cropping area
   :type x: int
   :param y: Top boundary of the cropping area
   :type y: int
   :param width: Width of the cropping area
   :type width: int
   :param height: Height of the cropping area
   :type height: int
   :param size: resize to new size after cropping
   :type size: list or int
   :param interpolation: Desired interpolation type,
                         support 'bilinear', 'nearest', 'bicubic'
   :type interpolation: str, default='bilinear'

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Resize the input image in sample with given location.



.. py:class:: CropResizeTransform(x, y, width, height, size, interpolation='bilinear')

   Bases: :py:obj:`BaseTransform`

   Crop the input image with given location and resize it.

   :param x: Left boundary of the cropping area
   :type x: int
   :param y: Top boundary of the cropping area
   :type y: int
   :param width: Width of the cropping area
   :type width: int
   :param height: Height of the cropping area
   :type height: int
   :param size: resize to new size after cropping
   :type size: list or int
   :param interpolation: Desired interpolation type,
                         support 'bilinear', 'nearest', 'bicubic'
   :type interpolation: str, default='bilinear'

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Crop the input image in sample with given location.



.. py:class:: CenterCropTransform(size)

   Bases: :py:obj:`BaseTransform`

   Crops the given image at the center to the given size.

   :param size: Size of the result
   :type size: list or int

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Crop the input image in sample at the center to the given size.



.. py:class:: MXNetNormalizeTransform(mean=[0.0], std=[1.0])

   Bases: :py:obj:`BaseTransform`

   Normalize a image with mean and standard deviation.

   :param mean: means for each channel, if len(mean)=1, mean will be broadcasted to each channel,
                otherwise its length should be same with the length of image shape
   :type mean: list, default=[0.0]
   :param std: stds for each channel, if len(std)=1, std will be broadcasted to each channel,
               otherwise its length should be same with the length of image shape
   :type std: list, default=[1.0]

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Normalize the image in sample.



.. py:class:: PyTorchNormalizeTransform(mean=[0.0], std=[1.0])

   Bases: :py:obj:`MXNetNormalizeTransform`

   Normalize a image with mean and standard deviation.

   :param mean: means for each channel, if len(mean)=1, mean will be broadcasted to each channel,
                otherwise its length should be same with the length of image shape
   :type mean: list, default=[0.0]
   :param std: stds for each channel, if len(std)=1, std will be broadcasted to each channel,
               otherwise its length should be same with the length of image shape
   :type std: list, default=[1.0]

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Normalize the image in sample.



.. py:class:: NormalizeTransform(mean=[0.0], std=[1.0])

   Bases: :py:obj:`BaseTransform`

   Normalize a image with mean and standard deviation.

   :param mean: means for each channel, if len(mean)=1, mean will be broadcasted to each channel,
                otherwise its length should be same with the length of image shape
   :type mean: list, default=[0.0]
   :param std: stds for each channel, if len(std)=1, std will be broadcasted to each channel,
               otherwise its length should be same with the length of image shape
   :type std: list, default=[1.0]

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Normalize the image in sample.



.. py:class:: RandomCropTransform(size)

   Bases: :py:obj:`BaseTransform`

   Crop the image at a random location to the given size.

   :param size: Size of the result
   :type size: list or tuple or int

   :returns: tuple of processed image and label

   .. py:method:: __call__(sample)

      Crop the image in sample to the given size.



.. py:class:: RandomResizedCropTransform(size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation='bilinear')

   Bases: :py:obj:`BaseTransform`

   Crop the given image to random size and aspect ratio.

   :param size: Size of the result
   :type size: list or int
   :param scale: range of size of the origin size cropped
   :type scale: tuple or list, default=(0.08, 1.0)
   :param ratio: range of aspect ratio of the origin aspect ratio cropped
   :type ratio: tuple or list, default=(3. / 4., 4. / 3.)
   :param interpolation: Desired interpolation type, support 'bilinear', 'nearest'
   :type interpolation: str, default='bilinear'

   :returns: tuple of processed image and label

   .. py:method:: get_params(image, scale, ratio)

      Get the image prameters: position, height and width.


   .. py:method:: __call__(sample)

      Crop the image in sample to random size.



.. py:function:: _compute_softmax(scores)

   Compute softmax probability over raw logits.


.. py:function:: _get_best_indexes(logits, n_best_size)

   Get the n-best logits from a list.


.. py:function:: get_final_text(pred_text, orig_text, do_lower_case)

   Project the tokenized prediction back to the original text.


.. py:class:: SquadExample(qas_id, question_text, doc_tokens, orig_answer_text=None, start_position=None, end_position=None, is_impossible=False)

   Bases: :py:obj:`object`

   A single training/test example for simple sequence classification.

   For examples without an answer, the start and end position are -1.


.. py:class:: InputFeatures(unique_id, example_index, doc_span_index, tokens, token_to_orig_map, token_is_max_context, input_ids, input_mask, segment_ids, start_position=None, end_position=None, is_impossible=None)

   Bases: :py:obj:`object`

   A single set of features of data.


.. py:function:: read_squad_examples(input_file)

   Read a SQuAD json file into a list of SquadExample.


.. py:function:: _check_is_max_context(doc_spans, cur_span_index, position)

   Check if this is the 'max context' doc span for the token.


.. py:function:: convert_examples_to_features(examples, tokenizer, max_seq_length, doc_stride, max_query_length, output_fn)

   Load a data file into a list of `InputBatch`s.


.. py:class:: CollectTransform(length=10833)

   Bases: :py:obj:`BaseTransform`

   Postprocess the predictions, collect data.

   .. py:method:: __call__(sample)

      Collect postprocess data.



.. py:class:: TFSquadV1PostTransform(label_file, vocab_file, n_best_size=20, max_seq_length=384, max_query_length=64, max_answer_length=30, do_lower_case=True, doc_stride=128)

   Bases: :py:obj:`BaseTransform`

   Postprocess the predictions of bert on SQuAD.

   :param label_file: path of label file
   :type label_file: str
   :param vocab_file: path of vocabulary file
   :type vocab_file: str
   :param n_best_size: The total number of n-best predictions to generate in nbest_predictions.json
   :type n_best_size: int, default=20
   :param max_seq_length: The maximum total input sequence length after WordPiece tokenization.
                          Sequences longer than this will be truncated, shorter than this will be padded
   :type max_seq_length: int, default=384
   :param max_query_length: The maximum number of tokens for the question.
                            Questions longer than this will be truncated to this length
   :type max_query_length: int, default=64
   :param max_answer_length: The maximum length of an answer that can be generated. This is needed because
                             the start and end predictions are not conditioned on one another
   :type max_answer_length: int, default=30
   :param do_lower_case: Whether to lower case the input text.
                         Should be True for uncased models and False for cased models
   :type do_lower_case: bool, default=True
   :param doc_stride: When splitting up a long document into chunks,
                      how much stride to take between chunks
   :type doc_stride: int, default=128

   :returns: tuple of processed prediction and label

   .. py:method:: process_result(results)

      Get the processed results.


   .. py:method:: get_postprocess_result(sample)

      Get the post processed results.


   .. py:method:: __call__(sample)

      Call the get_postprocess_result.



.. py:class:: TFModelZooCollectTransform(length=10833)

   Bases: :py:obj:`CollectTransform`

   Postprocess the predictions of model zoo, collect data.

   .. py:method:: __call__(sample)

      Collect postprocess data.



.. py:class:: TFSquadV1ModelZooPostTransform(label_file, vocab_file, n_best_size=20, max_seq_length=384, max_query_length=64, max_answer_length=30, do_lower_case=True, doc_stride=128)

   Bases: :py:obj:`TFSquadV1PostTransform`

   Postprocess the predictions of bert on SQuADV1.1.

   See class TFSquadV1PostTransform for more details

   .. py:method:: __call__(sample)

      Collect data and get postprocess results.



.. py:class:: ParseDecodeVocTransform

   Bases: :py:obj:`BaseTransform`

   Parse features in Example proto.

   :returns: tuple of parsed image and labels

   .. py:method:: __call__(sample)

      Parse decode voc.


