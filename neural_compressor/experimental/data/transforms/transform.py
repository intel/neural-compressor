#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import collections
from abc import abstractmethod
from neural_compressor.utils.utility import LazyImport, singleton
from neural_compressor.utils import logger

torchvision = LazyImport('torchvision')
torch = LazyImport('torch')
tf = LazyImport('tensorflow')
mx = LazyImport('mxnet')
cv2 = LazyImport('cv2')

class Transforms(object):
    def __init__(self, process, concat_general=True):
        transform_map = {"preprocess": self._get_preprocess,
                         "postprocess": self._get_postprocess,
                         "general": self._get_general, }
        self.transforms = transform_map[process]()
        # if set True users can use general transform in both preprocess or postprocess
        if concat_general:
            self.transforms.update(transform_map['general']())

    @abstractmethod
    def _get_preprocess(self):
        raise NotImplementedError

    @abstractmethod
    def _get_postprocess(self):
        raise NotImplementedError

    @abstractmethod
    def _get_general(self):
        raise NotImplementedError


class TensorflowTransforms(Transforms):

    def _get_preprocess(self):
        preprocess = {
            "DecodeImage": TensorflowWrapFunction(tf.io.decode_jpeg),
            "EncodeJpeg": TensorflowWrapFunction(tf.io.encode_jpeg),
        }
        # update the registry transforms
        preprocess.update(TENSORFLOW_TRANSFORMS["preprocess"])
        return preprocess

    def _get_postprocess(self):
        postprocess = {}
        postprocess.update(TENSORFLOW_TRANSFORMS["postprocess"])
        return postprocess

    def _get_general(self):
        general = {}
        general.update(TENSORFLOW_TRANSFORMS["general"])
        return general


class MXNetTransforms(Transforms):
    def _get_preprocess(self):
        preprocess = {
            'ToTensor': PytorchMxnetWrapFunction(
                mx.gluon.data.vision.transforms.ToTensor),
            'CenterCrop': PytorchMxnetWrapFunction(
                mx.gluon.data.vision.transforms.CenterCrop),
            'RandomHorizontalFlip': PytorchMxnetWrapFunction(
                mx.gluon.data.vision.transforms.RandomFlipLeftRight),
            'RandomVerticalFlip': PytorchMxnetWrapFunction(
                mx.gluon.data.vision.transforms.RandomFlipTopBottom),
        }
        preprocess.update(MXNET_TRANSFORMS["preprocess"])
        return preprocess

    def _get_postprocess(self):
        postprocess = {}
        postprocess.update(MXNET_TRANSFORMS["postprocess"])
        return postprocess

    def _get_general(self):
        general = {
            'Compose': mx.gluon.data.vision.transforms.Compose,
            'Cast': PytorchMxnetWrapFunction(
                mx.gluon.data.vision.transforms.Cast),
        }
        general.update(MXNET_TRANSFORMS["general"])
        return general


class PyTorchTransforms(Transforms):
    def _get_preprocess(self):
        preprocess = {
            "ToTensor": PytorchMxnetWrapFunction(
                torchvision.transforms.ToTensor),
            "ToPILImage": PytorchMxnetWrapFunction(
                torchvision.transforms.ToPILImage),
            "CenterCrop": PytorchMxnetWrapFunction(
                torchvision.transforms.CenterCrop),
            "RandomCrop": PytorchMxnetWrapFunction(
                torchvision.transforms.RandomCrop),
            "RandomHorizontalFlip": PytorchMxnetWrapFunction(
                torchvision.transforms.RandomHorizontalFlip),
            "RandomVerticalFlip": PytorchMxnetWrapFunction(
                torchvision.transforms.RandomVerticalFlip),
            "Pad": PytorchMxnetWrapFunction(
                torchvision.transforms.Pad),
            "ColorJitter": PytorchMxnetWrapFunction(
                torchvision.transforms.ColorJitter),
        }
        preprocess.update(PYTORCH_TRANSFORMS["preprocess"])
        return preprocess

    def _get_postprocess(self):
        postprocess = {}
        postprocess.update(PYTORCH_TRANSFORMS["postprocess"])
        return postprocess

    def _get_general(self):
        general = {
            "Compose": torchvision.transforms.Compose,
        }
        general.update(PYTORCH_TRANSFORMS["general"])
        return general

class ONNXRTQLTransforms(Transforms):
    def _get_preprocess(self):
        preprocess = {}
        preprocess.update(ONNXRT_QL_TRANSFORMS["preprocess"])
        return preprocess

    def _get_postprocess(self):
        postprocess = {}
        postprocess.update(ONNXRT_QL_TRANSFORMS["postprocess"])
        return postprocess

    def _get_general(self):
        general = {}
        general.update(ONNXRT_QL_TRANSFORMS["general"])
        return general


class ONNXRTITTransforms(Transforms):
    def _get_preprocess(self):
        preprocess = {}
        preprocess.update(ONNXRT_IT_TRANSFORMS["preprocess"])
        return preprocess

    def _get_postprocess(self):
        postprocess = {}
        postprocess.update(ONNXRT_IT_TRANSFORMS["postprocess"])
        return postprocess

    def _get_general(self):
        general = {}
        general.update(ONNXRT_IT_TRANSFORMS["general"])
        return general

class EngineTransforms(Transforms):
    def _get_preprocess(self):
        preprocess = {}
        preprocess.update(ENGINE_TRANSFORMS["preprocess"])
        return preprocess

    def _get_postprocess(self):
        postprocess = {}
        postprocess.update(ENGINE_TRANSFORMS["postprocess"])
        return postprocess

    def _get_general(self):
        general = {}
        general.update(ENGINE_TRANSFORMS["general"])
        return general

framework_transforms = {"tensorflow": TensorflowTransforms,
                        "tensorflow_itex": TensorflowTransforms,
                        "mxnet": MXNetTransforms,
                        "pytorch": PyTorchTransforms,
                        "pytorch_ipex": PyTorchTransforms,
                        "pytorch_fx": PyTorchTransforms,
                        "onnxrt_qlinearops": ONNXRTQLTransforms,
                        "onnxrt_integerops": ONNXRTITTransforms,
                        "engine": EngineTransforms}

# transform registry will register transforms into these dicts
TENSORFLOW_TRANSFORMS = {"preprocess": {}, "postprocess": {}, "general": {}}
MXNET_TRANSFORMS = {"preprocess": {}, "postprocess": {}, "general": {}}
PYTORCH_TRANSFORMS = {"preprocess": {}, "postprocess": {}, "general": {}}
ONNXRT_QL_TRANSFORMS = {"preprocess": {}, "postprocess": {}, "general": {}}
ONNXRT_IT_TRANSFORMS = {"preprocess": {}, "postprocess": {}, "general": {}}
ENGINE_TRANSFORMS = {"preprocess": {}, "postprocess": {}, "general": {}}

registry_transforms = {"tensorflow": TENSORFLOW_TRANSFORMS,
                       "tensorflow_itex": TENSORFLOW_TRANSFORMS,
                       "mxnet": MXNET_TRANSFORMS,
                       "pytorch": PYTORCH_TRANSFORMS,
                       "pytorch_ipex": PYTORCH_TRANSFORMS,
                       "pytorch_fx": PYTORCH_TRANSFORMS,
                       "onnxrt_qlinearops": ONNXRT_QL_TRANSFORMS,
                       "onnxrt_integerops": ONNXRT_IT_TRANSFORMS,
                       "engine": ENGINE_TRANSFORMS}

class TRANSFORMS(object):
    def __init__(self, framework, process):
        assert framework in ("tensorflow", "tensorflow_itex", "engine",
                             "pytorch", "pytorch_ipex", "pytorch_fx",
                             "onnxrt_qlinearops", "onnxrt_integerops", "mxnet"), \
                             "framework support tensorflow pytorch mxnet onnxrt engine"
        assert process in ("preprocess", "postprocess",
                           "general"), "process support preprocess postprocess, general"
        self.transforms = framework_transforms[framework](process).transforms
        self.framework = framework
        self.process = process

    def __getitem__(self, transform_type):
        assert transform_type in self.transforms.keys(), "transform support {}".\
            format(self.transforms.keys())
        return self.transforms[transform_type]

    def register(self, name, transform_cls):
        assert name not in registry_transforms[self.framework][self.process].keys(), \
            'register transform name already exists.'
        registry_transforms[self.framework][self.process].update({name: transform_cls})


def transform_registry(transform_type, process, framework):
    """The class decorator used to register all transform subclasses.


    Args:
        transform_type (str): Transform registration name
        process (str): support 3 process including 'preprocess', 'postprocess', 'general'
        framework (str): support 4 framework including 'tensorflow', 'pytorch', 'mxnet', 'onnxrt'
        cls (class): The class of register.

    Returns:
        cls: The class of register.
    """
    def decorator_transform(cls):
        for single_framework in [fwk.strip() for fwk in framework.split(',')]:
            assert single_framework in [
                "tensorflow",
                "tensorflow_itex",
                "mxnet",
                "pytorch",
                "pytorch_ipex",
                "pytorch_fx",
                "onnxrt_qlinearops",
                "onnxrt_integerops",
                "engine"
            ], "The framework support tensorflow mxnet pytorch onnxrt engine"
            if transform_type in registry_transforms[single_framework][process].keys():
                raise ValueError('Cannot have two transforms with the same name')
            registry_transforms[single_framework][process][transform_type] = cls
        return cls
    return decorator_transform


class BaseTransform(object):
    """The base class for transform. __call__ method is needed when write user specific transform

    """
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class TensorflowWrapFunction(object):
    def __init__(self, transform_func):
        self.transform_func = transform_func

    def __call__(self, **kwargs):
        return TensorflowTransform(self.transform_func, **kwargs)

class TensorflowTransform(BaseTransform):
    def __init__(self, transform_func, **kwargs):
        self.kwargs = kwargs
        self.transform_func = transform_func

    def __call__(self, sample):
        image, label = sample
        image = self.transform_func(image, **self.kwargs)
        return (image, label)

class PytorchMxnetWrapFunction(object):
    def __init__(self, transform_func):
        self.transform_func = transform_func

    def __call__(self, **args):
        return PytorchMxnetTransform(self.transform_func(**args))

class PytorchMxnetTransform(BaseTransform):
    def __init__(self, transform_func):
        self.transform_func = transform_func

    def __call__(self, sample):
        image, label = sample
        image = self.transform_func(image)
        return (image, label)

interpolation_map = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
}

interpolation_pytorch_map = {
    'nearest': 0,
    'bilinear': 2,
    'bicubic': 3,
}

interpolation_mxnet_map = {
    'nearest': 0,
    'bilinear': 1,
    'bicubic': 2,
}

def get_torchvision_map(interpolation):
    try:
        from torchvision.transforms.functional import InterpolationMode
        interpolation_torchvision_map = {
            0: InterpolationMode.NEAREST,
            2: InterpolationMode.BILINEAR,
            3: InterpolationMode.BICUBIC,
        }
        return interpolation_torchvision_map[interpolation]
    except: # pragma: no cover
        return interpolation

@transform_registry(transform_type="Compose", process="general", \
                 framework="onnxrt_qlinearops, onnxrt_integerops, tensorflow, engine")
class ComposeTransform(BaseTransform):
    """Composes several transforms together.

    Args:
        transform_list (list of Transform objects):  list of transforms to compose

    Returns:
        sample (tuple): tuple of processed image and label
    """

    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, sample):
        for transform in self.transform_list:
            sample = transform(sample)
        return sample

@transform_registry(transform_type="CropToBoundingBox", process="preprocess", \
        framework="pytorch")
class CropToBoundingBox(BaseTransform):
    """Crops an image to a specified bounding box.

    Args:
        offset_height (int): Vertical coordinate of the top-left corner of the result in the input
        offset_width (int): Horizontal coordinate of the top-left corner of the result in the input
        target_height (int): Height of the result
        target_width (int): Width of the result

    Returns:
        tuple of processed image and label
    """

    def __init__(self, offset_height, offset_width, target_height, target_width):
        self.offset_height = offset_height
        self.offset_width = offset_width
        self.target_height = target_height
        self.target_width = target_width

    def __call__(self, sample):
        image, label = sample
        image = torchvision.transforms.functional.crop(
                    image,
                    self.offset_height,
                    self.offset_width,
                    self.target_height,
                    self.target_width)
        return (image, label)

@transform_registry(transform_type="CropToBoundingBox", process="preprocess", \
        framework="mxnet")
class MXNetCropToBoundingBox(CropToBoundingBox):
    """Crops an image to a specified bounding box.

    Args:
        offset_height (int): Vertical coordinate of the top-left corner of the result in the input
        offset_width (int): Horizontal coordinate of the top-left corner of the result in the input
        target_height (int): Height of the result
        target_width (int): Width of the result

    Returns:
        tuple of processed image and label
    """

    def __call__(self, sample):
        image, label = sample
        image = mx.image.fixed_crop(
                    image,
                    self.offset_height,
                    self.offset_width,
                    self.target_height,
                    self.target_width)
        return (image, label)

@transform_registry(transform_type="CropToBoundingBox", process="preprocess", \
                framework="onnxrt_qlinearops, onnxrt_integerops")
class ONNXRTCropToBoundingBox(CropToBoundingBox):
    """Crops an image to a specified bounding box.

    Args:
        offset_height (int): Vertical coordinate of the top-left corner of the result in the input
        offset_width (int): Horizontal coordinate of the top-left corner of the result in the input
        target_height (int): Height of the result
        target_width (int): Width of the result

    Returns:
        tuple of processed image and label
    """

    def __call__(self, sample):
        image, label = sample
        image = image[self.offset_height : self.offset_height+self.target_height,
                      self.offset_width : self.offset_width+self.target_width, :]
        return (image, label)

@transform_registry(transform_type="CropToBoundingBox", process="preprocess", \
                framework="tensorflow")
class TensorflowCropToBoundingBox(CropToBoundingBox):
    """Crops an image to a specified bounding box.

    Args:
        offset_height (int): Vertical coordinate of the top-left corner of the result in the input
        offset_width (int): Horizontal coordinate of the top-left corner of the result in the input
        target_height (int): Height of the result
        target_width (int): Width of the result

    Returns:
        tuple of processed image and label
    """

    def __call__(self, sample):
        image, label = sample
        if isinstance(image, tf.Tensor):
            image = tf.image.crop_to_bounding_box(image, self.offset_height,
                    self.offset_width, self.target_height, self.target_width)
        else:
            image = image[self.offset_height : self.offset_height+self.target_height,
                    self.offset_width : self.offset_width+self.target_width, :]
        return (image, label)

@transform_registry(transform_type="ResizeWithRatio", process="preprocess", \
                framework="onnxrt_qlinearops, onnxrt_integerops, pytorch, mxnet")
class ResizeWithRatio(BaseTransform):
    """Resize image with aspect ratio and pad it to max shape(optional).
       If the image is padded, the label will be processed at the same time.
       The input image should be np.array.

    Args:
        min_dim (int, default=800):
            Resizes the image such that its smaller dimension == min_dim
        max_dim (int, default=1365):
            Ensures that the image longest side doesn't exceed this value
        padding (bool, default=False):
            If true, pads image with zeros so its size is max_dim x max_dim

    Returns:
        tuple of processed image and label
    """

    def __init__(self, min_dim=800, max_dim=1365, padding=False, constant_value=0):
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.padding = padding
        self.constant_value = constant_value

    def __call__(self, sample):
        image, label = sample
        height, width = image.shape[:2]
        scale = 1
        if self.min_dim:
            scale = max(1, self.min_dim / min(height, width))
        if self.max_dim:
            image_max = max(height, width)
            if round(image_max * scale) > self.max_dim:
                scale = self.max_dim / image_max
        if scale != 1:
            image = cv2.resize(image, (round(height * scale), round(width * scale)))

        bbox, str_label, int_label, image_id = label

        if self.padding:
            h, w = image.shape[:2]
            pad_param = [[(self.max_dim-h)//2, self.max_dim-h-(self.max_dim-h)//2],
                         [(self.max_dim-w)//2, self.max_dim-w-(self.max_dim-w)//2],
                         [0, 0]]
            if not isinstance(bbox, np.ndarray):
                bbox = np.array(bbox)
            resized_box = bbox * [height, width, height, width] * scale
            moved_box = (resized_box + [(self.max_dim-h)//2, (self.max_dim-w)//2, \
                (self.max_dim-h)//2, (self.max_dim-w)//2])
            bbox = moved_box / [self.max_dim, self.max_dim, self.max_dim, self.max_dim]
            image = np.pad(image, pad_param, mode='constant', constant_values=self.constant_value)
        return image, (bbox, str_label, int_label, image_id)

@transform_registry(transform_type="ResizeWithRatio", process="preprocess", \
                framework="tensorflow")
class TensorflowResizeWithRatio(BaseTransform):
    """Resize image with aspect ratio and pad it to max shape(optional).
       If the image is padded, the label will be processed at the same time.
       The input image should be np.array or tf.Tensor.

    Args:
        min_dim (int, default=800):
            Resizes the image such that its smaller dimension == min_dim
        max_dim (int, default=1365):
            Ensures that the image longest side doesn't exceed this value
        padding (bool, default=False):
            If true, pads image with zeros so its size is max_dim x max_dim

    Returns:
        tuple of processed image and label
    """

    def __init__(self, min_dim=800, max_dim=1365, padding=False, constant_value=0):
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.padding = padding
        self.constant_value = constant_value

    def __call__(self, sample):
        image, label = sample
        if isinstance(image, tf.Tensor):
            shape = tf.shape(input=image)
            height = tf.cast(shape[0], dtype=tf.float32)
            width = tf.cast(shape[1], dtype=tf.float32)
            scale = 1
            if self.min_dim:
                scale = tf.maximum(1., tf.cast(self.min_dim / tf.math.minimum(height, width),\
                                                                            dtype=tf.float32))
            if self.max_dim:
                image_max = tf.cast(tf.maximum(height, width), dtype=tf.float32)
                scale = tf.cond(pred=tf.greater(tf.math.round(image_max * scale), self.max_dim), \
                                true_fn=lambda: self.max_dim / image_max,
                                false_fn=lambda: scale)
            image = tf.image.resize(image, (tf.math.round(height * scale), \
                                            tf.math.round(width * scale)))
            bbox, str_label, int_label, image_id = label

            if self.padding:
                shape = tf.shape(input=image)
                h = tf.cast(shape[0], dtype=tf.float32)
                w = tf.cast(shape[1], dtype=tf.float32)
                pad_param = [[(self.max_dim-h)//2, self.max_dim-h-(self.max_dim-h)//2],
                             [(self.max_dim-w)//2, self.max_dim-w-(self.max_dim-w)//2],
                             [0, 0]]
                resized_box = bbox * [height, width, height, width] * scale
                moved_box = (resized_box + [(self.max_dim-h)//2, (self.max_dim-w)//2, \
                    (self.max_dim-h)//2, (self.max_dim-w)//2])
                bbox = moved_box / [self.max_dim, self.max_dim, self.max_dim, self.max_dim]
                image = tf.pad(image, pad_param, constant_values=self.constant_value)
        else:
            transform = ResizeWithRatio(self.min_dim, self.max_dim, self.padding)
            image, (bbox, str_label, int_label, image_id) = transform(sample)
        return image, (bbox, str_label, int_label, image_id)

@transform_registry(transform_type="Transpose", process="preprocess", \
        framework="onnxrt_qlinearops, onnxrt_integerops")
class Transpose(BaseTransform):
    """Transpose image according to perm.

    Args:
        perm (list): A permutation of the dimensions of input image

    Returns:
        tuple of processed image and label
    """

    def __init__(self, perm):
        self.perm = perm

    def __call__(self, sample):
        image, label = sample
        assert len(image.shape) == len(self.perm), "Image rank doesn't match Perm rank"
        image = np.transpose(image, axes=self.perm)
        return (image, label)

@transform_registry(transform_type="Transpose", process="preprocess", framework="tensorflow")
class TensorflowTranspose(Transpose):
    """Transpose image according to perm.

    Args:
        perm (list): A permutation of the dimensions of input image

    Returns:
        tuple of processed image and label
    """

    def __call__(self, sample):
        image, label = sample
        assert len(image.shape) == len(self.perm), "Image rank doesn't match Perm rank"
        if isinstance(image, tf.Tensor):
            image = tf.transpose(image, perm=self.perm)
        else:
            image = np.transpose(image, axes=self.perm)
        return (image, label)

@transform_registry(transform_type="Transpose", process="preprocess", framework="mxnet")
class MXNetTranspose(Transpose):
    """Transpose image according to perm.

    Args:
        perm (list): A permutation of the dimensions of input image

    Returns:
        tuple of processed image and label
    """

    def __call__(self, sample):
        image, label = sample
        assert len(image.shape) == len(self.perm), "Image rank doesn't match Perm rank"
        image = mx.ndarray.transpose(image, self.perm)
        return (image, label)

@transform_registry(transform_type="Transpose", process="preprocess", framework="pytorch")
class PyTorchTranspose(Transpose):
    """Transpose image according to perm.

    Args:
        perm (list): A permutation of the dimensions of input image

    Returns:
        tuple of processed image and label
    """

    def __call__(self, sample):
        image, label = sample
        assert len(image.shape) == len(self.perm), "Image rank doesn't match Perm rank"
        image = image.permute(self.perm)
        return (image, label)

@transform_registry(transform_type="RandomVerticalFlip", process="preprocess", \
        framework="onnxrt_qlinearops, onnxrt_integerops")
class RandomVerticalFlip(BaseTransform):
    """Vertically flip the given image randomly.

    Returns:
        tuple of processed image and label
    """

    def __call__(self, sample):
        image, label = sample
        if np.random.rand(1)[0] > 0.5:
            image = np.flipud(image)
        return (image, label)

@transform_registry(transform_type="RandomVerticalFlip", process="preprocess", \
        framework="tensorflow")
class TensorflowRandomVerticalFlip(BaseTransform):
    """Vertically flip the given image randomly.

    Returns:
        tuple of processed image and label
    """

    def __call__(self, sample):
        image, label = sample
        if isinstance(image, tf.Tensor):
            image = tf.image.random_flip_up_down(image)
        else:
            if np.random.rand(1)[0] > 0.5:
                image = np.flipud(image)
        return (image, label)

@transform_registry(transform_type="RandomHorizontalFlip", process="preprocess", \
        framework="onnxrt_qlinearops, onnxrt_integerops")
class RandomHorizontalFlip(BaseTransform):
    """Horizontally flip the given image randomly.

    Returns:
        tuple of processed image and label
    """

    def __call__(self, sample):
        image, label = sample
        if np.random.rand(1)[0] > 0.5:
            image = np.fliplr(image)
        return (image, label)

@transform_registry(transform_type="RandomHorizontalFlip", process="preprocess", \
        framework="tensorflow")
class TensorflowRandomHorizontalFlip(BaseTransform):
    """Horizontally flip the given image randomly.

    Returns:
        tuple of processed image and label
    """

    def __call__(self, sample):
        image, label = sample
        if isinstance(image, tf.Tensor):
            image = tf.image.random_flip_left_right(image)
        else:
            if np.random.rand(1)[0] > 0.5:
                image = np.fliplr(image)
        return (image, label)

@transform_registry(transform_type="ToArray", process="preprocess", \
        framework="onnxrt_qlinearops, onnxrt_integerops, tensorflow, pytorch, mxnet")
class ToArray(BaseTransform):
    """Convert PIL Image or NDArray to numpy array.

    Returns:
        tuple of processed image and label
    """

    def __call__(self, sample):
        from PIL import Image
        image, label = sample
        if isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, mx.ndarray.NDArray): # pylint: disable=no-member
            image = image.asnumpy()
        else:
            raise ValueError("Unknown image type!")
        return (image, label)

np_dtype_map = {'int8': np.int8, 'uint8': np.uint8, 'complex64': np.complex64,
           'uint16': np.uint16, 'int32': np.int32, 'uint32': np.uint32,
           'int64': np.int64, 'uint64': np.uint64, 'float32': np.float32,
           'float16': np.float16, 'float64': np.float64, 'bool': np.bool,
           'string': np.str, 'complex128': np.complex128, 'int16': np.int16}

@transform_registry(transform_type="Cast",
                    process="general", framework="tensorflow")
class CastTFTransform(BaseTransform):
    """Convert image to given dtype.

    Args:
        dtype (str, default='float32'): A dtype to convert image to

    Returns:
        tuple of processed image and label
    """

    def __init__(self, dtype='float32'):
        self.tf_dtype_map = {'int16': tf.int16, 'uint8': tf.uint8, 'uint16': tf.uint16,
                        'uint32':tf.uint32, 'uint64': tf.uint64, 'complex64': tf.complex64,
                        'int32': tf.int32, 'int64':tf.int64, 'float32': tf.float32,
                        'float16': tf.float16, 'float64':tf.float64, 'bool': tf.bool,
                        'string': tf.string, 'int8': tf.int8, 'complex128': tf.complex128}

        assert dtype in self.tf_dtype_map.keys(), 'Unknown dtype'
        self.dtype = dtype

    def __call__(self, sample):
        image, label = sample
        if isinstance(image, tf.Tensor):
            image = tf.image.convert_image_dtype(image, dtype=self.tf_dtype_map[self.dtype])
        else:
            image = image.astype(np_dtype_map[self.dtype])
        return (image, label)

@transform_registry(transform_type="Cast",
                    process="general", framework="onnxrt_qlinearops, onnxrt_integerops")
class CastONNXTransform(BaseTransform):
    """Convert image to given dtype.

    Args:
        dtype (str, default='float32'): A dtype to convert image to

    Returns:
        tuple of processed image and label
    """

    def __init__(self, dtype='float32'):
        assert dtype in np_dtype_map.keys(), 'Unknown dtype'
        self.dtype = dtype

    def __call__(self, sample):
        image, label = sample
        image = image.astype(np_dtype_map[self.dtype])
        return (image, label)

@transform_registry(transform_type="Cast", process="general", framework="pytorch")
class CastPyTorchTransform(BaseTransform):
    """Convert image to given dtype.

    Args:
        dtype (str, default='float32'): A dtype to convert image to

    Returns:
        tuple of processed image and label
    """

    def __init__(self, dtype='float32'):
        dtype_map = {'int8': torch.int8, 'uint8': torch.uint8, 'complex128': torch.complex128,
                     'int32':torch.int32, 'int64':torch.int64, 'complex64': torch.complex64,
                     'bfloat16':torch.bfloat16, 'float64':torch.float64, 'bool': torch.bool,
                     'float16':torch.float16, 'int16':torch.int16, 'float32': torch.float32}
        assert dtype in dtype_map.keys(), 'Unknown dtype'
        self.dtype = dtype_map[dtype]

    def __call__(self, sample):
        image, label = sample
        image = image.type(self.dtype)
        return (image, label)

@transform_registry(transform_type="CenterCrop",
                    process="preprocess", framework="tensorflow")
class CenterCropTFTransform(BaseTransform):
    """Crops the given image at the center to the given size.

    Args:
        size (list or int): Size of the result

    Returns:
        tuple of processed image and label
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = size, size
        elif isinstance(size, list):
            if len(size) == 1:
                self.size = size[0], size[0]
            elif len(size) == 2:
                self.size = size[0], size[1]

    def __call__(self, sample):
        image, label = sample
        if isinstance(image, tf.Tensor):
            if len(image.shape) == 3:
                height, width = image.shape[0:2]
            elif len(image.shape) == 4:
                height, width = image.shape[1:3]
            else:
                raise ValueError("Unknown image shape")
            if height < self.size[0] or width < self.size[1]:
                raise ValueError("Target size shouldn't be lager than image size")
            y0 = (height - self.size[0]) // 2
            x0 = (width - self.size[1]) // 2
            image = tf.image.crop_to_bounding_box(image, y0, x0, self.size[0], self.size[1])
        else:
            transform = CenterCropTransform(self.size)
            image, label = transform(sample)
        return (image, label)

@transform_registry(transform_type="PaddedCenterCrop", process="preprocess", \
                framework="tensorflow")
class PaddedCenterCropTransform(BaseTransform):
    def __init__(self, size, crop_padding=0):
        if isinstance(size, int):
            self.image_size = size
        elif isinstance(size, list):
            if len(size) == 1:
                self.image_size = size[0]
            elif len(size) == 2:
                if size[0] != size[1]:
                    raise ValueError("'crop height must eaqual to crop width'")
                self.image_size = size[0]
        self.crop_padding = crop_padding

    def __call__(self, sample):
        image, label = sample
        h, w = image.shape[0], image.shape[1]

        padded_center_crop_size = \
            int((self.image_size / (self.image_size + self.crop_padding)) * min(h, w))

        y0 = (h - padded_center_crop_size + 1) // 2
        x0 = (w - padded_center_crop_size + 1) // 2
        image = image[y0:y0 + padded_center_crop_size, x0:x0 + padded_center_crop_size, :]
        return (image, label)

@transform_registry(transform_type="Resize",
                    process="preprocess", framework="tensorflow")
class ResizeTFTransform(BaseTransform):
    """Resize the input image to the given size.

    Args:
        size (list or int): Size of the result
        interpolation (str, default='bilinear'):Desired interpolation type,
                                                support 'bilinear', 'nearest', 'bicubic'

    Returns:
        tuple of processed image and label
    """
    def __init__(self, size, interpolation='bilinear'):
        if isinstance(size, int):
            self.size = size, size
        elif isinstance(size, list):
            if len(size) == 1:
                self.size = size[0], size[0]
            elif len(size) == 2:
                self.size = size[0], size[1]
        self.interpolation = interpolation

        if self.interpolation not in ['bilinear', 'nearest', 'bicubic']:
            raise ValueError('Unsupported interpolation type!')

    def __call__(self, sample):
        image, label = sample
        if isinstance(image, tf.Tensor):
            image = tf.image.resize(image, self.size, method=self.interpolation)
        else:
            image = cv2.resize(image, self.size,
                interpolation=interpolation_map[self.interpolation])
        return (image, label)

@transform_registry(transform_type="Resize", process="preprocess", \
                        framework="pytorch")
class ResizePytorchTransform(BaseTransform):
    """Resize the input image to the given size.

    Args:
        size (list or int): Size of the result
        interpolation (str, default='bilinear'):Desired interpolation type,
                                                support 'bilinear', 'nearest', 'bicubic'

    Returns:
        tuple of processed image and label
    """

    def __init__(self, size, interpolation='bilinear'):
        self.size = size
        if interpolation in interpolation_pytorch_map.keys():
            self.interpolation = get_torchvision_map(interpolation_pytorch_map[interpolation])
        else:
            raise ValueError("Undefined interpolation type")

    def __call__(self, sample):
        image, label = sample
        transformer = torchvision.transforms.Resize(size=self.size,
                                        interpolation=self.interpolation)
        return (transformer(image), label)

@transform_registry(transform_type="RandomCrop",
                    process="preprocess", framework="tensorflow")
class RandomCropTFTransform(BaseTransform):
    """Crop the image at a random location to the given size.

    Args:
        size (list or tuple or int): Size of the result

    Returns:
        tuple of processed image and label
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = size, size
        elif isinstance(size, list) or isinstance(size, tuple):
            if len(size) == 1:
                self.size = size[0], size[0]
            elif len(size) == 2:
                self.size = size[0], size[1]

    def __call__(self, sample):
        image, label = sample
        if isinstance(image, tf.Tensor):
            if len(image.shape) == 3:
                height, width = image.shape[0:2]
            elif len(image.shape) == 4:
                height, width = image.shape[1:3]

            if self.size[0] > height or self.size[1] > width:
                raise ValueError('Crop size must be smaller than image size')

            if self.size[0] == height and self.size[1] == width:
                return (image, label)

            height = tf.cast(height, dtype=tf.float32)
            width = tf.cast(width, dtype=tf.float32)
            offset_height = (height - self.size[0]) / 2
            offset_width = (width - self.size[1]) / 2
            offset_height = tf.cast(offset_height, dtype=tf.int32)
            offset_width = tf.cast(offset_width, dtype=tf.int32)

            image = tf.image.crop_to_bounding_box(image, offset_height,
                        offset_width, self.size[0], self.size[1])
        else:
            transform = RandomCropTransform(self.size)
            image, label = transform(sample)
        return (image, label)

@transform_registry(transform_type="RandomResizedCrop", process="preprocess", \
                        framework="pytorch")
class RandomResizedCropPytorchTransform(BaseTransform):
    """Crop the given image to random size and aspect ratio.

    Args:
        size (list or int):
            Size of the result
        scale (tuple or list, default=(0.08, 1.0)):
            range of size of the origin size cropped
        ratio (tuple or list, default=(3. / 4., 4. / 3.)):
            range of aspect ratio of the origin aspect ratio cropped
        interpolation (str, default='bilinear'):
            Desired interpolation type, support 'bilinear', 'nearest', 'bicubic'

    Returns:
        tuple of processed image and label
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                        interpolation='bilinear'):
        self.size = size
        self.scale = scale
        self.ratio = ratio

        if interpolation in interpolation_pytorch_map.keys():
            self.interpolation = get_torchvision_map(interpolation_pytorch_map[interpolation])
        else:
            raise ValueError("Undefined interpolation type")

        if scale[0] > scale[1] or ratio[0] > ratio[1]:
            raise ValueError("Scale and ratio should be of kind (min, max)")

    def __call__(self, sample):
        image, label = sample
        transformer = torchvision.transforms.RandomResizedCrop(size=self.size,
            scale=self.scale, ratio=self.ratio, interpolation=self.interpolation)
        return (transformer(image), label)

@transform_registry(transform_type="RandomResizedCrop", process="preprocess", \
                        framework="mxnet")
class RandomResizedCropMXNetTransform(BaseTransform):
    """Crop the given image to random size and aspect ratio.

    Args:
        size (list or int):
            Size of the result
        scale (tuple or list, default=(0.08, 1.0)):
            range of size of the origin size cropped
        ratio (tuple or list, default=(3. / 4., 4. / 3.)):
            range of aspect ratio of the origin aspect ratio cropped
        interpolation (str, default='bilinear'):
            Desired interpolation type, support 'bilinear', 'nearest', 'bicubic'

    Returns:
        tuple of processed image and label
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                        interpolation='bilinear'):
        if isinstance(size, int):
            self.size = size, size
        elif isinstance(size, list):
            if len(size) == 1:
                self.size = size[0], size[0]
            elif len(size) == 2:
                self.size = size[1], size[0]
        self.scale = scale
        self.ratio = ratio

        if interpolation in interpolation_mxnet_map.keys():
            self.interpolation = interpolation_mxnet_map[interpolation]
        else:
            raise ValueError("Undefined interpolation type")

        if scale[0] > scale[1] or ratio[0] > ratio[1]:
            raise ValueError("Scale and ratio should be of kind (min, max)")

    def __call__(self, sample):
        image, label = sample
        transformer = mx.gluon.data.vision.transforms.RandomResizedCrop(size=self.size,
                    scale=self.scale, ratio=self.ratio, interpolation=self.interpolation)
        return (transformer(image), label)


@transform_registry(transform_type="RandomResizedCrop",
                    process="preprocess", framework="tensorflow")
class RandomResizedCropTFTransform(BaseTransform):
    """Crop the given image to random size and aspect ratio.

    Args:
        size (list or int):
            Size of the result
        scale (tuple or list, default=(0.08, 1.0)):
            range of size of the origin size cropped
        ratio (tuple or list, default=(3. / 4., 4. / 3.)):
            range of aspect ratio of the origin aspect ratio cropped
        interpolation (str, default='bilinear'):
            Desired interpolation type, support 'bilinear', 'nearest'

    Returns:
        tuple of processed image and label
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(
            3. / 4., 4. / 3.), interpolation='bilinear'):
        if isinstance(size, int):
            self.size = size, size
        elif isinstance(size, list):
            if len(size) == 1:
                self.size = size[0], size[0]
            elif len(size) == 2:
                self.size = size[0], size[1]

        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        if self.interpolation not in ['bilinear', 'nearest']:
            raise ValueError('Unsupported interpolation type!')
        if scale[0] > scale[1] or ratio[0] > ratio[1]:
            raise ValueError("Scale and ratio should be of kind (min, max)")

    def get_params(self, image, scale, ratio):
        shape = image.shape
        height = tf.cast(shape[0], dtype=tf.float32)
        width = tf.cast(shape[1], dtype=tf.float32)
        src_area = height * width

        for _ in range(10):
            target_area = np.random.uniform(scale[0], scale[1]) * src_area
            log_ratio = (np.log(ratio[0]), np.log(ratio[1]))
            new_ratio = np.exp(np.random.uniform(log_ratio[0], log_ratio[1]))

            new_w = tf.math.round(
                tf.math.sqrt(tf.math.multiply(target_area, new_ratio)))
            new_h = tf.math.round(
                tf.math.sqrt(tf.math.divide(target_area, new_ratio)))

            x0, y0 = tf.case(
                [(tf.math.logical_and(
                    tf.math.greater(width, new_w), tf.math.greater(height, new_h)),
                    lambda: (tf.random.uniform(
                        shape=[], maxval=tf.math.subtract(width, new_w)),
                    tf.random.uniform(
                        shape=[], maxval=tf.math.subtract(height, new_h)))
                  )],
                default=lambda: (-1.0, -1.0))
            if x0 != -1.0 and y0 != -1.0:
                return y0, x0, new_h, new_w

        in_ratio = width / height
        new_w, new_h = tf.case([(tf.math.greater(min(ratio), in_ratio),
                                 lambda: (width, tf.math.round(width / min(ratio)))),
                                (tf.math.greater(in_ratio, max(ratio)),
                                 lambda: (height, tf.math.round(height * max(ratio))))],
                               default=lambda: (width, height))

        y0 = (height - new_h) / 2
        x0 = (width - new_w) / 2
        return y0, x0, new_h, new_w

    def __call__(self, sample):
        image, label = sample
        if isinstance(image, tf.Tensor):
            y0, x0, h, w = self.get_params(image, self.scale, self.ratio)
            squeeze = False
            if len(image.shape) == 3:
                squeeze = True
                image = tf.expand_dims(image, axis=0)
            height, width = image.shape[1:3]
            height = tf.cast(height, dtype=tf.float32)
            width = tf.cast(width, dtype=tf.float32)
            box_indices = tf.range(0, image.shape[0], dtype=tf.int32)
            boxes = [y0/height, x0/width, (y0+h)/height, (x0+w)/width]
            boxes = tf.broadcast_to(boxes, [image.shape[0], 4])
            image = tf.image.crop_and_resize(image, boxes, box_indices,
                            self.size, self.interpolation)
            if squeeze:
                image = tf.squeeze(image, axis=0)
        else:
            transform = RandomResizedCropTransform(self.size, self.scale,
                    self.ratio, self.interpolation)
            image, label = transform(sample)
        return (image, label)

@transform_registry(transform_type="Normalize", process="preprocess",
                        framework="tensorflow")
class NormalizeTFTransform(BaseTransform):
    """Normalize a image with mean and standard deviation.

    Args:
        mean (list, default=[0.0]):
            means for each channel, if len(mean)=1, mean will be broadcasted to each channel,
            otherwise its length should be same with the length of image shape
        std (list, default=[1.0]):
            stds for each channel, if len(std)=1, std will be broadcasted to each channel,
            otherwise its length should be same with the length of image shape

    Returns:
        tuple of processed image and label
    """

    def __init__(self, mean=[0.0], std=[1.0]):
        self.mean = mean
        self.std = std
        for item in self.std:
            if item < 10**-6:
                raise ValueError("Std should be greater than 0")

    def __call__(self, sample):
        image, label = sample
        if isinstance(image, tf.Tensor):
            orig_dtype = image.dtype
            mean = tf.broadcast_to(self.mean, tf.shape(input=image))
            mean = tf.cast(mean, dtype=image.dtype)
            std = tf.broadcast_to(self.std, tf.shape(input=image))
            std = tf.cast(std, dtype=image.dtype)
            image = (image - mean) / std
            image = tf.cast(image, dtype=orig_dtype)
        else:
            transform = NormalizeTransform(self.mean, self.std)
            image, label = transform(sample)
        return (image, label)

@transform_registry(transform_type='Rescale', process="preprocess", \
                framework='tensorflow')
class RescaleTFTransform(BaseTransform):
    """Scale the values of image to [0,1].

    Returns:
        tuple of processed image and label
    """

    def __call__(self, sample):
        image, label = sample
        if isinstance(image, tf.Tensor):
            image = tf.cast(image, tf.float32) / 255.
        else:
            image = image.astype('float32') / 255.
        return (image, label)

@transform_registry(transform_type='Rescale', process="preprocess", \
                framework='onnxrt_qlinearops, onnxrt_integerops')
class RescaleTransform(BaseTransform):
    """Scale the values of image to [0,1].

    Returns:
        tuple of processed image and label
    """

    def __call__(self, sample):
        image, label = sample
        if isinstance(image, np.ndarray):
            image = image.astype('float32') / 255.
        return (image, label)

@transform_registry(transform_type='AlignImageChannel', process="preprocess", \
    framework='tensorflow, onnxrt_qlinearops, onnxrt_integerops, mxnet')
class AlignImageChannelTransform(BaseTransform):
    """ Align image channel, now just support [H,W]->[H,W,dim], [H,W,4]->[H,W,3] and
        [H,W,3]->[H,W].
        Input image must be np.ndarray.

    Returns:
        tuple of processed image and label
    """

    def __init__(self, dim=3):
        logger.warning("This transform is going to be deprecated")
        if dim < 1 or dim > 4:
            raise ValueError('Unsupport image dim!')
        self.dim = dim

    def __call__(self, sample):
        image, label = sample
        if len(image.shape) == 2:
            image = np.dstack([image]*self.dim)
        if isinstance(image, np.ndarray) and image.shape[-1] != self.dim:
            if image.shape[-1] == 4 and self.dim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[-1] == 3 and self.dim == 1:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                image = np.expand_dims(image, axis=-1)
            else:
                raise ValueError('Unsupport conversion!')
        return (image, label)

@transform_registry(transform_type='AlignImageChannel', process="preprocess", \
    framework='pytorch')
class PyTorchAlignImageChannel(BaseTransform):
    """ Align image channel, now just support [H,W,4]->[H,W,3] and
        [H,W,3]->[H,W].
        Input image must be PIL Image.

    Returns:
        tuple of processed image and label
    """

    def __init__(self, dim=3):
        logger.warning("This transform is going to be deprecated")
        if dim != 1 and dim != 3:
            raise ValueError('Unsupport image dim!')
        self.dim = dim

    def __call__(self, sample):
        from PIL import Image
        image, label = sample
        assert isinstance(image, Image.Image), 'Input image must be PIL Image'
        if self.dim == 3:
            image = image.convert('RGB')
        elif self.dim == 1:
            image = image.convert('L')
        else:
            raise ValueError('Unsupport conversion!')
        return (image, label)

@transform_registry(transform_type="ToNDArray", process="preprocess", \
                framework="mxnet")
class ToNDArrayTransform(BaseTransform):
    """Convert np.array to NDArray.

    Returns:
        tuple of processed image and label
    """

    def __call__(self, sample):
        image, label = sample
        image = mx.nd.array(image)
        return image, label

@transform_registry(transform_type="Resize", process="preprocess", framework="mxnet")
class ResizeMXNetTransform(BaseTransform):
    """Resize the input image to the given size.

    Args:
        size (list or int): Size of the result
        interpolation (str, default='bilinear'):Desired interpolation type,
                                                support 'bilinear', 'nearest', 'bicubic'

    Returns:
        tuple of processed image and label
    """

    def __init__(self, size, interpolation='bilinear'):
        if isinstance(size, int):
            self.size = size, size
        elif isinstance(size, list):
            if len(size) == 1:
                self.size = size[0], size[0]
            elif len(size) == 2:
                self.size = size[1], size[0]

        if interpolation in interpolation_mxnet_map.keys():
            self.interpolation = interpolation_mxnet_map[interpolation]
        else:
            raise ValueError("Undefined interpolation type")

    def __call__(self, sample):
        image, label = sample
        transformer = mx.gluon.data.vision.transforms.Resize(size=self.size,
                                interpolation=self.interpolation)
        return (transformer(image), label)


@transform_registry(transform_type="Resize", process="preprocess", \
                framework="onnxrt_qlinearops, onnxrt_integerops")
class ResizeTransform(BaseTransform):
    """Resize the input image to the given size.

    Args:
        size (list or int): Size of the result
        interpolation (str, default='bilinear'):Desired interpolation type,
                                                support 'bilinear', 'nearest', 'bicubic'

    Returns:
        tuple of processed image and label
    """

    def __init__(self, size, interpolation='bilinear'):
        if isinstance(size, int):
            self.size = size, size
        elif isinstance(size, list):
            if len(size) == 1:
                self.size = size[0], size[0]
            elif len(size) == 2:
                self.size = size[0], size[1]

        if interpolation in interpolation_map.keys():
            self.interpolation = interpolation_map[interpolation]
        else:
            raise ValueError("Undefined interpolation type")

    def __call__(self, sample):
        image, label = sample
        image = cv2.resize(image, self.size, interpolation=self.interpolation)
        if len(image.shape) == 2:
            image = np.expand_dims(image, -1)
        return (image, label)

@transform_registry(transform_type="CropResize", process="preprocess", \
                framework="tensorflow")
class CropResizeTFTransform(BaseTransform):
    """Crop the input image with given location and resize it.

    Args:
        x (int):Left boundary of the cropping area
        y (int):Top boundary of the cropping area
        width (int):Width of the cropping area
        height (int):Height of the cropping area
        size (list or int): resize to new size after cropping
        interpolation (str, default='bilinear'):Desired interpolation type,
                                                support 'bilinear', 'nearest', 'bicubic'

    Returns:
        tuple of processed image and label
    """

    def __init__(self, x, y, width, height, size, interpolation='bilinear'):
        if interpolation not in ['bilinear', 'nearest', 'bicubic']:
            raise ValueError('Unsupported interpolation type!')
        self.interpolation = interpolation
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        if isinstance(size, int):
            self.size = size, size
        elif isinstance(size, list):
            if len(size) == 1:
                self.size = size[0], size[0]
            elif len(size) == 2:
                self.size = size[0], size[1]

    def __call__(self, sample):
        image, label = sample
        if isinstance(image, tf.Tensor):
            image = tf.image.crop_to_bounding_box(
                image, self.y, self.x, self.height, self.width)
            image = tf.image.resize(image, self.size, method=self.interpolation)
        else:
            transform = CropResizeTransform(self.x, self.y, self.width,
                        self.height, self.size, self.interpolation)
            image, label = transform(sample)
        return (image, label)

@transform_registry(transform_type="CropResize", process="preprocess", framework="pytorch")
class PyTorchCropResizeTransform(BaseTransform):
    """Crop the input image with given location and resize it.

    Args:
        x (int):Left boundary of the cropping area
        y (int):Top boundary of the cropping area
        width (int):Width of the cropping area
        height (int):Height of the cropping area
        size (list or int): resize to new size after cropping
        interpolation (str, default='bilinear'):Desired interpolation type,
                                                support 'bilinear', 'nearest', 'bicubic'

    Returns:
        tuple of processed image and label
    """

    def __init__(self, x, y, width, height, size, interpolation='bilinear'):
        if interpolation in interpolation_pytorch_map.keys():
            self.interpolation = get_torchvision_map(interpolation_pytorch_map[interpolation])
        else:
            raise ValueError("Undefined interpolation type")
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.size = size

    def __call__(self, sample):
        image, label = sample
        image = image.crop((self.x, self.y, self.x + self.width, self.y + self.height))
        transformer = torchvision.transforms.Resize(size=self.size,
                            interpolation=self.interpolation)
        return (transformer(image), label)

@transform_registry(transform_type="CropResize", process="preprocess", framework="mxnet")
class MXNetCropResizeTransform(BaseTransform):
    """Crop the input image with given location and resize it.

    Args:
        x (int):Left boundary of the cropping area
        y (int):Top boundary of the cropping area
        width (int):Width of the cropping area
        height (int):Height of the cropping area
        size (list or int): resize to new size after cropping
        interpolation (str, default='bilinear'):Desired interpolation type,
                                                support 'bilinear', 'nearest', 'bicubic'

    Returns:
        tuple of processed image and label
    """

    def __init__(self, x, y, width, height, size, interpolation='bilinear'):
        if interpolation in interpolation_mxnet_map.keys():
            self.interpolation = interpolation_mxnet_map[interpolation]
        else:
            raise ValueError("Undefined interpolation type")
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.size = size

    def __call__(self, sample):
        image, label = sample
        transformer = mx.gluon.data.vision.transforms.CropResize(self.x, self.y, self.width,
                                self.height, self.size, self.interpolation)
        return (transformer(image), label)

@transform_registry(transform_type="CropResize", process="preprocess", \
                framework="onnxrt_qlinearops, onnxrt_integerops")
class CropResizeTransform(BaseTransform):
    """Crop the input image with given location and resize it.

    Args:
        x (int):Left boundary of the cropping area
        y (int):Top boundary of the cropping area
        width (int):Width of the cropping area
        height (int):Height of the cropping area
        size (list or int): resize to new size after cropping
        interpolation (str, default='bilinear'):Desired interpolation type,
                                                support 'bilinear', 'nearest', 'bicubic'

    Returns:
        tuple of processed image and label
    """

    def __init__(self, x, y, width, height, size, interpolation='bilinear'):
        if interpolation in interpolation_map.keys():
            self.interpolation = interpolation_map[interpolation]
        else:
            raise ValueError("Undefined interpolation type")
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        if isinstance(size, int):
            self.size = size, size
        elif isinstance(size, list) or isinstance(size, tuple):
            if len(size) == 1:
                self.size = size[0], size[0]
            elif len(size) == 2:
                self.size = size[0], size[1]

    def __call__(self, sample):
        image, label = sample
        image = image[self.y:self.y+self.height, self.x:self.x+self.width, :]
        image = cv2.resize(image, self.size, interpolation=self.interpolation)
        return (image, label)

@transform_registry(transform_type="CenterCrop", process="preprocess", \
                framework="onnxrt_qlinearops, onnxrt_integerops")
class CenterCropTransform(BaseTransform):
    """Crops the given image at the center to the given size.

    Args:
        size (list or int): Size of the result

    Returns:
        tuple of processed image and label
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.height, self.width = size, size
        elif isinstance(size, list) or isinstance(size, tuple):
            if len(size) == 1:
                self.height, self.width = size[0], size[0]
            elif len(size) == 2:
                self.height, self.width = size[0], size[1]

    def __call__(self, sample):
        image, label = sample
        h, w = image.shape[0], image.shape[1]
        if h + 1 < self.height or w + 1 < self.width:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format(
                    (self.height, self.width), (h, w)))

        if self.height == h and self.width == w:
            return (image, label)

        y0 = (h - self.height) // 2
        x0 = (w - self.width) // 2
        image = image[y0:y0 + self.height, x0:x0 + self.width, :]
        return (image, label)

@transform_registry(transform_type="Normalize", process="preprocess", framework="mxnet")
class MXNetNormalizeTransform(BaseTransform):
    """Normalize a image with mean and standard deviation.

    Args:
        mean (list, default=[0.0]):
            means for each channel, if len(mean)=1, mean will be broadcasted to each channel,
            otherwise its length should be same with the length of image shape
        std (list, default=[1.0]):
            stds for each channel, if len(std)=1, std will be broadcasted to each channel,
            otherwise its length should be same with the length of image shape

    Returns:
        tuple of processed image and label
    """

    def __init__(self, mean=[0.0], std=[1.0]):
        self.mean = mean
        self.std = std
        for item in self.std:
            if item < 10**-6:
                raise ValueError("Std should be greater than 0")

    def __call__(self, sample):
        image, label = sample
        axes = [len(image.shape) - 1]
        axes.extend(list(np.arange(len(image.shape)-1)))
        image = mx.ndarray.transpose(image, axes)
        assert len(self.mean) == image.shape[0], 'Mean channel must match image channel'
        transformer = mx.gluon.data.vision.transforms.Normalize(self.mean, self.std)
        image = transformer(image)
        axes = list(np.arange(1, len(image.shape)))
        axes.extend([0])
        image = mx.ndarray.transpose(image, axes)
        return (image, label)

@transform_registry(transform_type="Normalize", process="preprocess", framework="pytorch")
class PyTorchNormalizeTransform(MXNetNormalizeTransform):
    """Normalize a image with mean and standard deviation.

    Args:
        mean (list, default=[0.0]):
            means for each channel, if len(mean)=1, mean will be broadcasted to each channel,
            otherwise its length should be same with the length of image shape
        std (list, default=[1.0]):
            stds for each channel, if len(std)=1, std will be broadcasted to each channel,
            otherwise its length should be same with the length of image shape

    Returns:
        tuple of processed image and label
    """

    def __call__(self, sample):
        image, label = sample
        transformer = torchvision.transforms.Normalize(self.mean, self.std)
        image = transformer(image)
        return (image, label)

@transform_registry(transform_type="Normalize", process="preprocess", \
                framework="onnxrt_qlinearops, onnxrt_integerops")
class NormalizeTransform(BaseTransform):
    """Normalize a image with mean and standard deviation.

    Args:
        mean (list, default=[0.0]):
            means for each channel, if len(mean)=1, mean will be broadcasted to each channel,
            otherwise its length should be same with the length of image shape
        std (list, default=[1.0]):
            stds for each channel, if len(std)=1, std will be broadcasted to each channel,
            otherwise its length should be same with the length of image shape

    Returns:
        tuple of processed image and label
    """

    def __init__(self, mean=[0.0], std=[1.0]):
        self.mean = mean
        self.std = std
        for item in self.std:
            if item < 10**-6:
                raise ValueError("Std should be greater than 0")

    def __call__(self, sample):
        image, label = sample
        assert len(self.mean) == image.shape[-1], 'Mean channel must match image channel'
        image = (image - self.mean) / self.std
        return (image, label)

@transform_registry(transform_type="RandomCrop", process="preprocess", \
                framework="mxnet, onnxrt_qlinearops, onnxrt_integerops")
class RandomCropTransform(BaseTransform):
    """Crop the image at a random location to the given size.

    Args:
        size (list or tuple or int): Size of the result

    Returns:
        tuple of processed image and label
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.height, self.width = size, size
        elif isinstance(size, list) or isinstance(size, tuple):
            if len(size) == 1:
                self.height, self.width = size[0], size[0]
            elif len(size) == 2:
                self.height, self.width = size[0], size[1]

    def __call__(self, sample):
        image, label = sample
        h, w = image.shape[0], image.shape[1]
        if h + 1 < self.height or w + 1 < self.width:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format(
                    (self.height, self.width), (h, w)))

        if self.height == h and self.width == w:
            return (image, label)

        rand_h = np.random.randint(0, h - self.height + 1)
        rand_w = np.random.randint(0, w - self.width + 1)
        if len(image.shape) == 2:
            image = image[rand_h:rand_h + self.height, rand_w:rand_w + self.width]
        else:
            image = image[rand_h:rand_h + self.height, rand_w:rand_w + self.width, :]
        return (image, label)

@transform_registry(transform_type="RandomResizedCrop", process="preprocess", \
                framework="onnxrt_qlinearops, onnxrt_integerops")
class RandomResizedCropTransform(BaseTransform):
    """Crop the given image to random size and aspect ratio.

    Args:
        size (list or int):
            Size of the result
        scale (tuple or list, default=(0.08, 1.0)):
            range of size of the origin size cropped
        ratio (tuple or list, default=(3. / 4., 4. / 3.)):
            range of aspect ratio of the origin aspect ratio cropped
        interpolation (str, default='bilinear'):
            Desired interpolation type, support 'bilinear', 'nearest'

    Returns:
        tuple of processed image and label
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(
            3. / 4., 4. / 3.), interpolation='bilinear'):
        if isinstance(size, int):
            self.size = size, size
        elif isinstance(size, list) or isinstance(size, tuple):
            if len(size) == 1:
                self.size = size[0], size[0]
            elif len(size) == 2:
                self.size = size[0], size[1]

        self.scale = scale
        self.ratio = ratio

        if interpolation in interpolation_map.keys():
            self.interpolation = interpolation_map[interpolation]
        else:
            raise ValueError("Undefined interpolation type")

        if scale[0] > scale[1] or ratio[0] > ratio[1]:
            raise ValueError("Scale and ratio should be of kind (min, max)")

    def get_params(self, image, scale, ratio):
        h, w = image.shape[0], image.shape[1]
        src_area = h * w

        for _ in range(10):
            target_area = np.random.uniform(scale[0], scale[1]) * src_area
            log_ratio = (np.log(ratio[0]), np.log(ratio[1]))
            new_ratio = np.exp(np.random.uniform(log_ratio[0], log_ratio[1]))

            new_w = int(np.round(np.sqrt(target_area * new_ratio)))
            new_h = int(np.round(np.sqrt(target_area / new_ratio)))

            if new_w < w and new_h < h:
                x0 = np.random.randint(0, w - new_w)
                y0 = np.random.randint(0, h - new_h)
                return y0, x0, new_h, new_w

        in_ratio = float(w) / float(h)
        if in_ratio < min(ratio):
            new_w = w
            new_h = int(round(new_w / min(ratio)))
        elif in_ratio > max(ratio):
            new_h = h
            new_w = int(round(new_h * max(ratio)))
        else:
            new_w = w
            new_h = h
        y0 = (h - new_h) // 2
        x0 = (w - new_w) // 2
        return y0, x0, new_h, new_w

    def __call__(self, sample):
        image, label = sample
        y0, x0, h, w = self.get_params(image, self.scale, self.ratio)
        crop_img = image[y0:y0 + h, x0:x0 + w, :]
        image = cv2.resize(crop_img, self.size, interpolation=self.interpolation)
        return (image, label)

def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    import math
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

def get_final_text(pred_text, orig_text, do_lower_case):
    """Project the tokenized prediction back to the original text."""
    import six
    from . import tokenization
    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)
    tok_text = " ".join(tokenizer.tokenize(orig_text))
    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        return orig_text

    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
       return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text

class SquadExample(object):
    """A single training/test example for simple sequence classification.

       For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
        qas_id,
        question_text,
        doc_tokens,
        orig_answer_text=None,
        start_position=None,
        end_position=None,
        is_impossible=False):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

def read_squad_examples(input_file):
    """Read a SQuAD json file into a list of SquadExample."""
    import json
    with tf.io.gfile.GFile(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                examples.append(example)
    return examples

def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, output_fn):
    """Loads a data file into a list of `InputBatch`s."""
    unique_id = 1000000000
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)
        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None

            feature = InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                start_position=start_position,
                end_position=end_position,
                is_impossible=example.is_impossible)
            # Run callback
            output_fn(feature)
            unique_id += 1

@transform_registry(transform_type="Collect", \
                process="postprocess", framework="engine")
class CollectTransform(BaseTransform):
    def __init__(self, length=10833):
        self.length = length
        self.unique_id = []
        self.start_logits = []
        self.end_logits = []
        self.all_sample = (None, None)
        self.idx = 1000000000

    def __call__(self, sample):
        all_results, label = sample
        result_list = [np.expand_dims(result, 0) for result in all_results]
        for result in result_list:
            if len(self.unique_id) < self.length:
                result = result.transpose(2,0,1)
                self.unique_id.append(self.idx)
                self.start_logits.append(result[0])
                self.end_logits.append(result[1])
                self.idx += 1
        if len(self.unique_id) == self.length:
            self.all_sample = ([self.unique_id, self.start_logits, self.end_logits], label)
        return self.all_sample

@transform_registry(transform_type="SquadV1", \
                process="postprocess", framework="tensorflow, engine")
class SquadV1PostTransform(BaseTransform):
    """Postprocess the predictions of bert on SQuAD.

    Args:
        label_file (str): path of label file
        vocab_file(str): path of vocabulary file
        n_best_size (int, default=20):
            The total number of n-best predictions to generate in nbest_predictions.json
        max_seq_length (int, default=384):
            The maximum total input sequence length after WordPiece tokenization.
            Sequences longer than this will be truncated, shorter than this will be padded
        max_query_length (int, default=64):
            The maximum number of tokens for the question.
            Questions longer than this will be truncated to this length
        max_answer_length (int, default=30):
            The maximum length of an answer that can be generated. This is needed because
            the start and end predictions are not conditioned on one another
        do_lower_case (bool, default=True):
            Whether to lower case the input text.
            Should be True for uncased models and False for cased models
        doc_stride (int, default=128):
            When splitting up a long document into chunks,
            how much stride to take between chunks

    Returns:
        tuple of processed prediction and label
    """

    def __init__(self, label_file, vocab_file, n_best_size=20, max_seq_length=384, \
        max_query_length=64, max_answer_length=30, do_lower_case=True, doc_stride=128):

        from . import tokenization
        self.eval_examples = read_squad_examples(label_file)
        tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case)

        self.eval_features = []
        def append_feature(feature):
            self.eval_features.append(feature)

        convert_examples_to_features(
            examples=self.eval_examples,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            output_fn=append_feature)

        self.n_best_size = n_best_size
        self.max_answer_length = max_answer_length
        self.do_lower_case = do_lower_case
        self.RawResult = collections.namedtuple("RawResult",
            ["unique_id", "start_logits", "end_logits"])

    def process_result(self, results):
        processed_results = []
        # notice the result list sequence
        for unique_id, start_logits, end_logits in zip(*results):
            processed_results.append(
                self.RawResult(
                    unique_id=int(unique_id),
                    start_logits=[float(x) for x in start_logits.flat],
                    end_logits=[float(x) for x in end_logits.flat]))

        return processed_results

    def __call__(self, sample):
        if sample == (None, None):
            return (None, None)
        all_results, label = sample
        all_results = self.process_result(all_results)
        example_index_to_features = collections.defaultdict(list)
        for feature in self.eval_features:
            example_index_to_features[feature.example_index].append(feature)

        unique_id_to_result = {}
        for result in all_results:
            unique_id_to_result[result.unique_id] = result

        _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "PrelimPrediction",
            ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

        all_predictions = collections.OrderedDict()
        for (example_index, example) in enumerate(self.eval_examples):
            features = example_index_to_features[example_index]

            prelim_predictions = []
            # keep track of the minimum score of null start+end of position 0
            score_null = 1000000  # large and positive
            min_null_feature_index = 0  # the paragraph slice with min mull score
            null_start_logit = 0  # the start logit at the slice with min null score
            null_end_logit = 0  # the end logit at the slice with min null score
            for (feature_index, feature) in enumerate(features):
                result = unique_id_to_result[feature.unique_id]
                start_indexes = _get_best_indexes(result.start_logits, self.n_best_size)
                end_indexes = _get_best_indexes(result.end_logits, self.n_best_size)

                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # We could hypothetically create invalid predictions, e.g., predict
                        # that the start of the span is in the question. We throw out all
                        # invalid predictions.
                        if start_index >= len(feature.tokens):
                            continue
                        if end_index >= len(feature.tokens):
                            continue
                        if start_index not in feature.token_to_orig_map:
                            continue
                        if end_index not in feature.token_to_orig_map:
                            continue
                        if not feature.token_is_max_context.get(start_index, False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > self.max_answer_length:
                            continue
                        prelim_predictions.append(
                            _PrelimPrediction(
                                feature_index=feature_index,
                                start_index=start_index,
                                end_index=end_index,
                                start_logit=result.start_logits[start_index],
                                end_logit=result.end_logits[end_index]))

                prelim_predictions = sorted(
                    prelim_predictions,
                    key=lambda x: (x.start_logit + x.end_logit),
                    reverse=True)
                _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                    "NbestPrediction", ["text", "start_logit", "end_logit"])

                seen_predictions = {}
                nbest = []
                for pred in prelim_predictions:
                    if len(nbest) >= self.n_best_size:
                        break
                    feature = features[pred.feature_index]
                    if pred.start_index > 0:  # this is a non-null prediction
                        tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                        orig_doc_start = feature.token_to_orig_map[pred.start_index]
                        orig_doc_end = feature.token_to_orig_map[pred.end_index]
                        orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                        tok_text = " ".join(tok_tokens)

                        # De-tokenize WordPieces that have been split off.
                        tok_text = tok_text.replace(" ##", "")
                        tok_text = tok_text.replace("##", "")

                        # Clean whitespace
                        tok_text = tok_text.strip()
                        tok_text = " ".join(tok_text.split())
                        orig_text = " ".join(orig_tokens)

                        final_text = get_final_text(tok_text, orig_text, self.do_lower_case)
                        if final_text in seen_predictions:
                            continue

                        seen_predictions[final_text] = True
                    else:
                        final_text = ""
                        seen_predictions[final_text] = True

                    nbest.append(
                        _NbestPrediction(
                            text=final_text,
                            start_logit=pred.start_logit,
                            end_logit=pred.end_logit))

                # In very rare edge cases we could have no valid predictions. So we
                # just create a nonce prediction in this case to avoid failure.
                if not nbest:
                    nbest.append(
                        _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

                assert len(nbest) >= 1

                total_scores = []
                best_non_null_entry = None
                for entry in nbest:
                    total_scores.append(entry.start_logit + entry.end_logit)
                    if not best_non_null_entry:
                        if entry.text:
                            best_non_null_entry = entry
                probs = _compute_softmax(total_scores)

                nbest_json = []
                for (i, entry) in enumerate(nbest):
                    output = collections.OrderedDict()
                    output["text"] = entry.text
                    output["probability"] = probs[i]
                    output["start_logit"] = entry.start_logit
                    output["end_logit"] = entry.end_logit
                    nbest_json.append(output)

                assert len(nbest_json) >= 1
                all_predictions[example.qas_id] = nbest_json[0]["text"]
        return (all_predictions, label)


@transform_registry(transform_type="ParseDecodeVoc", \
                    process="preprocess", framework="tensorflow")
class ParseDecodeVocTransform(BaseTransform):
    """Parse features in Example proto.

    Returns:
        tuple of parsed image and labels
    """

    def __call__(self, sample):
        # Currently only supports jpeg and png.
        # Need to use this logic because the shape is not known for
        # tf.image.decode_image and we rely on this info to
        # extend label if necessary.
        def _decode_image(content, channels):
            return tf.cond(
                tf.image.is_jpeg(content),
                lambda: tf.image.decode_jpeg(content, channels),
                lambda: tf.image.decode_png(content, channels))

        features = {
            'image/encoded':
                tf.compat.v1.FixedLenFeature((), tf.string, default_value=''),
            'image/filename':
                tf.compat.v1.FixedLenFeature((), tf.string, default_value=''),
            'image/format':
                tf.compat.v1.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/height':
                tf.compat.v1.FixedLenFeature((), tf.int64, default_value=0),
            'image/width':
                tf.compat.v1.FixedLenFeature((), tf.int64, default_value=0),
            'image/segmentation/class/encoded':
                tf.compat.v1.FixedLenFeature((), tf.string, default_value=''),
            'image/segmentation/class/format':
                tf.compat.v1.FixedLenFeature((), tf.string, default_value='png'),
        }

        parsed_features = tf.compat.v1.parse_single_example(sample, features)

        image = _decode_image(parsed_features['image/encoded'], channels=3)

        label = None
        label = _decode_image(
            parsed_features['image/segmentation/class/encoded'], channels=1)

        sample = {
            'image': image,
        }

        label.set_shape([None, None, 1])

        sample['labels_class'] = label

        return sample['image'], sample['labels_class']
