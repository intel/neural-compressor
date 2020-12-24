#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Intel Corporation
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
from abc import abstractmethod
from lpot.utils.utility import LazyImport, singleton

torchvision = LazyImport('torchvision')
torch = LazyImport('torch')
tf = LazyImport('tensorflow')
mx = LazyImport('mxnet')


class BaseTransforms(object):
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


class TensorflowTransforms(BaseTransforms):

    def _get_preprocess(self):
        preprocess = {
            "ConvertImageDtype": TensorflowWrapFunction(tf.image.convert_image_dtype),
            "CropToBoundingBox": TensorflowWrapFunction(tf.image.crop_to_bounding_box),
            "RandomHorizontalFlip": TensorflowWrapFunction(tf.image.random_flip_left_right),
            "RandomVerticalFlip": TensorflowWrapFunction(tf.image.random_flip_up_down),
            "DecodeImage": TensorflowWrapFunction(tf.io.decode_image),
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
        general = {
            "Transpose": TensorflowWrapFunction(tf.transpose),
        }
        general.update(TENSORFLOW_TRANSFORMS["general"])
        return general


class MXNetTransforms(BaseTransforms):
    def _get_preprocess(self):
        preprocess = {
            'ToTensor': PytorchMxnetWrapFunction(
                mx.gluon.data.vision.transforms.ToTensor),
            'Normalize': PytorchMxnetWrapFunction(
                mx.gluon.data.vision.transforms.Normalize),
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


class PyTorchTransforms(BaseTransforms):
    def _get_preprocess(self):
        preprocess = {
            "ToTensor": PytorchMxnetWrapFunction(
                torchvision.transforms.ToTensor),
            "ToPILImage": PytorchMxnetWrapFunction(
                torchvision.transforms.ToPILImage),
            "Normalize": PytorchMxnetWrapFunction(
                torchvision.transforms.Normalize),
            "CenterCrop": PytorchMxnetWrapFunction(
                torchvision.transforms.CenterCrop),
            "Pad": PytorchMxnetWrapFunction(
                torchvision.transforms.Pad),
            "RandomCrop": PytorchMxnetWrapFunction(
                torchvision.transforms.RandomCrop),
            "RandomHorizontalFlip": PytorchMxnetWrapFunction(
                torchvision.transforms.RandomHorizontalFlip),
            "RandomVerticalFlip": PytorchMxnetWrapFunction(
                torchvision.transforms.RandomVerticalFlip),
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

class ONNXRTQLTransforms(BaseTransforms):
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


class ONNXRTITTransforms(BaseTransforms):
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

framework_transforms = {"tensorflow": TensorflowTransforms,
                        "mxnet": MXNetTransforms,
                        "pytorch": PyTorchTransforms,
                        "pytorch_ipex": PyTorchTransforms,
                        "onnxrt_qlinearops": ONNXRTQLTransforms,
                        "onnxrt_integerops": ONNXRTITTransforms}

# transform registry will register transforms into these dicts
TENSORFLOW_TRANSFORMS = {"preprocess": {}, "postprocess": {}, "general": {}}
MXNET_TRANSFORMS = {"preprocess": {}, "postprocess": {}, "general": {}}
PYTORCH_TRANSFORMS = {"preprocess": {}, "postprocess": {}, "general": {}}
ONNXRT_QL_TRANSFORMS = {"preprocess": {}, "postprocess": {}, "general": {}}
ONNXRT_IT_TRANSFORMS = {"preprocess": {}, "postprocess": {}, "general": {}}

registry_transforms = {"tensorflow": TENSORFLOW_TRANSFORMS,
                       "mxnet": MXNET_TRANSFORMS,
                       "pytorch": PYTORCH_TRANSFORMS,
                       "pytorch_ipex": PYTORCH_TRANSFORMS,
                       "onnxrt_qlinearops": ONNXRT_QL_TRANSFORMS,
                       "onnxrt_integerops": ONNXRT_IT_TRANSFORMS}

class TRANSFORMS(object):
    def __init__(self, framework, process):
        assert framework in ("tensorflow", "pytorch", "pytorch_ipex", "onnxrt_qlinearops",
                             "onnxrt_integerops", "mxnet"), \
                             "framework support tensorflow pytorch mxnet onnxrt"
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
                "mxnet",
                "pytorch",
                "pytorch_ipex",
                "onnxrt_qlinearops",
                "onnxrt_integerops"
            ], "The framework support tensorflow mxnet pytorch onnxrt"
            if transform_type in registry_transforms[single_framework][process].keys():
                raise ValueError('Cannot have two transforms with the same name')
            registry_transforms[single_framework][process][transform_type] = cls
        return cls
    return decorator_transform


class Transform(object):
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

class TensorflowTransform(Transform):
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


class PytorchMxnetTransform(Transform):
    def __init__(self, transform_func):
        self.transform_func = transform_func

    def __call__(self, sample):
        image, label = sample
        image = self.transform_func(image)
        return (image, label)

interpolation_onnx_map = {
    'nearest': 0,
    'bilinear': 1,
    'bicubic': 3,
}

interpolation_mxnet_map = {
     'nearest': 0,
     'bilinear': 1,
     'bicubic': 2,
}

interpolation_pytorch_map = {
     'nearest': 0,
     'bilinear': 2,
     'bicubic': 3,
}

@transform_registry(transform_type="Compose", process="general", \
                 framework="onnxrt_qlinearops, onnxrt_integerops, tensorflow")
class ComposeONNXRTTransform(Transform):
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, sample):
        for transform in self.transform_list:
            sample = transform(sample)
        return sample

@transform_registry(transform_type="ImageTypeParse", process="preprocess", \
                framework="onnxrt_qlinearops, onnxrt_integerops, tensorflow")
class ImageTypeParse(Transform):
    def __call__(self, sample):
        # TODO: need to add more image type
        image, label = sample
        try:
            image = np.array(image)
        except:
            raise ValueError("Unknown image type!")
        return (image, label)

@transform_registry(transform_type="CenterCrop",
                    process="preprocess", framework="tensorflow")
class CenterCropTFTransform(Transform):
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
        return (image, label)

@transform_registry(transform_type="Resize",
                    process="preprocess", framework="tensorflow")
class ResizeTFTransform(Transform):
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
        image = tf.image.resize(image, self.size, method=self.interpolation)
        return (image, label)

@transform_registry(transform_type="RandomCrop",
                    process="preprocess", framework="tensorflow")
class RandomCropTFTransform(Transform):
    def __init__(self, size):
        self.size = size
    
    def __call__(self, sample):
        image, label = sample
        if len(self.size) != len(image.shape):
            raise ValueError('Shape rank is {} but input shape \
                rank is {}'.format(len(self.size), len(image.shape)))
        image = tf.image.random_crop(image, self.size)
        return (image, label)

@transform_registry(transform_type="RandomResizedCrop",
                    process="preprocess", framework="tensorflow")
class RandomResizedCropTFTransform(Transform):
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
        return (image, label)


@transform_registry(transform_type="Normalize", process="preprocess", framework="tensorflow")
class NormalizeTFTransform(Transform):
    def __init__(self, mean=[0.0], std=[1.0]):
        self.mean = mean
        self.std = std
        for item in self.std:
            if item < 10**-6:
                raise ValueError("Std should be greater than 0")

    def __call__(self, sample):
        image, label = sample
        orig_dtype = image.dtype
        mean = tf.broadcast_to(self.mean, tf.shape(input=image))
        mean = tf.cast(mean, dtype=image.dtype)
        std = tf.broadcast_to(self.std, tf.shape(input=image))
        std = tf.cast(std, dtype=image.dtype)
        image = (image - mean) / std
        image = tf.cast(image, dtype=orig_dtype)
        return (image, label)

@transform_registry(transform_type='Rescale', process="preprocess", \
                framework='tensorflow')
class RescaleTFTransform(Transform):
    def __call__(self, sample):
        image, label = sample
        image = tf.cast(image, tf.float32) / 255.
        return (image, label)

@transform_registry(transform_type="Resize", process="preprocess", \
                framework="mxnet")
class ResizeMxnetTransform(Transform):
    def __init__(self, size, interpolation='bilinear'):
        self.size = size
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
                framework="pytorch")
class ResizePytorchTransform(Transform):
    def __init__(self, size, interpolation='bilinear'):
        self.size = size
        if interpolation in interpolation_pytorch_map.keys():
            self.interpolation = interpolation_pytorch_map[interpolation]
        else:
            raise ValueError("Undefined interpolation type")

    def __call__(self, sample):
        image, label = sample
        transformer = torchvision.transforms.Resize(size=self.size,
                        interpolation=self.interpolation)
        return (transformer(image), label)

@transform_registry(transform_type="Resize", process="preprocess", \
                framework="onnxrt_qlinearops, onnxrt_integerops")
class ResizeONNXTransform(Transform):
    def __init__(self, size, interpolation='bilinear'):
        if isinstance(size, int):
            self.size = size, size  
        elif isinstance(size, list):
            if len(size) == 1:
                self.size = size[0], size[0]
            elif len(size) == 2:
                self.size = size[0], size[1]

        if interpolation in interpolation_onnx_map.keys():
            self.interpolation = interpolation_onnx_map[interpolation]
        else:
            raise ValueError("Undefined interpolation type")

    def __call__(self, sample):
        from skimage.transform import resize
        image, label = sample
        image = resize(image, self.size, order=self.interpolation)
        return (image, label)

@transform_registry(transform_type="CropResize", process="preprocess", \
                framework="mxnet")
class CropResizeMxnetTransform(Transform):
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
                framework="tensorflow")
class CropResizeTFTransform(Transform):
    def __init__(self, x, y, width, height, size, interpolation='bilinear'):
        if interpolation not in ['bilinear', 'nearest']:
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
        squeeze = False
        if len(image.shape) == 3:
            squeeze = True
            image = tf.expand_dims(image, axis=0)
        h, w = image.shape[1:3]
        h = tf.cast(h, dtype=tf.float32)
        w = tf.cast(w, dtype=tf.float32)
        self.y = tf.cast(self.y, dtype=tf.float32)
        self.x = tf.cast(self.x, dtype=tf.float32)
        box_indices = tf.range(0, image.shape[0], dtype=tf.int32)
        boxes = [self.y/h, self.x/w, (self.y+self.height)/h, (self.x+self.width)/w]
        boxes = tf.broadcast_to(boxes, [image.shape[0], 4])
        image = tf.image.crop_and_resize(image, boxes, box_indices, 
                        self.size, self.interpolation)
        if squeeze:
            image = tf.squeeze(image, axis=0)
        return (image, label)

@transform_registry(transform_type="CropResize", process="preprocess", \
                framework="onnxrt_qlinearops, onnxrt_integerops")
class CropResizeTransform(Transform):
    def __init__(self, x, y, width, height, size, interpolation='bilinear'):
        if interpolation in interpolation_onnx_map.keys():
            self.interpolation = interpolation_onnx_map[interpolation]
        else:
            raise ValueError("Undefined interpolation type")
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
        from skimage.transform import resize
        image, label = sample
        image = image[self.y:self.y+self.height, self.x:self.x+self.width, :]
        image = resize(image, self.size, order=self.interpolation)
        return (image, label)

@transform_registry(transform_type="CenterCrop", process="preprocess", \
                framework="onnxrt_qlinearops, onnxrt_integerops")
class CenterCropONNXTransform(Transform):
    def __init__(self, size):
        if isinstance(size, int):
            self.height, self.width = size, size  
        elif isinstance(size, list):
            if len(size) == 1:
                self.height, self.width = size[0], size[0]
            elif len(size) == 2:
                self.height, self.width = size[0], size[1]

    def __call__(self, sample):
        image, label = sample
        h, w, _ = image.shape
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


@transform_registry(transform_type="Normalize", process="preprocess", \
                framework="onnxrt_qlinearops, onnxrt_integerops")
class NormalizeONNXTransform(Transform):
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
class RandomCropONNXTransform(Transform):
    def __init__(self, size):
        if isinstance(size, int):
            self.height, self.width = size, size  
        elif isinstance(size, list):
            if len(size) == 1:
                self.height, self.width = size[0], size[0]
            elif len(size) == 2:
                self.height, self.width = size[0], size[1]

    def __call__(self, sample):
        image, label = sample
        h, w, _ = image.shape
        if h + 1 < self.height or w + 1 < self.width:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format(
                    (self.height, self.width), (h, w)))

        if self.height == h and self.width == w:
            return (image, label)

        rand_h = np.random.randint(0, h - self.height + 1)
        rand_w = np.random.randint(0, w - self.width + 1)
        image = image[rand_h:rand_h + self.height, rand_w:rand_w + self.width, :]
        return (image, label)

@transform_registry(transform_type="RandomResizedCrop", process="preprocess", \
                framework="pytorch")
class RandomResizedCropPytorchTransform(Transform):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(
            3. / 4., 4. / 3.), interpolation='bilinear'):
        self.size = size
        self.scale = scale
        self.ratio = ratio

        if interpolation in interpolation_pytorch_map.keys():
            self.interpolation = interpolation_pytorch_map[interpolation]
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
class RandomResizedCropMxnetTransform(Transform):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(
            3. / 4., 4. / 3.), interpolation='bilinear'):
        self.size = size
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

@transform_registry(transform_type="RandomResizedCrop", process="preprocess", \
                framework="onnxrt_qlinearops, onnxrt_integerops")
class RandomResizedCropONNXTransform(Transform):
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

        if interpolation in interpolation_onnx_map.keys():
            self.interpolation = interpolation_onnx_map[interpolation]
        else:
            raise ValueError("Undefined interpolation type")

        if scale[0] > scale[1] or ratio[0] > ratio[1]:
            raise ValueError("Scale and ratio should be of kind (min, max)")

    def get_params(self, image, scale, ratio):
        h, w, _ = image.shape
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
        from skimage.transform import resize
        image, label = sample
        y0, x0, h, w = self.get_params(image, self.scale, self.ratio)
        crop_img = image[y0:y0 + h, x0:x0 + w, :]
        image = resize(crop_img, self.size, order=self.interpolation)
        return (image, label)
