from abc import abstractmethod
from ilit.utils.utility import LazyImport, singleton

torchvision = LazyImport('torchvision')
torch = LazyImport('torch')
tf = LazyImport('tensorflow')
mx = LazyImport('mxnet')


class BaseTransforms(object):
    def __init__(self, process, concat_general=True):
        transform_map = {"preprocess": self._get_preprocess,
                         "postprocess": self._get_postprocess,
                         "general": self._get_general,}
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

@singleton
class TensorflowTransforms(BaseTransforms):

    def _get_preprocess(self):
        preprocess = {
            "resize" : WrapFunction(tf.image.resize),
            # "resize_with_pad" : WrapFunction(tf.image.resize_with_pad),
            "resize_with_crop_or_pad" : WrapFunction(tf.image.resize_with_crop_or_pad),
            "grayscale_to_rgb" : WrapFunction(tf.image.grayscale_to_rgb),
            "rgb_to_grayscale" : WrapFunction(tf.image.rgb_to_grayscale),
            "hsv_to_rgb" : WrapFunction(tf.image.hsv_to_rgb),
            "rgb_to_hsv" : WrapFunction(tf.image.rgb_to_hsv),
            "yiq_to_rgb" : WrapFunction(tf.image.yiq_to_rgb),
            "rgb_to_yiq" : WrapFunction(tf.image.rgb_to_yiq),
            "yuv_to_rgb" : WrapFunction(tf.image.yuv_to_rgb),
            "rgb_to_yuv" : WrapFunction(tf.image.rgb_to_yuv),
            "image_gradients" : WrapFunction(tf.image.image_gradients),
            "convert_image_dtype" : WrapFunction(tf.image.convert_image_dtype),
            "adjust_brightness" : WrapFunction(tf.image.adjust_brightness),
            "adjust_contrast" : WrapFunction(tf.image.adjust_contrast),
            "adjust_gamma" : WrapFunction(tf.image.adjust_gamma),
            "adjust_hue" : WrapFunction(tf.image.adjust_hue),
            "adjust_jpeg_quality" : WrapFunction(tf.image.adjust_jpeg_quality),
            "adjust_saturation" : WrapFunction(tf.image.adjust_saturation),
            "random_brightness" : WrapFunction(tf.image.random_brightness),
            "random_contrast" : WrapFunction(tf.image.random_contrast),
            "random_saturation" : WrapFunction(tf.image.random_hue),
            "per_image_standardization" : WrapFunction(tf.image.per_image_standardization),
            "central_crop" : WrapFunction(tf.image.central_crop),
            "crop_and_resize" : WrapFunction(tf.image.crop_and_resize),
            "crop_to_bounding_box" : WrapFunction(tf.image.crop_to_bounding_box),
            "extract_glimpse" : WrapFunction(tf.image.extract_glimpse),
            "random_crop" : WrapFunction(tf.image.random_crop),
            "resize_with_crop_or_pad" : WrapFunction(tf.image.resize_with_crop_or_pad),
            "flip_left_right" : WrapFunction(tf.image.flip_left_right),
            "flip_up_down" : WrapFunction(tf.image.flip_up_down),
            "random_flip_left_right" : WrapFunction(tf.image.random_flip_left_right),
            "random_flip_up_down" : WrapFunction(tf.image.random_flip_up_down),
            "rot90" : WrapFunction(tf.image.rot90),
            "decode_and_crop_jpeg" : WrapFunction(tf.io.decode_and_crop_jpeg),
            "decode_bmp" : WrapFunction(tf.io.decode_bmp),
            "decode_gif" : WrapFunction(tf.io.decode_gif),
            "decode_image" : WrapFunction(tf.io.decode_image),
            "decode_jpeg" : WrapFunction(tf.io.decode_jpeg),
            "decode_png" : WrapFunction(tf.io.decode_png),
            "encode_jpeg" : WrapFunction(tf.io.encode_jpeg),
        }
        # update the registry transforms
        preprocess.update(TENSORFLOWTRANSFORMS["preprocess"])
        return preprocess

    def _get_postprocess(self):
        postprocess = {
            "non_max_suppression" : WrapFunction(tf.image.non_max_suppression),
            "non_max_suppression_overlaps" : WrapFunction(tf.image.non_max_suppression_overlaps),
            "non_max_suppression_padded" : WrapFunction(tf.image.non_max_suppression_padded),
            "non_max_suppression_with_scores" : WrapFunction(tf.image.non_max_suppression_with_scores),
            "pad_to_bounding_box" : WrapFunction(tf.image.pad_to_bounding_box),
            "sample_distorted_bounding_box" : WrapFunction(tf.image.sample_distorted_bounding_box),
            "draw_bounding_boxes" : WrapFunction(tf.image.draw_bounding_boxes),
            "combined_non_max_suppression" : WrapFunction(tf.image.combined_non_max_suppression),
            "generate_bounding_box_proposals" : WrapFunction(tf.image.generate_bounding_box_proposals),
        }
        postprocess.update(TENSORFLOWTRANSFORMS["postprocess"])
        return postprocess

    def _get_general(self):
        general = {
            "transpose" : WrapFunction(tf.image.transpose),
        }
        general.update(TENSORFLOWTRANSFORMS["general"])
        return general
 
@singleton
class MXNetTransforms(BaseTransforms):
    def _get_preprocess(self):
        preprocess = {
            'ToTensor': mx.gluon.data.vision.transforms.ToTensor, 
            'Normalize': mx.gluon.data.vision.transforms.Normalize, 
            'Rotate': mx.gluon.data.vision.transforms.Rotate, 
            'RandomRotation': mx.gluon.data.vision.transforms.RandomRotation,
            'RandomResizedCrop': mx.gluon.data.vision.transforms.RandomResizedCrop, 
            'CropResize': mx.gluon.data.vision.transforms.CropResize, 
            'RandomCrop': mx.gluon.data.vision.transforms.RandomCrop,
            'CenterCrop': mx.gluon.data.vision.transforms.CenterCrop, 
            'Resize': mx.gluon.data.vision.transforms.Resize, 
            'RandomFlipLeftRight': mx.gluon.data.vision.transforms.RandomFlipLeftRight, 
            'RandomFlipTopBottom': mx.gluon.data.vision.transforms.RandomFlipTopBottom,
            'RandomBrightness': mx.gluon.data.vision.transforms.RandomBrightness, 
            'RandomContrast': mx.gluon.data.vision.transforms.RandomContrast, 
            'RandomSaturation': mx.gluon.data.vision.transforms.RandomSaturation, 
            'RandomHue': mx.gluon.data.vision.transforms.RandomHue,
            'RandomColorJitter': mx.gluon.data.vision.transforms.RandomColorJitter, 
            'RandomLighting': mx.gluon.data.vision.transforms.RandomLighting, 
            'RandomGray': mx.gluon.data.vision.transforms.RandomGray
        }
        preprocess.update(MXNETTRANSFORMS["preprocess"])
        return preprocess

    def _get_postprocess(self):
        postprocess = {}
        postprocess.update(MXNETTRANSFORMS["postprocess"])
        return postprocess
    
    def _get_general(self):
        general = {
            'Compose': mx.gluon.data.vision.transforms.Compose,
            'HybridCompose': mx.gluon.data.vision.transforms.HybridCompose,
            'Cast': mx.gluon.data.vision.transforms.Cast,
            'RandomApply': mx.gluon.data.vision.transforms.RandomApply,
            'HybridRandomApply': mx.gluon.data.vision.transforms.HybridRandomApply,
        }
        general.update(MXNETTRANSFORMS["general"])
        return general

@singleton
class PyTorchTransforms(BaseTransforms):
    def _get_preprocess(self):
        preprocess = {
            "ToTensor" : torchvision.transforms.ToTensor, 
            "ToPILImage":torchvision.transforms.ToPILImage, 
            "Normalize":torchvision.transforms.Normalize, 
            "Resize":torchvision.transforms.Resize, 
            "Scale":torchvision.transforms.Scale,
            "CenterCrop":torchvision.transforms.CenterCrop, 
            "Pad":torchvision.transforms.Pad, 
            "RandomChoice":torchvision.transforms.RandomChoice, 
            "RandomOrder":torchvision.transforms.RandomOrder, 
            "RandomCrop":torchvision.transforms.RandomCrop,
            "RandomHorizontalFlip":torchvision.transforms.RandomHorizontalFlip, 
            "RandomVerticalFlip":torchvision.transforms.RandomVerticalFlip, 
            "RandomResizedCrop":torchvision.transforms.RandomResizedCrop, 
            "RandomSizedCrop":torchvision.transforms.RandomSizedCrop, 
            "FiveCrop":torchvision.transforms.FiveCrop, 
            "TenCrop":torchvision.transforms.TenCrop,
            "ColorJitter":torchvision.transforms.ColorJitter, 
            "RandomRotation":torchvision.transforms.RandomRotation, 
            "RandomAffine":torchvision.transforms.RandomAffine, 
            "Grayscale":torchvision.transforms.Grayscale, 
            "RandomGrayscale":torchvision.transforms.RandomGrayscale,
            "RandomPerspective":torchvision.transforms.RandomPerspective, 
            "RandomErasing":torchvision.transforms.RandomErasing
        }
        preprocess.update(PYTORCHTRANSFORMS["preprocess"])
        return preprocess
    def _get_postprocess(self):
        postprocess = {}
        postprocess.update(PYTORCHTRANSFORMS["postprocess"])
        return postprocess
    
    def _get_general(self):
        general = {
            "Compose" : torchvision.transforms.Compose, 
            "Lambda":torchvision.transforms.Lambda, 
            "RandomApply":torchvision.transforms.RandomApply, 
            "LinearTransformation":torchvision.transforms.LinearTransformation, 
        }
        general.update(PYTORCHTRANSFORMS["general"])
        return general

framework_transforms = {"tensorflow":TensorflowTransforms,
                        "mxnet":MXNetTransforms,
                        "pytorch":PyTorchTransforms,}


class TRANSFORMS(object):
    def __init__(self, framework, process):
        assert framework in ("tensorflow", "pytorch", "mxnet"), "framework support tensorflow pytorch mxnet"
        assert process in ("preprocess", "postprocess", "general"), "process support preprocess postprocess, general"
        self.transforms = framework_transforms[framework](process).transforms

    def __getitem__(self, transform_type):
        assert transform_type in self.transforms.keys(), "transform support {}".format(self.transforms.keys())
        return self.transforms[transform_type]

# transform registry will register transforms into these dicts
TENSORFLOWTRANSFORMS = {"preprocess": {}, "postprocess":{}, "general": {}}
MXNETTRANSFORMS = {"preprocess": {}, "postprocess":{}, "general": {}}
PYTORCHTRANSFORMS = {"preprocess": {}, "postprocess":{}, "general": {}}

registry_transforms = {"tensorflow": TENSORFLOWTRANSFORMS,
                       "mxnet":MXNETTRANSFORMS,
                       "pytorch":PYTORCHTRANSFORMS,}

def transform_registry(transform_type, process, framework):
    """The class decorator used to register all transform subclasses.
       

    Args:
        transform_type (str): Transform registration name 
        process (str): support 3 process including 'preprocess', 'postprocess', 'general'
        framework (str): support 3 framework including 'tensorflow', 'pytorch', 'mxnet'
        cls (class): The class of register.

    Returns:
        cls: The class of register.
    """    
    def decorator_transform(cls):
        assert framework in ("tensorflow", "mxnet", "pytorch"), "The framework support tensorflow, mxnet and pytorch"
        if transform_type in registry_transforms[framework][process].keys():
            raise ValueError('Cannot have two transforms with the same name')
        registry_transforms[framework][process][transform_type] = cls
        return cls
    return decorator_transform


class Transform(object):
    """The base class for transform. __call__ method is needed when write user specific transform

    """
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

class WrapTransform(Transform):
    def __init__(self, transform_func, **kwargs):
        self.kwargs = kwargs
        self.transform_func = transform_func
    def __call__(self, sample):
        return self.transform_func(sample, **self.kwargs)

# wrap tensorflow functions to a transform
class WrapFunction(object):
    def __init__(self, transform_func):
        self.transform_func = transform_func
    def __call__(self, **kwargs):
        return WrapTransform(self.transform_func, **kwargs)

@transform_registry(transform_type="Compose", process="general", framework="tensorflow")
class ComposeTFTransform(Transform):
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, sample):
        for transform in self.transform_list:
            sample = transform(sample) 
        return sample
