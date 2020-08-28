from abc import abstractmethod
import functools

from ilit.utils.utility import LazyImport, singleton
torchvision = LazyImport('torchvision')
tf = LazyImport('tensorflow')
mx = LazyImport('mxnet')

@singleton
class TensorflowDatasets(object):
    def __init__(self):
        self.datasets = {
            "cifar10" : tf.keras.datasets.cifar10,
            "cifar100" : tf.keras.datasets.cifar100,
            "fashion_mnist" : tf.keras.datasets.fashion_mnist,
            "imdb" : tf.keras.datasets.imdb,
            "mnist" : tf.keras.datasets.mnist,
            "reuters" : tf.keras.datasets.reuters,
            "list_files" : tf.data.Dataset.list_files,
            "TFRecordDataset": tf.data.TFRecordDataset,
            "FixedLengthRecordDataset": tf.data.FixedLengthRecordDataset,
            "TextLineDataset": tf.data.TextLineDataset,
        }
        self.datasets.update(TENSORFLOWDATASETS)

@singleton
class PyTorchDatasets(object):
    def __init__(self):
        self.datasets = {
            'LSUN': torchvision.datasets.LSUN,
            'LSUNClass': torchvision.datasets.LSUNClass,
            'ImageFolder': torchvision.datasets.ImageFolder,
            'DatasetFolder': torchvision.datasets.DatasetFolder,
            'FakeData': torchvision.datasets.FakeData,
            'CocoCaptions': torchvision.datasets.CocoCaptions,
            'CocoDetection': torchvision.datasets.CocoDetection,
            'CIFAR10': torchvision.datasets.CIFAR10,
            'CIFAR100': torchvision.datasets.CIFAR100,
            'EMNIST': torchvision.datasets.EMNIST,
            'FashionMNIST': torchvision.datasets.FashionMNIST,
            'QMNIST': torchvision.datasets.QMNIST,
            'MNIST': torchvision.datasets.MNIST,
            'KMNIST': torchvision.datasets.KMNIST,
            'STL10': torchvision.datasets.STL10, 
            'SVHN': torchvision.datasets.SVHN,
            'PhotoTour': torchvision.datasets.PhotoTour, 
            'SEMEION': torchvision.datasets.SEMEION,
            'Omniglot': torchvision.datasets.Omniglot,
            'SBU': torchvision.datasets.SBU,
            'Flickr8k': torchvision.datasets.Flickr8k, 
            'Flickr30k': torchvision.datasets.Flickr30k,
            'VOCSegmentation': torchvision.datasets.VOCSegmentation, 
            'VOCDetection': torchvision.datasets.VOCDetection, 
            'Cityscapes': torchvision.datasets.Cityscapes, 
            'ImageNet': torchvision.datasets.ImageNet,
            'Caltech101': torchvision.datasets.Caltech101, 
            'Caltech256': torchvision.datasets.Caltech256, 
            'CelebA': torchvision.datasets.CelebA, 
            'SBDataset': torchvision.datasets.SBDataset, 
            'VisionDataset': torchvision.datasets.VisionDataset,
            'USPS': torchvision.datasets.USPS, 
            'Kinetics400': torchvision.datasets.Kinetics400, 
            'HMDB51': torchvision.datasets.HMDB51, 
            'UCF101': torchvision.datasets.UCF101
        }
        self.datasets.update(PYTORCHDATASETS)

@singleton
class MXNetDatasets(object):
    def __init__(self):
        self.datasets = {
            "MNIST": mx.gluon.data.vision.datasets.MNIST,
            "FashionMNIST": mx.gluon.data.vision.datasets.FashionMNIST,
            "CIFAR10": mx.gluon.data.vision.datasets.CIFAR10,
            "CIFAR100": mx.gluon.data.vision.datasets.CIFAR100,
            "ImageRecordDataset": mx.gluon.data.vision.datasets.ImageRecordDataset,
            "ImageFolderDataset": mx.gluon.data.vision.datasets.ImageFolderDataset,
            # "ImageListDataset": mx.gluon.data.vision.datasets.ImageListDataset,
        }
        self.datasets.update(MXNETDATASETS)


framework_datasets = {"tensorflow":TensorflowDatasets,
                      "mxnet":MXNetDatasets,
                      "pytorch":PyTorchDatasets,}

"""The datasets supported by ilit, it's model specific and can be configured by yaml file.

   User could add new datasets by implementing new Dataset subclass under this directory.
   The naming convention of new dataset subclass should be something like ImageClassifier, user
   could choose this dataset by setting "imageclassifier" string in tuning.strategy field of yaml.

   DATASETS variable is used to store all implelmented Dataset subclasses to support
   model specific dataset.
"""
class DATASETS(object):
    def __init__(self, framework):
        assert framework in ["tensorflow", "mxnet", "pytorch"], "framework support tensorflow pytorch mxnet"
        self.datasets = framework_datasets[framework]().datasets 

    def __getitem__(self, dataset_type):
        assert dataset_type in self.datasets.keys(), "dataset type only support {}".format(self.datasets.keys())
        return self.datasets[dataset_type] 

# user/model specific datasets will be registered here
TENSORFLOWDATASETS = {}
MXNETDATASETS = {}
PYTORCHDATASETS = {}

registry_datasets = {"tensorflow":TENSORFLOWDATASETS,
                     "mxnet":MXNETDATASETS,
                     "pytorch":PYTORCHDATASETS,}

def dataset_registry(dataset_type, framework, dataset_format=''):
    """The class decorator used to register all Dataset subclasses.
       

    Args:
        cls (class): The class of register.
        dataset_type (str): The dataset registration name
        framework (str): support 3 framework including 'tensorflow', 'pytorch', 'mxnet'
        data_format (str): The format dataset saved, eg 'raw_image', 'tfrecord'

    Returns:
        cls: The class of register.
    """    
    def decorator_dataset(cls):
        for single_framework in [fwk.strip() for fwk in framework.split(',')]:
            assert single_framework in ["tensorflow", "mxnet", "pytorch"], "The framework support tensorflow mxnet pytorch"
            dataset_name = dataset_type + dataset_format
            if dataset_name in registry_datasets[single_framework].keys():
                raise ValueError('Cannot have two datasets with the same name')
            registry_datasets[single_framework][dataset_name] = cls
        return cls
    return decorator_dataset

class Dataset(object):
    """ The base class of dataset. Subclass datasets should overwrite two methods:
    `__getitem__` for indexing to data sample and `__len__`for the size of the dataset

    """

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    # it's suggested to implement your  __len__ method though we do not set it in abstract class
    # @abstractmethod
    # def __len__(self):
    #     raise NotImplementedError

class IterableDataset(object):
    """An iterable Dataset. Subclass iterable dataset should aslo implement a method:
    `__iter__` for interating over the samples of the dataset.

    """

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

