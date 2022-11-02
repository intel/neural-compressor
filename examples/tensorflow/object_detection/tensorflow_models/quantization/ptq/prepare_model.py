import os
import argparse
import enum
import tarfile
import abc


class SupportedModels(enum.Enum):
    """
    Enumeration containing supported models
    """
    ssd_resnet50_v1 = 'ssd_resnet50_v1'
    ssd_mobilnet_v1 = 'ssd_mobilenet_v1'


class Model(abc.ABC):
    """
    Base model class used to obtain the model (and perform any necessary operations to make it usable)
    """

    @abc.abstractmethod
    def get_pretrained_model(self, destination):
        """
        Base method for obtaining a ready to use model
        Args:
            destination: path to where the file should be stored
        """
        pass


class SsdMobilenetV1(Model):
    """ Concrete implementation of the Model base class for ssd_mobilenet_v1"""

    def get_pretrained_model(self, destination):
        """
        Obtains a ready to use ssd_mobilenet_v1 model file.
        Args:
            destination: path to where the file should be stored
        """
        url = 'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz'
        os.system("curl -o ssd_mobilenet_v1_coco_2018_01_28.tar.gz {0}".format(url))
        with tarfile.open("ssd_mobilenet_v1_coco_2018_01_28.tar.gz") as tar:
            if not os.path.exists(destination):
                os.makedirs(destination)
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, destination)


class SsdResnet50(Model):
    """ Concrete implementation of the Model base class for ssd_resnet_50"""

    def get_pretrained_model(self, destination):
        """
        Obtains a ready to use ssd_resnet_50 model file.
        Args:
            destination: path to where the file should be stored
        """
        url = "http://download.tensorflow.org/models/object_detection/" \
              "ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz"
        os.system("curl -o ssd_resnet50_v1.tar.gz {0}".format(url))
        with tarfile.open("ssd_resnet50_v1.tar.gz") as tar:
            if not os.path.exists(destination):
                os.makedirs(destination)
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, destination)


def get_model(model: SupportedModels) -> Model:
    """
    Factory method that returns the requested model object
    Args:
        model: model from SupportedModels enumeration

    Returns: Concrete object inheriting the Model base class

    """
    if model == SupportedModels.ssd_resnet50_v1:
        return SsdResnet50()
    if model == SupportedModels.ssd_mobilnet_v1:
        return SsdMobilenetV1()
    else:
        raise AttributeError("The model {0} is not supported. Supported models: {1}"
                             .format(model_name, SupportedModels.__members__.keys()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare pre-trained model for COCO object detection')
    parser.add_argument('--model_name', type=str, default='ssd_resnet50_v1',
                        help='model to download, default is ssd_resnet50_v1',
                        choices=["ssd_resnet50_v1", "ssd_mobilenet_v1"])
    parser.add_argument('--model_path', type=str, default='./model', help='directory to put models, default is ./model')

    args = parser.parse_args()
    model_name = args.model_name
    model_path = args.model_path
    try:
        model = get_model(SupportedModels(model_name))
        model.get_pretrained_model(model_path)
    except AttributeError:
        print("The model {0} is not supported. Supported models: {1}"
              .format(model_name, SupportedModels.__members__.keys()))
