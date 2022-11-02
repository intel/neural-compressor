import os
import argparse
import enum
import tarfile
import abc


class SupportedModels(enum.Enum):
    """
    Enumeration containing supported models
    """
    inception_v1 = 'inception_v1'
    inception_v2 = 'inception_v2'
    inception_v3 = 'inception_v3'
    inception_v4 = 'inception_v4'
    mobilenet_v1 = 'mobilenet_v1'
    mobilenet_v2 = 'mobilenet_v2'
    mobilenet_v3 = 'mobilenet_v3'
    resnet50_v1 = "resnet50_v1"
    resnet101_v1 = "resnet101_v1"
    resnet_v2_50 = "resnet_v2_50"
    resnet_v2_101 = "resnet_v2_101"
    resnet_v2_152 = "resnet_v2_152"


class Model(abc.ABC):
    """
    Base model class used to obtain the model (and perform any necessary operations to make it usable)
    """
    @property
    @abc.abstractmethod
    def model_url(self) -> str:
        """
        Returns the model download url
        Returns: url string

        """
        pass

    @property
    @abc.abstractmethod
    def package_name(self) -> str:
        """
        Returns the downloaded package path
        Returns: path string
        """
        pass

    def get_pretrained_model(self, destination):
        """
        Obtains a ready to use pretrained model file.
        Args:
            destination: path to where the file should be stored
        """
        print("Downloading the model from: {0}".format(self.model_url))
        os.system("curl -o {0} {1}".format(self.package_name, self.model_url))
        if tarfile.is_tarfile(self.package_name):
            with tarfile.open(self.package_name) as tar:
                if not os.path.exists(destination):
                    os.makedirs(destination)
                print("Extracting the model package to {0}".format(destination))
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


class InceptionV1(Model):
    """ Concrete implementation of the Model base class for Inception_v1"""
    # TODO This will download the ckpt file, need to add handling for
    #  https://github.com/tensorflow/models/tree/master/research/slim#exporting-the-inference-graph

    @property
    def model_url(self) -> str:
        """
        Gets model download url
        Returns: model url

        """
        return "http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz"

    @property
    def package_name(self) -> str:
        """
        Gets the package name
        Returns: package name

        """
        return "inception_v1.tar.gz"


class InceptionV2(Model):
    """ Concrete implementation of the Model base class for Inception_v2"""
    # TODO This will download the ckpt file, need to add handling for
    #  https://github.com/tensorflow/models/tree/master/research/slim#exporting-the-inference-graph

    @property
    def model_url(self) -> str:
        """
        Gets model download url
        Returns: model url

        """
        return "http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz"

    @property
    def package_name(self) -> str:
        """
        Gets the package name
        Returns: package name

        """
        return "inception_v2.tar.gz"


class InceptionV3(Model):
    """ Concrete implementation of the Model base class for Inception_v3"""

    @property
    def model_url(self) -> str:
        """
        Gets model download url
        Returns: model url

        """
        return "https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/inceptionv3_fp32_pretrained_model.pb"

    @property
    def package_name(self) -> str:
        """
        Gets the package name
        Returns: package name

        """
        return "inception_v3.pb"


class InceptionV4(Model):
    """ Concrete implementation of the Model base class for Inception_v4"""

    @property
    def model_url(self) -> str:
        """
        Gets model download url
        Returns: model url

        """
        return "https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/inceptionv4_fp32_pretrained_model.pb"

    @property
    def package_name(self) -> str:
        """
        Gets the package name
        Returns: package name

        """
        return "inception_v4.pb"


class Resnet50V1(Model):
    """ Concrete implementation of the Model base class for resnet50_V1"""

    @property
    def model_url(self) -> str:
        """
        Gets model download url
        Returns: model url

        """
        return "https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/resnet50_fp32_pretrained_model.pb"

    @property
    def package_name(self) -> str:
        """
        Gets the package name
        Returns: package name

        """
        return "resnet_v1_50.pb"


class Resnet101V1(Model):
    """ Concrete implementation of the Model base class for resnet101_V1"""

    @property
    def model_url(self) -> str:
        """
        Gets model download url
        Returns: model url

        """
        return "https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/resnet101_fp32_pretrained_model.pb"

    @property
    def package_name(self) -> str:
        """
        Gets the package name
        Returns: package name

        """
        return "resnet_v1_101.pb"


class ResnetV250(Model):
    """ Concrete implementation of the Model base class for Resnet v2 50 """
    def __init__(self):
        raise NotImplementedError("Resnet_V2_50 is not supported yet")

    @property
    def model_url(self) -> str:
        """
        Gets model download url
        Returns: model url

        """
        pass

    @property
    def package_name(self) -> str:
        """
        Gets the package name
        Returns: package name

        """
        pass


class ResnetV2101(Model):
    """ Concrete implementation of the Model base class for Resnet V2 101"""
    def __init__(self):
        raise NotImplementedError("Resnet_V2_101 is not supported yet")

    @property
    def model_url(self) -> str:
        """
        Gets model download url
        Returns: model url

        """
        pass

    @property
    def package_name(self) -> str:
        """
        Gets the package name
        Returns: package name

        """
        pass


class ResnetV2152(Model):
    """ Concrete implementation of the Model base class for Resnet v2 151"""
    def __init__(self):
        raise NotImplementedError("Resnet_V2_152 is not supported yet")

    @property
    def model_url(self) -> str:
        """
        Gets model download url
        Returns: model url

        """
        pass

    @property
    def package_name(self) -> str:
        """
        Gets the package name
        Returns: package name

        """
        pass


class MobilenetV1(Model):
    """ Concrete implementation of the Model base class for Mobilenetv1"""

    @property
    def model_url(self) -> str:
        """
        Gets model download url
        Returns: model url

        """
        return "http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz"

    @property
    def package_name(self) -> str:
        """
        Gets the package name
        Returns: package name

        """
        return "mobilenet_v1_1.0_224.tgz"


class MobilenetV2(Model):
    """ Concrete implementation of the Model base class for Mobilenetv2"""
    @property
    def model_url(self) -> str:
        """
        Gets model download url
        Returns: model url

        """
        return "https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz"

    @property
    def package_name(self) -> str:
        """
        Gets the package name
        Returns: package name

        """
        return "mobilenet_v2_1.0_224.tgz"


class MobilenetV3(Model):
    """ Concrete implementation of the Model base class for Mobilenetv3"""

    @property
    def model_url(self) -> str:
        """
        Gets model download url
        Returns: model url

        """
        return "https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-large_224_1.0_float.tgz"

    @property
    def package_name(self) -> str:
        """
        Gets the package name
        Returns: package name

        """
        return "v3-large_224_1.0_float.tgz"


def get_model(model: SupportedModels) -> Model:
    """
    Factory method that returns the requested model object
    Args:
        model: model from SupportedModels enumeration

    Returns: Concrete object inheriting the Model base class

    """
    model_map = {
        SupportedModels.inception_v1: InceptionV1(),
        SupportedModels.inception_v2: InceptionV2(),
        SupportedModels.inception_v3: InceptionV3(),
        SupportedModels.inception_v4: InceptionV4(),
        SupportedModels.mobilenet_v1: MobilenetV1(),
        SupportedModels.mobilenet_v2: MobilenetV2(),
        SupportedModels.mobilenet_v3: MobilenetV3(),
        SupportedModels.resnet50_v1: Resnet50V1(),
        SupportedModels.resnet101_v1: Resnet101V1()
    }
    return model_map.get(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare pre-trained model for COCO object detection')
    parser.add_argument('--model_name', type=str, default='inception_v1',
                        help='model to download, default is inception_v1',
                        choices=["inception_v1", "inception_v2", "inception_v3", "inception_v4",
                                 "mobilenet_v1", "mobilenet_v2", "mobilenet_v3",
                                 "resnet50_v1", "resnet101_v1",
                                 "resnet_v2_50", "resnet_v2_101", "resnet_v2_152"])
    parser.add_argument('--model_path', type=str, default='{0}/model'.format(os.getcwd()),
                        help='directory to put models, default is {0}/model'.format(os.getcwd()))

    args = parser.parse_args()
    model_name = args.model_name
    model_path = args.model_path
    try:
        model = get_model(SupportedModels(model_name))
        model.get_pretrained_model(model_path)
    except AttributeError:
        print("The model {0} is not supported. Supported models: {1}"
              .format(model_name, SupportedModels.__members__.keys()))
