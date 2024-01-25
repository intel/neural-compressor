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
"""This is the base class for each framework."""

import os
from abc import abstractmethod

from PIL import Image

from neural_compressor.utils.utility import LazyImport, singleton

torch = LazyImport("torch")
torchvision = LazyImport("torchvision")
tf = LazyImport("tensorflow")
mx = LazyImport("mxnet")
np = LazyImport("numpy")
hashlib = LazyImport("hashlib")
gzip = LazyImport("gzip")
tarfile = LazyImport("tarfile")
zipfile = LazyImport("zipfile")
pickle = LazyImport("pickle")
glob = LazyImport("glob")


@singleton
class TensorflowDatasets(object):  # pragma: no cover
    """The base class of Tensorflow datasets class."""

    def __init__(self):
        """Initialize the attributes of class."""
        self.datasets = {}
        self.datasets.update(TENSORFLOW_DATASETS)


@singleton
class PyTorchDatasets(object):  # pragma: no cover
    """The base class of PyTorch datasets class."""

    def __init__(self):
        """Initialize the attributes of class."""
        self.datasets = {
            "ImageFolder": PytorchMxnetWrapDataset(torchvision.datasets.ImageFolder),
        }
        self.datasets.update(PYTORCH_DATASETS)


@singleton
class MXNetDatasets(object):  # pragma: no cover
    """The base class of MXNet datasets class."""

    def __init__(self):
        """Initialize the attributes of class."""
        self.datasets = {}
        self.datasets.update(MXNET_DATASETS)


@singleton
class ONNXRTQLDatasets(object):  # pragma: no cover
    """The base class of ONNXRT QLinear datasets class."""

    def __init__(self):
        """Initialize the attributes of class."""
        self.datasets = {}
        self.datasets.update(ONNXRTQL_DATASETS)


@singleton
class ONNXRTITDatasets(object):  # pragma: no cover
    """The base class of ONNXRT IT datasets class."""

    def __init__(self):
        """Initialize the attributes of class."""
        self.datasets = {}
        self.datasets.update(ONNXRTIT_DATASETS)


class PytorchMxnetWrapDataset:  # pragma: no cover
    """The base class for PyTorch and MXNet frameworks.

    Args:
        datafunc: The datasets class of PyTorch or MXNet.
    """

    def __init__(self, datafunc):
        """Initialize the attributes of class."""
        self.datafunc = datafunc

    def __call__(self, transform=None, filter=None, *args, **kwargs):
        """Wrap the dataset for PyTorch and MXNet framework."""
        return PytorchMxnetWrapFunction(self.datafunc, transform=transform, filter=filter, *args, **kwargs)


class PytorchMxnetWrapFunction:  # pragma: no cover
    """The Helper class for PytorchMxnetWrapDataset.

    Args:
        dataset (datasets class): The datasets class of PyTorch or MXNet.
        transform (transform object):  transform to process input data.
        filter (Filter objects): filter out examples according to specific
                                 conditions.
    """

    def __init__(self, dataset, transform, filter, *args, **kwargs):
        """Initialize the attributes of class."""
        self.dataset = dataset(*args, **kwargs)
        self.transform = transform
        self.filter = filter

    def __len__(self):
        """Length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        """Magic method.

        x[i] is roughly equivalent to type(x).__getitem__(x, index)
        """
        sample = self.dataset[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


framework_datasets = {
    "tensorflow": TensorflowDatasets,
    "tensorflow_itex": TensorflowDatasets,
    "mxnet": MXNetDatasets,
    "pytorch": PyTorchDatasets,
    "pytorch_ipex": PyTorchDatasets,
    "pytorch_fx": PyTorchDatasets,
    "onnxrt_qdq": ONNXRTQLDatasets,
    "onnxrt_qlinearops": ONNXRTQLDatasets,
    "onnxruntime": ONNXRTQLDatasets,
    "onnxrt_integerops": ONNXRTITDatasets,
}
"""The datasets supported by neural_compressor, it's model specific and can be configured by yaml file.

User could add new datasets by implementing new Dataset subclass under this directory.
The naming convention of new dataset subclass should be something like ImageClassifier, user
could choose this dataset by setting "imageclassifier" string in tuning.strategy field of yaml.

Datasets variable is used to store all implemented Dataset subclasses to support
model specific dataset.
"""


class Datasets(object):  # pragma: no cover
    """A base class for all framework datasets.

    Args:
        framework (str): framework name, like:"tensorflow", "tensorflow_itex", "keras",
                         "mxnet", "onnxrt_qdq", "onnxrt_qlinearops", "onnxrt_integerops",
                         "pytorch", "pytorch_ipex", "pytorch_fx", "onnxruntime".
    """

    def __init__(self, framework):
        """Initialize the attributes of class."""
        assert framework in [
            "tensorflow",
            "tensorflow_itex",
            "keras",
            "mxnet",
            "onnxrt_qdq",
            "onnxrt_qlinearops",
            "onnxrt_integerops",
            "pytorch",
            "pytorch_ipex",
            "pytorch_fx",
            "onnxruntime",
        ], "framework support tensorflow pytorch mxnet onnxrt"
        self.datasets = framework_datasets[framework]().datasets

    def __getitem__(self, dataset_type):
        """Magic method.

        x[i] is roughly equivalent to type(x).__getitem__(x, index)
        """
        assert dataset_type in self.datasets.keys(), "dataset type only support {}".format(self.datasets.keys())
        return self.datasets[dataset_type]


# user/model specific datasets will be registered here
TENSORFLOW_DATASETS = {}
TENSORFLOWITEX_DATASETS = {}
MXNET_DATASETS = {}
PYTORCH_DATASETS = {}
PYTORCHIPEX_DATASETS = {}
PYTORCHFX_DATASETS = {}
ONNXRTQL_DATASETS = {}
ONNXRTIT_DATASETS = {}

registry_datasets = {
    "tensorflow": TENSORFLOW_DATASETS,
    "tensorflow_itex": TENSORFLOWITEX_DATASETS,
    "mxnet": MXNET_DATASETS,
    "pytorch": PYTORCH_DATASETS,
    "pytorch_ipex": PYTORCHIPEX_DATASETS,
    "pytorch_fx": PYTORCHFX_DATASETS,
    "onnxrt_integerops": ONNXRTIT_DATASETS,
    "onnxrt_qdq": ONNXRTQL_DATASETS,
    "onnxruntime": ONNXRTQL_DATASETS,
    "onnxrt_qlinearops": ONNXRTQL_DATASETS,
}


def dataset_registry(dataset_type, framework, dataset_format=""):  # pragma: no cover
    """Register dataset subclasses.

    Args:
        cls (class): The class of register.
        dataset_type (str): The dataset registration name
        framework (str): support 3 framework including 'tensorflow', 'pytorch', 'mxnet'
        data_format (str): The format dataset saved, eg 'raw_image', 'tfrecord'

    Returns:
        cls: The class of register.
    """

    def decorator_dataset(cls):
        for single_framework in [fwk.strip() for fwk in framework.split(",")]:
            assert single_framework in [
                "tensorflow",
                "tensorflow_itex",
                "mxnet",
                "pytorch",
                "pytorch_ipex",
                "pytorch_fx",
                "onnxrt_qlinearops",
                "onnxrt_integerops",
                "onnxrt_qdq",
                "onnxruntime",
            ], "The framework support tensorflow mxnet pytorch onnxrt"
            dataset_name = dataset_type + dataset_format
            if dataset_name in registry_datasets[single_framework].keys():
                raise ValueError("Cannot have two datasets with the same name")
            registry_datasets[single_framework][dataset_name] = cls
        return cls

    return decorator_dataset


class Dataset(object):  # pragma: no cover
    """The base class of dataset.

    Subclass datasets should overwrite two methods:
    `__getitem__` for indexing to data sample and `__len__`for the size of the dataset
    """

    @abstractmethod
    def __getitem__(self, index):
        """Magic method.

        x[i] is roughly equivalent to type(x).__getitem__(x, index)
        """
        raise NotImplementedError

    # it's suggested to implement your  __len__ method though we do not set it in abstract class
    # @abstractmethod
    # def __len__(self):
    #     raise NotImplementedError


class IterableDataset(object):  # pragma: no cover
    """An iterable Dataset.

    Subclass iterable dataset should also implement a method:
    `__iter__` for iterating over the samples of the dataset.
    """

    @abstractmethod
    def __iter__(self):
        """Magic method.

        Returns the iterator object itself.
        """
        raise NotImplementedError


def download_url(url, root, filename=None, md5=None):  # pragma: no cover
    """Download from url.

    Args:
        url (str): the address to download from.
        root (str): the path for saving.
        filename (str): the file name for saving.
        md5 (str): the md5 string.
    """
    import ssl
    import urllib

    ssl._create_default_https_context = ssl._create_unverified_context

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    if check_integrity(fpath, md5):
        print("Using downloaded and verified file: " + fpath)
    else:
        try:
            print("Downloading " + url + " to " + fpath)
            urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater())
        except (urllib.error.URLError, IOError) as e:
            if url[:5] == "https":
                url = url.replace("https:", "http:")
                print("Failed download. Trying https -> http instead." " Downloading " + url + " to " + fpath)
                urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater())
            else:
                raise e
        if not check_integrity(fpath, md5):
            raise RuntimeError("File not found or corrupted.")


def gen_bar_updater():  # pragma: no cover
    """Generate progress bar."""
    from tqdm import tqdm

    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        """Update progress bar."""
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def check_integrity(fpath, md5):  # pragma: no cover
    """Check MD5 checksum."""
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return md5 == calculate_md5(fpath)


def calculate_md5(fpath, chunk_size=1024 * 1024):  # pragma: no cover
    """Generate MD5 checksum for a file."""
    md5 = hashlib.md5()  # nosec
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


@dataset_registry(
    dataset_type="CIFAR10",
    framework="onnxrt_qlinearops, \
                    onnxrt_integerops",
    dataset_format="",
)
class CIFAR10(Dataset):  # pragma: no cover
    """The CIFAR10 and CIFAR100 database.

    For CIFAR10: If download is True, it will download dataset to root/ and extract it
                 automatically, otherwise user can download file from
                 https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz manually to
                 root/ and extract it.
    For CIFAR100: If download is True, it will download dataset to root/ and extract it
                  automatically, otherwise user can download file from
                  https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz manually to
                  root/ and extract it.

    Args:
        root (str): Root directory of dataset.
        train (bool, default=False): If True, creates dataset from train subset,
                                     otherwise from validation subset.
        transform (transform object, default=None):  transform to process input data.
        filter (Filter objects, default=None): filter out examples according to specific
                                               conditions.
        download (bool, default=True): If true, downloads the dataset from the internet
                                       and puts it in root directory. If dataset is already
                                       downloaded, it is not downloaded again.
    """

    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]

    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(self, root, train=False, transform=None, filter=None, download=True):  # pragma: no cover
        """Initialize the attributes of class."""
        self.root = root
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")
        if train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self.load_meta()
        self.transform = transform

    def load_meta(self):  # pragma: no cover
        """Load meta."""
        path = os.path.join(self.root, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError(
                "Dataset metadata file not found or corrupted." + " You can use download=True to download it"
            )
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):  # pragma: no cover
        """Magic method.

        x[i] is roughly equivalent to type(x).__getitem__(x, index)
        """
        image, label = self.data[index], self.targets[index]
        if self.transform is not None:
            image, label = self.transform((image, label))
        return image, label

    def __len__(self):  # pragma: no cover
        """Length of the dataset."""
        return len(self.data)

    def download(self):  # pragma: no cover
        """Download a file."""
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_root = os.path.expanduser(self.root)
        filename = os.path.basename(self.url)
        download_url(self.url, download_root, filename, self.tgz_md5)
        archive = os.path.join(download_root, filename)
        print("Extracting {} to {}".format(archive, download_root))
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=download_root)

    def _check_integrity(self):  # pragma: no cover
        """Check MD5 checksum."""
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, filename)
            if not check_integrity(fpath, md5):
                return False
            return True


@dataset_registry(dataset_type="CIFAR10", framework="pytorch", dataset_format="")
class PytorchCIFAR10(CIFAR10):
    """The PyTorch datasets for CIFAR10."""

    def __getitem__(self, index):  # pragma: no cover
        """Magic method.

        x[i] is roughly equivalent to type(x).__getitem__(x, index)
        """
        image, label = self.data[index], self.targets[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image, label = self.transform((image, label))
        return (image, label)


@dataset_registry(dataset_type="CIFAR10", framework="mxnet", dataset_format="")
class MXNetCIFAR10(CIFAR10):
    """The MXNet datasets for CIFAR10."""

    def __getitem__(self, index):  # pragma: no cover
        """Magic method.

        x[i] is roughly equivalent to type(x).__getitem__(x, index)
        """
        image, label = self.data[index], self.targets[index]
        image = mx.nd.array(image)
        if self.transform is not None:
            image, label = self.transform((image, label))
        return (image, label)


@dataset_registry(dataset_type="CIFAR10", framework="tensorflow, tensorflow_itex", dataset_format="")
class TensorflowCIFAR10(CIFAR10):
    """The Tensorflow datasets for CIFAR10."""

    def __getitem__(self, index):  # pragma: no cover
        """Magic method.

        x[i] is roughly equivalent to type(x).__getitem__(x, index)
        """
        image, label = self.data[index], self.targets[index]
        if self.transform is not None:
            image, label = self.transform((image, label))
        if type(image).__name__ == "Tensor":
            with tf.compat.v1.Session() as sess:
                image = sess.run(image)
        elif type(image).__name__ == "EagerTensor":
            image = image.numpy()
        return (image, label)


@dataset_registry(
    dataset_type="CIFAR100",
    framework="onnxrt_qlinearops, \
                    onnxrt_integerops",
    dataset_format="",
)
class CIFAR100(CIFAR10):
    """CIFAR100 database.

    For CIFAR100: If download is True, it will download dataset to root/ and extract it
                  automatically, otherwise user can download file from
                  https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz manually to
                  root/ and extract it.

    Args:
        root (str): Root directory of dataset.
        train (bool, default=False): If True, creates dataset from train subset,
                                     otherwise from validation subset.
        transform (transform object, default=None):  transform to process input data.
        filter (Filter objects, default=None): filter out examples according to specific
                                               conditions.
        download (bool, default=True): If true, downloads the dataset from the internet
                                       and puts it in root directory. If dataset is already
                                       downloaded, it is not downloaded again.
    """

    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]
    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }


@dataset_registry(dataset_type="CIFAR100", framework="pytorch", dataset_format="")
class PytorchCIFAR100(CIFAR100):
    """The PyTorch datasets for CIFAR100."""

    def __getitem__(self, index):  # pragma: no cover
        """Magic method.

        x[i] is roughly equivalent to type(x).__getitem__(x, index)
        """
        image, label = self.data[index], self.targets[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image, label = self.transform((image, label))
        image = np.array(image)
        return (image, label)


@dataset_registry(dataset_type="CIFAR100", framework="mxnet", dataset_format="")
class MXNetCIFAR100(CIFAR100):
    """The MXNet datasets for CIFAR100."""

    def __getitem__(self, index):  # pragma: no cover
        """Magic method.

        x[i] is roughly equivalent to type(x).__getitem__(x, index)
        """
        image, label = self.data[index], self.targets[index]
        image = mx.nd.array(image)
        if self.transform is not None:
            image, label = self.transform((image, label))
        return (image, label)


@dataset_registry(dataset_type="CIFAR100", framework="tensorflow, tensorflow_itex", dataset_format="")
class TensorflowCIFAR100(CIFAR100):
    """The Tensorflow datasets for CIFAR100."""

    def __getitem__(self, index):  # pragma: no cover
        """Magic method.

        x[i] is roughly equivalent to type(x).__getitem__(x, index)
        """
        image, label = self.data[index], self.targets[index]
        if self.transform is not None:
            image, label = self.transform((image, label))
        if type(image).__name__ == "Tensor":
            with tf.compat.v1.Session() as sess:
                image = sess.run(image)
        elif type(image).__name__ == "EagerTensor":
            image = image.numpy()
        return (image, label)


@dataset_registry(
    dataset_type="MNIST",
    framework="onnxrt_qlinearops, \
                    onnxrt_integerops",
    dataset_format="",
)
class MNIST(Dataset):  # pragma: no cover
    """Modified National Institute of Standards and Technology database and FashionMNIST database.

    For MNIST: If download is True, it will download dataset to root/MNIST/, otherwise user
               should put mnist.npz under root/MNIST/ manually.
    For FashionMNIST: If download is True, it will download dataset to root/FashionMNIST/,
                      otherwise user should put train-labels-idx1-ubyte.gz,
                      train-images-idx3-ubyte.gz, t10k-labels-idx1-ubyte.gz
                      and t10k-images-idx3-ubyte.gz under root/FashionMNIST/ manually.

    Args:
        root (str): Root directory of dataset.
        train (bool, default=False): If True, creates dataset from train subset,
                                     otherwise from validation subset.
        transform (transform object, default=None):  transform to process input data.
        filter (Filter objects, default=None): filter out examples according to specific
                                               conditions.
        download (bool, default=True): If true, downloads the dataset from the internet
                                       and puts it in root directory. If dataset is already
                                       downloaded, it is not downloaded again.
    """

    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]
    resource = [
        ("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz", "8a61469f7ea1b51cbae51d4f78837e45")
    ]

    def __init__(self, root, train=False, transform=None, filter=None, download=True):
        """Initialize the attributes of class."""
        self.root = root
        self.train = train
        self.transform = transform
        if download:
            self.download()

        self.read_data()

    def read_data(self):
        """Read data from a file."""
        for file_name, checksum in self.resource:
            file_path = os.path.join(self.root, os.path.basename(file_name))
            if not os.path.exists(file_path):
                raise RuntimeError("Dataset not found. You can use download=True to download it")
            with np.load(file_path, allow_pickle=True) as f:
                if self.train:
                    self.data, self.targets = f["x_train"], f["y_train"]
                else:
                    self.data, self.targets = f["x_test"], f["y_test"]

    def __len__(self):
        """Length of the dataset."""
        return len(self.data)

    def __getitem__(self, index):
        """Magic method.

        x[i] is roughly equivalent to type(x).__getitem__(x, index)
        """
        image, label = self.data[index], int(self.targets[index])
        image = np.expand_dims(image, -1)
        if self.transform is not None:
            image, label = self.transform((image, label))
        return image, label

    @property
    def class_to_idx(self):
        """Return a dict of class."""
        return {_class: i for i, _class in enumerate(self.classes)}

    def download(self):
        """Download a file."""
        for url, md5 in self.resource:
            filename = os.path.basename(url)
            if os.path.exists(os.path.join(self.root, filename)):
                continue
            else:
                download_url(url, root=self.root, filename=filename, md5=md5)


@dataset_registry(dataset_type="MNIST", framework="pytorch", dataset_format="")
class PytorchMNIST(MNIST):  # pragma: no cover
    """The PyTorch datasets for MNIST."""

    def __getitem__(self, index):
        """Magic method.

        x[i] is roughly equivalent to type(x).__getitem__(x, index)
        """
        image, label = self.data[index], int(self.targets[index])
        image = Image.fromarray(image, mode="L")
        if self.transform is not None:
            image, label = self.transform((image, label))
        image = np.array(image)
        return (image, label)


@dataset_registry(dataset_type="MNIST", framework="mxnet", dataset_format="")
class MXNetMNIST(MNIST):  # pragma: no cover
    """The MXNet datasets for MNIST."""

    def __getitem__(self, index):
        """Magic method.

        x[i] is roughly equivalent to type(x).__getitem__(x, index)
        """
        image, label = self.data[index], int(self.targets[index])
        image = mx.nd.array(image)
        image = image.reshape((image.shape[0], image.shape[1], 1))
        if self.transform is not None:
            image, label = self.transform((image, label))
        return (image, label)


@dataset_registry(dataset_type="MNIST", framework="tensorflow, tensorflow_itex", dataset_format="")
class TensorflowMNIST(MNIST):  # pragma: no cover
    """The Tensorflow datasets for MNIST."""

    def __getitem__(self, index):
        """Magic method.

        x[i] is roughly equivalent to type(x).__getitem__(x, index)
        """
        image, label = self.data[index], int(self.targets[index])
        image = np.expand_dims(image, -1)
        if self.transform is not None:
            image, label = self.transform((image, label))
        if type(image).__name__ == "Tensor":
            with tf.compat.v1.Session() as sess:
                image = sess.run(image)
        elif type(image).__name__ == "EagerTensor":
            image = image.numpy()
        return (image, label)


@dataset_registry(
    dataset_type="FashionMNIST",
    framework="onnxrt_qlinearops, \
                    onnxrt_integerops",
    dataset_format="",
)
class FashionMNIST(MNIST):  # pragma: no cover
    """FashionMNIST database.

    For FashionMNIST: If download is True, it will download dataset to root/FashionMNIST/,
                      otherwise user should put train-labels-idx1-ubyte.gz,
                      train-images-idx3-ubyte.gz, t10k-labels-idx1-ubyte.gz
                      and t10k-images-idx3-ubyte.gz under root/FashionMNIST/ manually.

    Args:
        root (str): Root directory of dataset.
        train (bool, default=False): If True, creates dataset from train subset,
                                     otherwise from validation subset.
        transform (transform object, default=None):  transform to process input data.
        filter (Filter objects, default=None): filter out examples according to specific
                                               conditions.
        download (bool, default=True): If true, downloads the dataset from the internet
                                       and puts it in root directory. If dataset is already
                                       downloaded, it is not downloaded again.
    """

    resource = [
        ("https://storage.googleapis.com/tensorflow/tf-keras-datasets/" + file_name, None)
        for file_name in [
            "train-labels-idx1-ubyte.gz",
            "train-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
        ]
    ]

    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    def read_data(self):
        """Read data from a file."""
        import struct

        if self.train:
            label_path = os.path.join(self.root, "train-labels-idx1-ubyte.gz")
            image_path = os.path.join(self.root, "train-images-idx3-ubyte.gz")
        else:
            label_path = os.path.join(self.root, "t10k-labels-idx1-ubyte.gz")
            image_path = os.path.join(self.root, "t10k-images-idx3-ubyte.gz")
        with gzip.open(label_path, "rb") as f:
            struct.unpack(">II", f.read(8))
            self.targets = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int32)
        with gzip.open(image_path, "rb") as f:
            struct.unpack(">IIII", f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8)
            self.data = data.reshape(len(self.targets), 28, 28)


@dataset_registry(dataset_type="FashionMNIST", framework="pytorch", dataset_format="")
class PytorchFashionMNIST(FashionMNIST):  # pragma: no cover
    """The PyTorch datasets for FashionMNIST."""

    def __getitem__(self, index):
        """Magic method.

        x[i] is roughly equivalent to type(x).__getitem__(x, index)
        """
        image, label = self.data[index], int(self.targets[index])
        image = Image.fromarray(image, mode="L")
        if self.transform is not None:
            image, label = self.transform((image, label))
        image = np.array(image)
        return (image, label)


@dataset_registry(dataset_type="FashionMNIST", framework="mxnet", dataset_format="")
class MXNetFashionMNIST(FashionMNIST):  # pragma: no cover
    """The MXNet Dataset for FashionMNIST."""

    def __getitem__(self, index):
        """Magic method.

        x[i] is roughly equivalent to type(x).__getitem__(x, index)
        """
        image, label = self.data[index], int(self.targets[index])
        image = mx.nd.array(image)
        image = image.reshape((image.shape[0], image.shape[1], 1))
        if self.transform is not None:
            image, label = self.transform((image, label))
        return (image, label)


@dataset_registry(dataset_type="FashionMNIST", framework="tensorflow, tensorflow_itex", dataset_format="")
class TensorflowFashionMNIST(FashionMNIST):  # pragma: no cover
    """The Tensorflow Dataset for FashionMNIST."""

    def __getitem__(self, index):
        """Magic method.

        x[i] is roughly equivalent to type(x).__getitem__(x, index)
        """
        image, label = self.data[index], int(self.targets[index])
        image = np.expand_dims(image, -1)
        if self.transform is not None:
            image, label = self.transform((image, label))
        if type(image).__name__ == "Tensor":
            with tf.compat.v1.Session() as sess:
                image = sess.run(image)
        elif type(image).__name__ == "EagerTensor":
            image = image.numpy()
        return (image, label)


@dataset_registry(
    dataset_type="ImageFolder",
    framework="onnxrt_qlinearops, \
                    onnxrt_integerops",
    dataset_format="",
)
class ImageFolder(Dataset):  # pragma: no cover
    """The base class for ImageFolder.

    Expects the data folder to contain subfolders representing the classes to which
    its images belong.

    Please arrange data in this way:
        root/class_1/xxx.png
        root/class_1/xxy.png
        root/class_1/xxz.png
        ...
        root/class_n/123.png
        root/class_n/nsdf3.png
        root/class_n/asd932_.png
    Please put images of different categories into different folders.

    Args: root (str): Root directory of dataset.
          transform (transform object, default=None):  transform to process input data.
          filter (Filter objects, default=None): filter out examples according to specific
                                                 conditions.
    """

    def __init__(self, root, transform=None, filter=None):
        """Initialize the attributes of class."""
        self.root = root
        assert os.path.exists(self.root), "Datapath doesn't exist!"

        self.transform = transform
        self.image_list = []
        files = glob.glob(os.path.join(self.root, "*"))
        files.sort()
        for idx, file in enumerate(files):
            imgs = glob.glob(os.path.join(file, "*"))
            imgs.sort()
            for img in imgs:
                self.image_list.append((img, idx))

    def __len__(self):
        """Length of the dataset."""
        return len(self.image_list)

    def __getitem__(self, index):
        """Magic method.

        x[i] is roughly equivalent to type(x).__getitem__(x, index)
        """
        sample = self.image_list[index]
        label = sample[1]
        with Image.open(sample[0]) as image:
            image = np.array(image)
            if self.transform is not None:
                image, label = self.transform((image, label))
            return (image, label)


@dataset_registry(dataset_type="ImageFolder", framework="mxnet", dataset_format="")
class MXNetImageFolder(ImageFolder):  # pragma: no cover
    """The MXNet Dataset for image folder.

    Expects the data folder to contain subfolders representing the classes to which
    its images belong.

    Please arrange data in this way:
        root/class_1/xxx.png
        root/class_1/xxy.png
        root/class_1/xxz.png
        ...
        root/class_n/123.png
        root/class_n/nsdf3.png
        root/class_n/asd932_.png
    Please put images of different categories into different folders.

    Args: root (str): Root directory of dataset.
          transform (transform object, default=None):  transform to process input data.
          filter (Filter objects, default=None): filter out examples according to specific
                                                 conditions.
    """

    def __getitem__(self, index):
        """Magic method.

        x[i] is roughly equivalent to type(x).__getitem__(x, index)
        """
        sample = self.image_list[index]
        label = sample[1]
        image = mx.image.imread(sample[0])
        if self.transform is not None:
            image, label = self.transform((image, label))
        return (image, label)


@dataset_registry(dataset_type="ImageFolder", framework="tensorflow, tensorflow_itex", dataset_format="")
class Tensorflow(ImageFolder):  # pragma: no cover
    """The Tensorflow Dataset for image folder.

    Expects the data folder to contain subfolders representing the classes to which
    its images belong.

    Please arrange data in this way:
        root/class_1/xxx.png
        root/class_1/xxy.png
        root/class_1/xxz.png
        ...
        root/class_n/123.png
        root/class_n/nsdf3.png
        root/class_n/asd932_.png
    Please put images of different categories into different folders.

    Args: root (str): Root directory of dataset.
          transform (transform object, default=None):  transform to process input data.
          filter (Filter objects, default=None): filter out examples according to specific
                                                 conditions.
    """

    def __getitem__(self, index):
        """Magic method.

        x[i] is roughly equivalent to type(x).__getitem__(x, index)
        """
        sample = self.image_list[index]
        label = sample[1]
        with Image.open(sample[0]) as image:
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = np.array(image)
            if self.transform is not None:
                image, label = self.transform((image, label))
            if type(image).__name__ == "Tensor":
                with tf.compat.v1.Session() as sess:
                    image = sess.run(image)
            elif type(image).__name__ == "EagerTensor":
                image = image.numpy()
            return (image, label)


@dataset_registry(dataset_type="TFRecordDataset", framework="tensorflow, tensorflow_itex", dataset_format="")
class TensorflowTFRecordDataset(IterableDataset):  # pragma: no cover
    """The Tensorflow TFRecord Dataset.

    Root is a full path to tfrecord file, which contains the file name.

    Args: root (str): filename of dataset.
          transform (transform object, default=None):  transform to process input data.
          filter (Filter objects, default=None): filter out examples according
                                                 to specific conditions.
    """

    def __new__(cls, root, transform=None, filter=None):
        """Build a new object of TensorflowTFRecordDataset class."""
        # pylint: disable=no-name-in-module
        from tensorflow.python.data.experimental import parallel_interleave
        from tensorflow.python.platform import gfile

        file_names = gfile.Glob(root)
        ds = tf.data.Dataset.from_tensor_slices(file_names)
        ds = ds.apply(parallel_interleave(tf.data.TFRecordDataset, cycle_length=len(file_names)))
        if transform is not None:
            ds = ds.map(transform, num_parallel_calls=None)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # this number can be tuned
        return ds


@dataset_registry(dataset_type="ImageRecord", framework="tensorflow, tensorflow_itex", dataset_format="")
class TensorflowImageRecord(IterableDataset):  # pragma: no cover
    """Tensorflow imageNet database in tf record format.

    Please arrange data in this way:
        root/validation-000-of-100
        root/validation-001-of-100
        ...
        root/validation-099-of-100
    The file name needs to follow this pattern: '* - * -of- *'

    Args: root (str): Root directory of dataset.
          transform (transform object, default=None):  transform to process input data.
          filter (Filter objects, default=None): filter out examples according
                                                 to specific conditions.
    """

    """Configuration for Imagenet dataset."""

    def __new__(cls, root, transform=None, filter=None):
        """Build a new object of TensorflowImageRecord class."""
        from tensorflow.python.platform import gfile  # pylint: disable=no-name-in-module

        glob_pattern = os.path.join(root, "*-*-of-*")
        file_names = gfile.Glob(glob_pattern)
        if not file_names:
            raise ValueError("Found no files in --root matching: {}".format(glob_pattern))

        # pylint: disable=no-name-in-module
        from tensorflow.python.data.experimental import parallel_interleave

        from neural_compressor.data.transforms.imagenet_transform import ParseDecodeImagenet

        ds = tf.data.TFRecordDataset.list_files(file_names, shuffle=False)
        ds = ds.apply(parallel_interleave(tf.data.TFRecordDataset, cycle_length=len(file_names)))

        if transform is not None:
            transform.transform_list.insert(0, ParseDecodeImagenet())
        else:
            transform = ParseDecodeImagenet()
        ds = ds.map(transform, num_parallel_calls=None)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # this number can be tuned
        return ds


@dataset_registry(dataset_type="VOCRecord", framework="tensorflow, tensorflow_itex", dataset_format="")
class TensorflowVOCRecord(IterableDataset):  # pragma: no cover
    """The Tensorflow PASCAL VOC 2012 database in tf record format.

    Please arrange data in this way:
        root/val-00000-of-00004.tfrecord
        root/val-00001-of-00004.tfrecord
        ...
        root/val-00003-of-00004.tfrecord
    The file name needs to follow this pattern: 'val-*-of-*'

    Args: root (str): Root directory of dataset.
          transform (transform object, default=None):  transform to process input data.
          filter (Filter objects, default=None): filter out examples according
                                                 to specific conditions.
    """

    def __new__(cls, root, transform=None, filter=None):
        """Build a new object of TensorflowVOCRecord class."""
        from tensorflow.python.platform import gfile  # pylint: disable=no-name-in-module

        glob_pattern = os.path.join(root, "%s-*" % "val")
        file_names = gfile.Glob(glob_pattern)
        if not file_names:
            raise ValueError("Found no files in --root matching: {}".format(glob_pattern))

        # pylint: disable=no-name-in-module
        from tensorflow.python.data.experimental import parallel_interleave

        ds = tf.data.TFRecordDataset.list_files(file_names, shuffle=False)
        ds = ds.apply(parallel_interleave(tf.data.TFRecordDataset, cycle_length=len(file_names)))

        if transform is not None:
            ds = ds.map(transform, num_parallel_calls=None)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # this number can be tuned
        return ds
