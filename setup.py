import os
import re
import subprocess
import sys
from io import open

from setuptools import find_packages, setup


def fetch_requirements(path):
    with open(path, "r") as fd:
        return [r.strip() for r in fd.readlines()]


def is_commit_on_tag():
    try:
        result = subprocess.run(
            ["git", "describe", "--exact-match", "--tags"], capture_output=True, text=True, check=True
        )
        tag_name = result.stdout.strip()
        return tag_name
    except subprocess.CalledProcessError:
        return False


def get_build_version():
    if is_commit_on_tag():
        return __version__
    try:
        result = subprocess.run(["git", "describe", "--tags"], capture_output=True, text=True, check=True)
        distance = result.stdout.strip().split("-")[-2]
        commit = result.stdout.strip().split("-")[-1]
        return f"{__version__}.dev{distance}+{commit}"
    except subprocess.CalledProcessError:
        return __version__


try:
    filepath = "./neural_compressor/version.py"
    with open(filepath) as version_file:
        (__version__,) = re.findall('__version__ = "(.*)"', version_file.read())
except Exception as error:
    assert False, "Error: Could not open '%s' due %s\n" % (filepath, error)

PKG_INSTALL_CFG = {
    # overall installation config, pip install neural-compressor
    "neural_compressor": {
        "project_name": "neural_compressor",
        "include_packages": find_packages(
            include=["neural_compressor", "neural_compressor.*"],
            exclude=[
                "neural_compressor.template",
            ],
        ),
        "package_data": {"": ["*.yaml"]},
        "install_requires": fetch_requirements("requirements.txt"),
        "extras_require": {
            "pt": fetch_requirements("requirements_pt.txt"),
            "tf": fetch_requirements("requirements_tf.txt"),
        },
    },
    # 3.x pt binary build config, pip install neural-compressor-pt, install 3.x PyTorch API.
    "neural_compressor_pt": {
        "project_name": "neural_compressor_pt",
        "include_packages": find_packages(
            include=[
                "neural_compressor.common",
                "neural_compressor.common.*",
                "neural_compressor.torch",
                "neural_compressor.torch.*",
                "neural_compressor.transformers",
                "neural_compressor.transformers.*",
                "neural_compressor.evaluation",
                "neural_compressor.evaluation.*",
            ],
        ),
        "package_data": {"": ["*.json"]},
        "install_requires": fetch_requirements("requirements_pt.txt"),
    },
    # 3.x tf binary build config, pip install neural-compressor-tf, install 3.x TensorFlow API.
    "neural_compressor_tf": {
        "project_name": "neural_compressor_tf",
        "include_packages": find_packages(
            include=[
                "neural_compressor.common",
                "neural_compressor.common.*",
                "neural_compressor.tensorflow",
                "neural_compressor.tensorflow.*",
            ],
        ),
        "package_data": {"": ["*.yaml"]},
        "install_requires": fetch_requirements("requirements_tf.txt"),
    },
}


if __name__ == "__main__":
    cfg_key = "neural_compressor"

    # Temporary implementation of fp8 tensor saving and loading
    # Will remove after Habana torch applies below patch:
    # https://github.com/pytorch/pytorch/pull/114662
    ext_modules = []
    cmdclass = {}

    if "pt" in sys.argv:
        sys.argv.remove("pt")
        cfg_key = "neural_compressor_pt"

    if "tf" in sys.argv:
        sys.argv.remove("tf")
        cfg_key = "neural_compressor_tf"

    project_name = PKG_INSTALL_CFG[cfg_key].get("project_name")
    include_packages = PKG_INSTALL_CFG[cfg_key].get("include_packages") or {}
    package_data = PKG_INSTALL_CFG[cfg_key].get("package_data") or {}
    install_requires = PKG_INSTALL_CFG[cfg_key].get("install_requires") or []
    extras_require = PKG_INSTALL_CFG[cfg_key].get("extras_require") or {}
    entry_points = {
        "console_scripts": [
            "incbench = neural_compressor.common.benchmark:benchmark",
        ]
    }

    setup(
        name=project_name,
        author="Intel AIPT Team",
        version=get_build_version(),
        author_email="feng.tian@intel.com, haihao.shen@intel.com, suyue.chen@intel.com",
        description="Repository of Intel® Neural Compressor",
        long_description=open("README.md", "r", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        keywords="quantization,auto-tuning,post-training static quantization,"
        "post-training dynamic quantization,quantization-aware training",
        license="Apache 2.0",
        url="https://github.com/intel/neural-compressor",
        packages=include_packages,
        include_package_data=True,
        package_data=package_data,
        install_requires=install_requires,
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        entry_points=entry_points,
        extras_require=extras_require,
        python_requires=">=3.7.0",
        classifiers=[
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "License :: OSI Approved :: Apache Software License",
        ],
    )
