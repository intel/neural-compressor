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
    # pip install neural-compressor, install the whole package with all APIs, including PyTorch, TensorFlow and JAX APIs.
    "neural_compressor": {
        "project_name": "neural_compressor",
        "include_packages": find_packages(
            include=["neural_compressor", "neural_compressor.*"],
        ),
        "package_data": {"": ["*.json", "*.yaml"]},
        "extras_require": {
            "pt": fetch_requirements("requirements_pt.txt"),
            "tf": fetch_requirements("requirements_tf.txt"),
            "jax": fetch_requirements("requirements_jax.txt"),
        },
    },

    # pip install neural-compressor-pt, install PyTorch API.
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
    # pip install neural-compressor-tf, install TensorFlow API.
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
    # 3.x JAX binary build config, pip install neural-compressor-jax, install 3.x JAX API.
    "neural_compressor_jax": {
        "project_name": "neural_compressor_jax",
        "include_packages": find_packages(
            include=[
                "neural_compressor.common",
                "neural_compressor.common.*",
                "neural_compressor.jax",
                "neural_compressor.jax.*",
            ],
        ),
        "package_data": {"": ["*.yaml"]},
        "install_requires": fetch_requirements("requirements_jax.txt"),
    },
}


if __name__ == "__main__":
    # for setuptools>=80.0.0, `INC_PT_ONLY=1 pip install -e .`
    only_set = []
    cfg_key = "neural_compressor"
    if os.environ.get("INC_PT_ONLY", False):
        cfg_key = "neural_compressor_pt"
        only_set.append("INC_PT_ONLY")
    if os.environ.get("INC_TF_ONLY", False):
        cfg_key = "neural_compressor_tf"
        only_set.append("INC_TF_ONLY")
    if os.environ.get("INC_JAX_ONLY", False):
        cfg_key = "neural_compressor_jax"
        only_set.append("INC_JAX_ONLY")
    if len(only_set) > 1:
        raise ValueError(f"Environment variables {' and '.join(only_set)} are set. Please set only one.")

    # for setuptools < 80.0.0, `python setup.py develop pt`
    if "pt" in sys.argv:
        sys.argv.remove("pt")
        cfg_key = "neural_compressor_pt"
    if "tf" in sys.argv:
        sys.argv.remove("tf")
        cfg_key = "neural_compressor_tf"
    if "jax" in sys.argv:
        sys.argv.remove("jax")
        cfg_key = "neural_compressor_jax"
    ext_modules = []
    cmdclass = {}
    project_name = PKG_INSTALL_CFG[cfg_key].get("project_name")
    include_packages = PKG_INSTALL_CFG[cfg_key].get("include_packages") or {}
    package_data = PKG_INSTALL_CFG[cfg_key].get("package_data") or {}
    install_requires = PKG_INSTALL_CFG[cfg_key].get("install_requires") or []
    extras_require = PKG_INSTALL_CFG[cfg_key].get("extras_require") or {}

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
        extras_require=extras_require,
        python_requires=">=3.7.0",
        classifiers=[
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "License :: OSI Approved :: Apache Software License",
        ],
    )
