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
        _, distance, commit = result.stdout.strip().split("-")
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
    # overall install config for build from source, python setup.py install
    "neural_compressor": {
        "project_name": "neural_compressor",
        "include_packages": find_packages(
            include=["neural_compressor", "neural_compressor.*", "neural_coder", "neural_coder.*"],
            exclude=[
                "neural_compressor.template",
            ],
        ),
        "package_data": {"": ["*.yaml"]},
        "install_requires": fetch_requirements("requirements.txt"),
    },
    # 2.x binary build config, pip install neural-compressor
    "neural_compressor_2x": {
        "project_name": "neural_compressor",
        "include_packages": find_packages(
            include=["neural_compressor", "neural_compressor.*", "neural_coder", "neural_coder.*"],
            exclude=[
                "neural_compressor.template",
                "neural_compressor.common",
                "neural_compressor.common.*",
                "neural_compressor.torch",
                "neural_compressor.torch.*",
                "neural_compressor.tensorflow",
                "neural_compressor.tensorflow.*",
                "neural_compressor.onnxrt",
                "neural_compressor.onnxrt.*",
            ],
        ),
        "package_data": {"": ["*.yaml"]},
        "install_requires": fetch_requirements("requirements.txt"),
        "extras_require": {
            "pt": [f"neural_compressor_3x_pt=={__version__}"],
            "tf": [f"neural_compressor_3x_tf=={__version__}"],
            "ort": [f"neural_compressor_3x_ort=={__version__}"],
        },
    },
    # 3.x pt binary build config, pip install neural-compressor[pt], install 2.x API + 3.x PyTorch API.
    "neural_compressor_3x_pt": {
        "project_name": "neural_compressor_3x_pt",
        "include_packages": find_packages(
            include=[
                "neural_compressor.common",
                "neural_compressor.common.*",
                "neural_compressor.torch",
                "neural_compressor.torch.*",
            ],
        ),
        "install_requires": fetch_requirements("requirements_pt.txt"),
    },
    # 3.x tf binary build config, pip install neural-compressor[tf], install 2.x API + 3.x TensorFlow API.
    "neural_compressor_3x_tf": {
        "project_name": "neural_compressor_3x_tf",
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
    # 3.x ort binary build config, pip install neural-compressor[ort], install 2.x API + 3.x ONNXRT API.
    "neural_compressor_3x_ort": {
        "project_name": "neural_compressor_3x_ort",
        "include_packages": find_packages(
            include=[
                "neural_compressor.common",
                "neural_compressor.common.*",
                "neural_compressor.onnxrt",
                "neural_compressor.onnxrt.*",
            ],
        ),
        "install_requires": fetch_requirements("requirements_ort.txt"),
    },
    "neural_insights": {
        "project_name": "neural_insights",
        "include_packages": find_packages(include=["neural_insights", "neural_insights.*"], exclude=["test.*", "test"]),
        "package_data": {
            "neural_insights": [
                "bin/*",
                "*.yaml",
                "web/app/*.*",
                "web/app/static/css/*",
                "web/app/static/js/*",
                "web/app/static/media/*",
                "web/app/icons/*",
            ]
        },
        "install_requires": fetch_requirements("neural_insights/requirements.txt"),
        "entry_points": {"console_scripts": ["neural_insights = neural_insights.bin.neural_insights:execute"]},
    },
    "neural_solution": {
        "project_name": "neural_solution",
        "include_packages": find_packages(include=["neural_solution", "neural_solution.*"]),
        "package_data": {
            "neural_solution": [
                "scripts/*.*",
                "frontend/*.json",
            ]
        },
        "install_requires": fetch_requirements("neural_solution/requirements.txt"),
        "entry_points": {"console_scripts": ["neural_solution = neural_solution.bin.neural_solution:exec"]},
    },
}


if __name__ == "__main__":
    cfg_key = "neural_compressor"

    # Temporary implementation of fp8 tensor saving and loading
    # Will remove after Habana torch applies below patch:
    # https://github.com/pytorch/pytorch/pull/114662
    ext_modules = []
    cmdclass = {}

    if "neural_insights" in sys.argv:
        sys.argv.remove("neural_insights")
        cfg_key = "neural_insights"

    if "neural_solution" in sys.argv:
        sys.argv.remove("neural_solution")
        cfg_key = "neural_solution"

    if "2x" in sys.argv:
        sys.argv.remove("2x")
        cfg_key = "neural_compressor_2x"

    if "pt" in sys.argv:
        sys.argv.remove("pt")
        cfg_key = "neural_compressor_3x_pt"

    if "tf" in sys.argv:
        sys.argv.remove("tf")
        cfg_key = "neural_compressor_3x_tf"

    if "ort" in sys.argv:
        sys.argv.remove("ort")
        cfg_key = "neural_compressor_3x_ort"

    if bool(os.getenv("USE_FP8_CONVERT", False)):
        from torch.utils.cpp_extension import BuildExtension, CppExtension

        ext_modules = [
            CppExtension(
                "fp8_convert",
                ["neural_compressor/torch/algorithms/habana_fp8/tensor/convert.cpp"],
            ),
        ]
        cmdclass = {"build_ext": BuildExtension}

    project_name = PKG_INSTALL_CFG[cfg_key].get("project_name")
    include_packages = PKG_INSTALL_CFG[cfg_key].get("include_packages") or {}
    package_data = PKG_INSTALL_CFG[cfg_key].get("package_data") or {}
    install_requires = PKG_INSTALL_CFG[cfg_key].get("install_requires") or []
    entry_points = PKG_INSTALL_CFG[cfg_key].get("entry_points") or {}
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
        ext_modules=ext_modules,  # for fp8
        cmdclass=cmdclass,  # for fp8
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
