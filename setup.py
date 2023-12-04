import re
import sys
from io import open

from setuptools import find_packages, setup


def fetch_requirements(path):
    with open(path, "r") as fd:
        return [r.strip() for r in fd.readlines()]


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

    project_name = PKG_INSTALL_CFG[cfg_key].get("project_name")
    include_packages = PKG_INSTALL_CFG[cfg_key].get("include_packages") or {}
    package_data = PKG_INSTALL_CFG[cfg_key].get("package_data") or {}
    install_requires = PKG_INSTALL_CFG[cfg_key].get("install_requires") or []
    entry_points = PKG_INSTALL_CFG[cfg_key].get("entry_points") or {}
    extras_require = PKG_INSTALL_CFG[cfg_key].get("extras_require") or {}

    setup(
        name=project_name,
        version=__version__,
        author="Intel AIA Team",
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
