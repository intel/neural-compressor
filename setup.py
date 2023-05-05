from io import open
from setuptools import find_packages, setup
import os
import re
import sys

cwd = os.path.dirname(os.path.abspath(__file__))

try:
    filepath = './neural_compressor/version.py'
    with open(filepath) as version_file:
        __version__, = re.findall('__version__ = "(.*)"', version_file.read())
except Exception as error:
    assert False,  "Error: Could not open '%s' due %s\n" % (filepath, error)

neural_insights_installation = False
if "--neural-insights" in sys.argv:
    neural_insights_installation = True
    sys.argv.remove("--neural-insights")

# define package data
package_data = {'': ['*.py', '*.yaml']}


# define install requirements
install_requires_list = [
        'numpy', 'pyyaml', 'scikit-learn', 'schema', 'py-cpuinfo', 'pandas', 'pycocotools',
        'opencv-python', 'requests', 'psutil', 'Pillow', 'prettytable', 'deprecated']

# define scripts
scripts_list = []
project_name = "neural_compressor"
packages_exclude = find_packages(exclude=["test.*", "test", "neural_insights", "neural_insights/*"])
author_email = "feng.tian@intel.com, haihao.shen@intel.com, suyue.chen@intel.com"
description = "Repository of Intel® Neural Compressor"

if neural_insights_installation:
    try:
        filepath = './neural_insights/version.py'
        with open(filepath) as version_file:
            __version__, = re.findall('__version__ = "(.*)"', version_file.read())
    except Exception as error:
        assert False, "Error: Could not open '%s' due %s\n" % (filepath, error)
    project_name = "neural_insights"
    packages_exclude = find_packages(exclude=["test.*", "test"])
    package_data = {
        'neural_insights': [
            "web/app/*.*",
            "web/app/static/*.*",
            "web/app/static/css/*.*",
            "web/app/static/js/*.*",
            "web/app/static/media/*.*",
        ]
    }
    install_requires_list = [
        'Flask-Cors', 'Flask-SocketIO', 'Flask', 'gevent-websocket', 'gevent', 'cryptography',
    ]
    scripts_list = ['neural_insights/bin/neural_insights']
    author_email = "agata.radys@intel.com, bartosz.myrcha@intel.com"
    description = "Repository of Intel® Neural Insights"

if __name__ == '__main__':
    setup(
        name=project_name,
        version=__version__,
        author="Intel AIA Team",
        author_email=author_email,
        description=description,
        long_description=open("README.md", "r", encoding='utf-8').read(),
        long_description_content_type="text/markdown",
        keywords='quantization, auto-tuning, post-training static quantization, post-training dynamic quantization, quantization-aware training, tuning strategy',
        license='Apache 2.0',
        url="https://github.com/intel/neural-compressor",
        packages=packages_exclude,
        include_package_data=True,
        package_data=package_data,
        install_requires=install_requires_list,
        scripts=scripts_list,
        python_requires='>=3.6.0',
        classifiers=[
              'Intended Audience :: Science/Research',
              'Programming Language :: Python :: 3',
              'Topic :: Scientific/Engineering :: Artificial Intelligence',
              'License :: OSI Approved :: Apache Software License',
        ],
    )