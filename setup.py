from io import open
from setuptools import find_packages, setup
import re
import sys

try:
    filepath = './lpot/version.py'
    with open( filepath ) as version_file:
        __version__ ,= re.findall( '__version__ = "(.*)"', version_file.read() )
except Exception as error:
    assert False,  "Error: Could not open '%s' due %s\n" % (filepath, error)

setup(
    name="lpot",
    version=__version__,
    author="Intel MLP/MLPC Team",
    author_email="feng.tian@intel.com, chuanqi.wang@intel.com, pengxin.yuan@intel.com, guoming.zhang@intel.com, haihao.shen@intel.com, jiong.gong@intel.com, xi2.chen@intel.com",
    description="Repository of Intel® Low Precision Optimization Tool",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    keywords='quantization, auto-tuning, post-training static quantization, post-training dynamic quantization, quantization-aware training, tuning strategy',
    license='',
    url="https://github.com/intel/lpot",
    packages = find_packages(),
    package_dir = {'':'.'},
    package_data={
        '': ['*.py', '*.yaml'],
        "lpot.ux": [
            "web/static/*.*",
            "web/static/assets/*.*",
            "web/static/assets/fonts/*.*",
            "utils/configs/*.json",
            "utils/configs/predefined_configs/**/*.yaml",
            "utils/templates/*.txt",
        ],
    },
    install_requires=[
        'numpy', 'pyyaml', 'scikit-learn', 'schema', 'py-cpuinfo', 'hyperopt', 'pandas', 'pycocotools', 'opencv-python',
        'requests', 'Flask-Cors', 'Flask-SocketIO', 'Flask', 'gevent-websocket', 'gevent', 'psutil', 'Pillow', 'sigopt'],
    entry_points={
      'console_scripts':  [
            "lpot_ux = lpot.ux.lpot_ux:main"
        ]
    },
    python_requires='>=3.6.0',
    classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
