from io import open
from setuptools import find_packages, setup

setup(
    name="ilit",
    version="1.0a0",
    author="Intel MLP/MLPC Team",
    author_email="feng.tian@intel.com, chuanqi.wang@intel.com, pengxin.yuan@intel.com, guoming.zhang@intel.com, haihao.shen@intel.com, jiong.gong@intel.com",
    description="Repository of Intel Low Precision Optimization Tool",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    keywords='quantization, auto-tuning, post-training static quantization, post-training dynamic quantization, quantization-aware training, tuning strategy',
    license='',
    url="https://github.com/intel/lp-opt-tool",
    packages = find_packages(),
    package_dir = {'':'.'},
    package_data={'': ['*.py', '*.yaml']},
    install_requires=['numpy', 'pyyaml', 'scikit-learn'],
    entry_points={
      'console_scripts':  [""]
    },
    python_requires='>=3.5.0',
    classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
