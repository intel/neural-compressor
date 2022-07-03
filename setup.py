from io import open
from setuptools import find_packages, setup
import os
import re
import sys

cwd = os.path.dirname(os.path.abspath(__file__))

try:
    filepath = './neural_compressor/version.py'
    with open( filepath ) as version_file:
        __version__ ,= re.findall( '__version__ = "(.*)"', version_file.read() )
except Exception as error:
    assert False,  "Error: Could not open '%s' due %s\n" % (filepath, error)


if __name__ == '__main__':

    setup(
        name="neural_compressor",
        version=__version__,
        author="Intel AIA/AIPC Team",
        author_email="feng.tian@intel.com, haihao.shen@intel.com, penghui.cheng@intel.com, xi2.chen@intel.com, jiong.gong@intel.com",
        description="Repository of Intel® Neural Compressor",
        long_description=open("README.md", "r", encoding='utf-8').read(),
        long_description_content_type="text/markdown",
        keywords='quantization, auto-tuning, post-training static quantization, post-training dynamic quantization, quantization-aware training, tuning strategy',
        license='Apache 2.0',
        url="https://github.com/intel/neural-compressor",
        packages = find_packages(exclude=["test.*", "test"]),
        include_package_data = True,
        package_data={
            '': ['*.py', '*.yaml'],
            'neural_compressor.ux': [
                "web/static/*.*",
                "web/static/assets/*.*",
                "web/static/assets/fonts/*.*",
                "components/db_manager/alembic.ini",
                "components/db_manager/alembic/*",
                "components/db_manager/alembic/versions/*.py",
                "utils/configs/*.json",
                "utils/configs/predefined_configs/**/*.yaml",
                "utils/templates/*.txt",
            ],
        },
        install_requires=[
            'numpy', 'pyyaml', 'scikit-learn', 'schema', 'py-cpuinfo', 'hyperopt', 'pandas', 'pycocotools', 'opencv-python',
            'requests', 'Flask-Cors', 'Flask-SocketIO', 'Flask', 'gevent-websocket', 'gevent', 'psutil', 'Pillow', 'sigopt',
            'prettytable', 'cryptography', 'Cython', 'sqlalchemy==1.4.27', 'alembic==1.7.7'],
        scripts=['neural_compressor/ux/bin/inc_bench'],
        python_requires='>=3.7.0',
        classifiers=[
              'Intended Audience :: Science/Research',
              'Programming Language :: Python :: 3',
              'Topic :: Scientific/Engineering :: Artificial Intelligence',
              'License :: OSI Approved :: Apache Software License',
        ],
    )
