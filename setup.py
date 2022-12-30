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

full_installation = False
if "--full" in sys.argv:
    full_installation = True
    sys.argv.remove("--full")

# define package data
package_data = {'': ['*.py', '*.yaml']}
ux_package_data = {
    'neural_compressor.ux': [
    "web/static/*.*",
    "web/static/assets/*.*",
    "web/static/assets/dark/*.*",
    "web/static/assets/fonts/*.*",
    "components/db_manager/alembic.ini",
    "components/db_manager/alembic/*",
    "components/db_manager/alembic/versions/*.py",
    "utils/configs/*.json",
    "utils/configs/predefined_configs/**/*.yaml",
    "utils/templates/*.txt"]
}

# define install requirements
install_requires_list = [
        'numpy', 'pyyaml', 'scikit-learn', 'schema', 'py-cpuinfo', 'pandas', 'pycocotools',
        'opencv-python', 'requests', 'psutil', 'Pillow', 'prettytable', 'deprecated']
ux_install_requires_list = [
        'Flask-Cors', 'Flask-SocketIO', 'Flask', 'gevent-websocket', 'gevent','sqlalchemy==1.4.27',
        'alembic==1.7.7', 'cryptography']

# define scripts
scripts_list = []
ux_scripts_list = ['neural_compressor/ux/bin/inc_bench']

if full_installation:
    project_name = "neural_compressor_full"
    packages_exclude = find_packages(exclude=["test.*", "test"])
    package_data.update(ux_package_data)
    install_requires_list.extend(ux_install_requires_list)
    scripts_list.extend(ux_scripts_list)
else:
    project_name = "neural_compressor"
    packages_exclude = find_packages(exclude=["test.*", "test", "neural_compressor.ux", "neural_compressor.ux.*"])

if __name__ == '__main__':

    setup(
        name=project_name,
        version=__version__,
        author="Intel AIA Team",
        author_email="feng.tian@intel.com, haihao.shen@intel.com, suyue.chen@intel.com",
        description="Repository of Intel® Neural Compressor",
        long_description=open("README.md", "r", encoding='utf-8').read(),
        long_description_content_type="text/markdown",
        keywords='quantization, auto-tuning, post-training static quantization, post-training dynamic quantization, quantization-aware training, tuning strategy',
        license='Apache 2.0',
        url="https://github.com/intel/neural-compressor",
        packages = packages_exclude,
        include_package_data = True,
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
