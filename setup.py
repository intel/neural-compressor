from io import open
from setuptools import find_packages, setup, Extension
import setuptools.command.build_ext
import os
import re
import sys
import shutil

cwd = os.path.dirname(os.path.abspath(__file__))

try:
    filepath = './neural_compressor/version.py'
    with open( filepath ) as version_file:
        __version__ ,= re.findall( '__version__ = "(.*)"', version_file.read() )
except Exception as error:
    assert False,  "Error: Could not open '%s' due %s\n" % (filepath, error)

class build_ext(setuptools.command.build_ext.build_ext):
    def run(self):
        if not sys.platform.startswith("win"):
            for ext in self.extensions:
                self.build_cmake(ext)
            super().run()
        else:
            print("Engine is not support windows for now")

    def build_cmake(self, ext):
        import pathlib
        cwd = pathlib.Path().absolute()

        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        executable_path = extdir.parent.absolute()
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(extdir.parent.absolute()),
            '-DPYTHON_EXECUTABLE={}'.format(sys.executable)
        ]

        build_args = [
            '-j'
        ]

        os.chdir(str(build_temp))
        self.spawn(['cmake3', ext.sourcedir] + cmake_args)
        self.spawn(['make'] + build_args)
        if os.path.exists('inferencer'):
            shutil.copy('inferencer', executable_path)
        os.chdir(str(cwd))

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

def check_submodules():
    def check_for_files(folder, files):
        if not any(os.path.exists(os.path.join(folder, f)) for f in files):
            report("Could not find any of {} in {}".format(", ".join(files), folder))
            report("Did you run 'git submodule update --init --recursive'?")
            sys.exit(1)

    def not_exists_or_empty(folder):
        return not os.path.exists(folder) or (os.path.isdir(folder) and len(os.listdir(folder)) == 0)

    git_modules_path = os.path.join(cwd, ".gitmodules")
    with open(git_modules_path) as f:
        folders = [os.path.join(cwd, line.split("=", 1)[1].strip()) for line in
                   f.readlines() if line.strip().startswith("path")]

    # If none of the submodule folders exists, try to initialize them
    if all(not_exists_or_empty(folder) for folder in folders) and not sys.platform.startswith("win"):
        try:
            print(' --- Trying to initialize submodules')
            start = time.time()
            subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"], cwd=cwd)
            end = time.time()
            print(' --- Submodule initialization took {:.2f} sec'.format(end - start))
        except Exception:
            print(' --- Submodule initalization failed')
            print('Please run:\n\tgit submodule update --init --recursive')
            sys.exit(1)

if __name__ == '__main__':
    check_submodules()

    setup(
        name="neural_compressor",
        version=__version__,
        author="Intel MLP/MLPC Team",
        author_email="feng.tian@intel.com, chuanqi.wang@intel.com, pengxin.yuan@intel.com, guoming.zhang@intel.com, haihao.shen@intel.com, jiong.gong@intel.com, xi2.chen@intel.com",
        description="Repository of Intel® Neural Compressor",
        long_description=open("README.md", "r", encoding='utf-8').read(),
        long_description_content_type="text/markdown",
        keywords='quantization, auto-tuning, post-training static quantization, post-training dynamic quantization, quantization-aware training, tuning strategy',
        license='',
        url="https://github.com/intel/neural-compressor",
        ext_modules=[CMakeExtension("engine_py", str(cwd) + '/engine/executor/')],
        packages = find_packages(),
        include_package_data = True,
        package_dir = {'':'.'},
        package_data={
            '': ['*.py', '*.yaml'],
            'neural_compressor.ux': [
                "web/static/*.*",
                "web/static/assets/*.*",
                "web/static/assets/fonts/*.*",
                "utils/configs/*.json",
                "utils/configs/predefined_configs/**/*.yaml",
                "utils/templates/*.txt",
            ],
            'engine': ['*.py'],
        },
        cmdclass={
            'build_ext': build_ext,
        },
        install_requires=[
            'numpy', 'pyyaml', 'scikit-learn', 'schema', 'py-cpuinfo', 'hyperopt', 'pandas', 'pycocotools', 'opencv-python',
            'requests', 'Flask-Cors', 'Flask-SocketIO', 'Flask', 'gevent-websocket', 'gevent', 'psutil', 'Pillow', 'sigopt',
            'prettytable'],
        scripts=['neural_compressor/ux/bin/neural_compressor_bench', 'engine/bin/inferencer'],
        python_requires='>=3.6.0',
        classifiers=[
              'Intended Audience :: Science/Research',
              'Programming Language :: Python :: 3',
              'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
    )
