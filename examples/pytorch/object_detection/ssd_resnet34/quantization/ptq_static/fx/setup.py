# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import distutils.command.build
import os
import subprocess
from collections import namedtuple
from textwrap import dedent

import setuptools.command.build_py
import setuptools.command.develop
import setuptools.command.install
from setuptools import setup, find_packages, Command

TOP_DIR = os.path.realpath(os.path.dirname(__file__))
SRC_DIR = os.path.join(TOP_DIR, 'python')

try:
    git_version = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=TOP_DIR).decode('ascii').strip()
except (OSError, subprocess.CalledProcessError):
    git_version = None

with open(os.path.join(TOP_DIR, 'VERSION_NUMBER')) as version_file:
    VersionInfo = namedtuple('VersionInfo', ['version', 'git_version'])(
        version=version_file.read().strip(),
        git_version=git_version
    )


class create_version(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        with open(os.path.join(SRC_DIR, 'version.py'), 'w') as f:
            f.write(dedent('''
            version = '{version}'
            git_version = '{git_version}'
            '''.format(**dict(VersionInfo._asdict()))))


class build_py(setuptools.command.build_py.build_py):
    def run(self):
        self.run_command('create_version')
        setuptools.command.build_py.build_py.run(self)


class build(distutils.command.build.build):
    def run(self):
        self.run_command('build_py')


class develop(setuptools.command.develop.develop):
    def run(self):
        self.run_command('create_version')
        self.run_command('build')
        setuptools.command.develop.develop.run(self)


cmdclass = {
    'create_version': create_version,
    'build_py': build_py,
    'build': build,
    'develop': develop,
}

setup(
    name="mlperf-inference",
    version=VersionInfo.version,
    description='mlperf inference benchmark',
    setup_requires=['pytest-runner'],
    tests_require=['graphviz', 'parameterized', 'pytest', 'pytest-cov', 'pyyaml'],
    cmdclass=cmdclass,
    packages=find_packages(),
    author='guschmue@microsoft.com',
    author_email='guschmue@microsoft.com',
    url='https://github.com/mlperf/inference',
    install_requires=['numpy>=1.14.1', 'onnx>=1.5', 'pybind11', 'Cython',
                        'pycocotools', 'mlperf_loadgen', 'opencv-python-headless']
)
