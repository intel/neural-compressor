from io import open
from setuptools import find_packages, setup
import re
import sys


def fetch_requirements(path):
    with open(path, 'r') as fd:
        return [r.strip() for r in fd.readlines()]


try:
    filepath = './neural_compressor/version.py'
    with open(filepath) as version_file:
        __version__, = re.findall('__version__ = "(.*)"', version_file.read())
except Exception as error:
    assert False,  "Error: Could not open '%s' due %s\n" % (filepath, error)

neural_insights = False
if "neural_insights" in sys.argv:
    neural_insights = True
    sys.argv.remove("neural_insights")

neural_solution  = False
if "neural_solution" in sys.argv:
    neural_solution = True
    sys.argv.remove("neural_solution")

# define include packages
include_packages = find_packages(include=['neural_compressor', 'neural_compressor.*',
                                 'neural_coder', 'neural_coder.*'])
neural_insights_packages = find_packages(include=['neural_insights', 'neural_insights.*'])
neural_solution_packages = find_packages(include=['neural_solution', 'neural_solution.*'])

# define package data
package_data = {'': ['*.yaml']}
neural_insights_data = {'': ['*.yaml', 'web/app/*.*']}
neural_solution_data = {'': ['Launcher.sh', 'scripts/*.*']}

# define install requirements
install_requires_list = fetch_requirements('requirements.txt')
neural_insights_requires = fetch_requirements('neural_insights/requirements.txt')
neural_solution_requires = fetch_requirements('neural_solution/requirements.txt')

# define scripts
scripts_list = []
neural_insights_scripts_list = ['neural_insights/bin/neural_insights']
neural_solution_scripts_list = ['neural_solution/bin/neural_solution']

if neural_insights:
    project_name = "neural_insights"
    package_data = neural_insights_data
    install_requires_list = neural_insights_requires
    scripts_list = neural_insights_scripts_list
    include_packages = neural_insights_packages
elif neural_solution:
    project_name = "neural_solution"
    package_data = neural_solution_data
    install_requires_list = neural_solution_requires
    scripts_list = neural_solution_scripts_list
    include_packages = neural_solution_packages
else:
    project_name = "neural_compressor"

if __name__ == '__main__':

    setup(
        name=project_name,
        version=__version__,
        author="Intel AIA Team",
        author_email="feng.tian@intel.com, haihao.shen@intel.com, suyue.chen@intel.com",
        description="Repository of Intel® Neural Compressor",
        long_description=open("README.md", "r", encoding='utf-8').read(),
        long_description_content_type="text/markdown",
        keywords='quantization, auto-tuning, post-training static quantization, post-training dynamic quantization, quantization-aware training',
        license='Apache 2.0',
        url="https://github.com/intel/neural-compressor",
        packages=include_packages,
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
