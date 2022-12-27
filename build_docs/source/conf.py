# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../neural_compressor/'))
import version as ver

version= ver.__version__
release = version

with open("version.txt", "w") as f:
    f.write(version)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Intel® Neural Compressor'
copyright = '2022, Intel® Neural Compressor, Intel'
author = 'Intel® Neural Compressor developers'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
        'recommonmark',
        'sphinx_markdown_tables',
        'sphinx.ext.coverage',
        'sphinx.ext.autosummary',
        'sphinx_md',
        'autoapi.extension',
        'sphinx.ext.napoleon',
        'sphinx.ext.githubpages'
        ]

autoapi_dirs = ['../../neural_compressor']
autoapi_root = "autoapi"
autoapi_keep_files = True
autoapi_add_toctree_entry = False
autosummary_generate = True
autoapi_options = ['members',  'show-inheritance',
                   'show-module-summary', 'imported-members', ]
autoapi_ignore = []

templates_path = ['_templates']

source_suffix = ['.rst', '.md']

# The master toctree document.
master_doc = 'index'

exclude_patterns = []

pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

#html_static_path = ['_static']

