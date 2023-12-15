# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config


# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import subprocess
import sys
from datetime import datetime
import covasim as cv

# Set environment
os.environ['SPHINX_BUILD'] = 'True' # This is used so cv.options.set('jupyter') doesn't reset the Matplotlib renderer
on_rtd = os.environ.get('READTHEDOCS') == 'True'

# Rename "covasim package" to "API reference"
filename = 'modules.rst' # This must match the Makefile
with open(filename) as f: # Read existing file
    lines = f.readlines()
lines[0] = "API reference\n" # Blast away the existing heading and replace with this
lines[1] = "=============\n" # Ensure the heading is the right length
with open(filename, "w") as f: # Write new file
    f.writelines(lines)


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',  # Add a link to the Python source code for classes, functions etc.
    'nbsphinx',
    'sphinx_search.extension', # search across multiple docsets in domain
    'sphinx.ext.viewcode', # link to view source code
]

autodoc_default_options = {
    'member-order': 'bysource',
    'members': None
}

autodoc_mock_imports = []


napoleon_google_docstring = True

# Configure autosummary
autosummary_generate = True  # Turn on sphinx.ext.autosummary
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
html_show_sourcelink = False  # Remove 'view source code' from top of page (for html, not python)
autodoc_member_order = 'bysource' # Keep original ordering
add_module_names = False  # NB, does not work
autodoc_inherit_docstrings = False # Stops sublcasses from including docs from parent classes

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = '.rst'
master_doc = 'index'

# General information about the project.
project = 'Covasim'
copyright = f'2020 - {datetime.today().year}, Bill & Melinda Gates Foundation. All rights reserved.\nThese docs were built for Covasim version {cv.__version__}.\n'
author = 'Institute for Disease Modeling'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = cv.__version__
# The full version, including alpha/beta/rc tags.
release = cv.__version__

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# suppress warnings for multiple possible Python references in the namespace
# suppress_warnings = ['ref.python']
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# RST epilog is added to the end of every topic. Useful for replace
# directives to use across the docset.
rst_epilog = "\n.. include:: /variables.txt"

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"


html_logo = "images/IDM_white.png"
html_favicon = "images/favicon.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_context = {
    'rtd_url': 'https://docs.idmod.org/projects/covasim/en/latest',
    'theme_vcs_pageview_mode': 'edit'
}
# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
#
if not on_rtd:
    html_extra_path = ['robots.txt']


# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_last_updated_fmt = '%Y-%b-%d'
html_show_sourcelink = True
html_show_sphinx = False
html_copy_source = False
htmlhelp_basename = 'Covasim'

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
html_use_opensearch = 'docs.idmod.org/projects/covasim/en/latest'


# -- RTD Sphinx search for searching across the entire domain, default child -------------

if os.environ.get('READTHEDOCS') == 'True':

    search_project_parent = "institute-for-disease-modeling-idm"
    search_project = os.environ["READTHEDOCS_PROJECT"]
    search_version = os.environ["READTHEDOCS_VERSION"]

    rtd_sphinx_search_default_filter = f"subprojects:{search_project}/{search_version}"

    rtd_sphinx_search_filters = {
        "Search this project": f"project:{search_project}/{search_version}",
        "Search all IDM docs": f"subprojects:{search_project_parent}/{search_version}",
    }

def setup(app):
    app.add_css_file("theme_overrides.css")


# Modify this to not rerun the Jupyter notebook cells -- usually set by build_docs
nb_ex_default = ['auto', 'never'][0]
nb_ex = os.getenv('NBSPHINX_EXECUTE')
if not nb_ex: nb_ex = nb_ex_default
print(f'\n\nBuilding Jupyter notebooks with build option: {nb_ex}\n\n')
nbsphinx_execute = nb_ex
