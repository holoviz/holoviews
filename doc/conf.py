# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.getcwd())

from builder.shared_conf import * # pyflakes:ignore (API import)

paths = ['../param/', '.', '..']
add_paths(paths)

from ..setup import setup_args

# Declare information specific to this project.
project = u'HoloViews'
authors = u'IOAM: Jean-Luc R. Stevens, Philipp Rudiger, and James A. Bednar'
copyright = u'2015 ' + authors
ioam_module = 'holoviews'
description = 'Composable, declarative data structures for building even complex visualizations easily'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = setup_args['version']
# The full version, including alpha/beta/rc tags.
release = setup_args['version']


rst_epilog = """
.. _zip archive: notebooks-{version}.zip
.. _tutorial notebooks: Tutorials/notebooks-{version}.zip
""".format(version=version)

# Override IOAM theme
import sphinx_rtd_theme
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_logo = '_static/holoviews_logo.png'
html_favicon = '_static/favicon.ico'

# -------------------------------------------------------------------------
# -- The remaining items are less likely to need changing for a new project

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'test_data', 'reference_data', 'nbpublisher',
                    'builder']

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = project

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static', 'builder/_shared_static']

# Output file base name for HTML help builder.
htmlhelp_basename = ioam_module+'doc'


# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
  ('index', ioam_module+'.tex', project+u' Documentation', authors, 'manual'),
]

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', ioam_module, project+u' Documentation', [authors], 1)
]
# If true, show URL addresses after external links.
#man_show_urls = False


# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
  ('index', project, project+u' Documentation', authors, project, description,
   'Miscellaneous'),
]


# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'http://docs.python.org/': None,
                       'http://ipython.org/ipython-doc/2/': None,
                       'http://ioam.github.io/param/': None}

from builder.paramdoc import param_formatter
from nbpublisher import nbbuild


def setup(app):
    app.connect('autodoc-process-docstring', param_formatter)
    try:
        import runipy # pyflakes:ignore (Warning import)
        nbbuild.setup(app)
    except:
        print('RunIPy could not be imported; pages including the '
              'Notebook directive will not build correctly')
