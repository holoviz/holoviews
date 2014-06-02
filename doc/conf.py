# -*- coding: utf-8 -*-

from builder.shared_conf import * # pyflakes:ignore (API import)

paths = ['../param/', '.', '..']
add_paths(paths)

# General information about the project.
project = u'DataViews'
copyright = u'2014, IOAM: Jean-Luc Stevens and Philipp Rudiger'
ioam_module = 'dataviews'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = '2014.05.14'
# The full version, including alpha/beta/rc tags.
release = '2014.05.14'

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
htmlhelp_basename = 'DataViewsdoc'


# -- Options for LaTeX output ---------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
  ('index', 'DataViews.tex', u'DataViews Documentation',
   u'IOAM: Jean-Luc Stevens and Philipp Rudiger', 'manual'),
]

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', ioam_module, u'DataViews Documentation',
     [u'IOAM: Jean-Luc Stevens and Philipp Rudiger'], 1)
]

# If true, show URL addresses after external links.
#man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
  ('index', project, u'DataViews Documentation',
   u'IOAM: Jean-Luc Stevens and Philipp Rudiger', project, 'One line description of project.',
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
        import runipy
        nbbuild.setup(app)
    except:
        print('RunIPy could not be imported, pages including the '
              'Notebook directive will not build correctly')