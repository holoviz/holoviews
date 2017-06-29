# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.getcwd())

from builder.shared_conf import * # noqa (API import)

paths = ['../param/', '.', '..']
add_paths(paths)

from ..setup import setup_args

# Declare information specific to this project.
project = u'HoloViews'
authors = u'IOAM: Jean-Luc R. Stevens, Philipp Rudiger, and James A. Bednar'
copyright = u'2017 ' + authors
ioam_module = 'holoviews'
description = 'Stop plotting your data - annotate your data and let it visualize itself.'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = setup_args['version']
# The full version, including alpha/beta/rc tags.
release = setup_args['version']

ASSETS_URL = 'http://assets.holoviews.org'

rst_epilog = """
.. _tutorial notebooks: {url}/notebooks-{version}.zip
""".format(url=ASSETS_URL, version=version)

# Override IOAM theme
html_theme = 'holoviews_theme'
html_theme_path = ['.']
MAIN_SITE = '//holoviews.org'

html_context = {
    'SITEMAP_BASE_URL': 'http://holoviews.org/', # Trailing slash is needed
    'SITE_URL': '/',
    'DESCRIPTION': 'HoloViews library, documentation site.',
    'AUTHOR': 'HoloViews contributors',
    'VERSION': version,
    # Nav
    'NAV': (
        ('About', MAIN_SITE + '/about.html'),
        ('Gallery', '/gallery/index.html'),
        ('Docs', '//holoviews.org/'),
        ('Github', '//github.com/ioam/holoviews'),
    ),
    # Links
    'LINKS': (
        ('Getting started', '/getting_started/index.html'),
        ('User Guide', '/user_guide/index.html'),
        ('Gallery', '/gallery/index.html'),
        ('Reference Gallery', '/reference/index.html'),
        ('API Docs', '/Reference_Manual/index.html'),
        ('FAQ', '/FAQ.html'),
        ('About', '/about.html')
    ),
    # About Links
    'ABOUT': (
        ('About', MAIN_SITE + '/about.html')
    ),
    # Social links
    'SOCIAL': (
        ('Gitter', '//gitter.im/ioam/holoviews'),
        ('Twitter', '//twitter.com/holoviews'),
        ('Github', '//github.com/ioam/holoviews'),
    ),
    # Links for the docs sub navigation
    'NAV_DOCS': (
        ('Getting started', 'getting_started/index'),
        ('User Guide', 'user_guide/index'),
        ('Gallery', 'gallery/index'),
        ('Reference Gallery', 'reference/index'),
        ('API', 'Reference_Manual/index'),
        ('FAQ', 'FAQ')
    ),
    'css_server': os.environ.get('HOLOVIEWS_DOCS_CSS_SERVER', 'assets.holoviews.org'),
    'js_includes': ['custom.js', 'require.js']
}

# (Optional) Logo. Should be small enough to fit the navbar (ideally 24x24).
# Path should be relative to the ``_static`` files directory.
html_logo = "my_logo.png"


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
        import runipy # noqa (Warning import)
        nbbuild.setup(app)
    except:
        print('RunIPy could not be imported; pages including the '
              'Notebook directive will not build correctly')
