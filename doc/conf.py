# -*- coding: utf-8 -*-

from nbsite.shared_conf import *

# Declare information specific to this project.
project = u'HoloViews'
authors = u'PyViz developers'
copyright = u'2019 ' + authors
description = 'Stop plotting your data - annotate your data and let it visualize itself.'

import holoviews
version = release = holoviews.__version__

html_theme = 'sphinx_ioam_theme'
html_static_path += ['_static']
html_theme_options = {
    'logo': 'logo.png',
    'favicon': 'favicon.ico',
    'custom_css': 'holoviews.css'
}
nbbuild_cell_timeout = 360

extensions += ['nbsite.gallery']

templates_path = ['_templates']

nbsite_gallery_conf = {
    'backends': ['bokeh', 'matplotlib', 'plotly'],
    'galleries': {},
    'github_org': 'pyviz',
    'github_project': 'holoviews'
}

if os.environ.get('HV_DOC_GALLERY') not in ('False', 'false', '0'):
    nbsite_gallery_conf['galleries']['gallery'] = {
        'title': 'Gallery',
        'sections': [
            {'path': 'apps', 'title': 'Applications', 'skip': True},
            'demos'
        ]
    }

if os.environ.get('HV_DOC_REF_GALLERY') not in ('False', 'false', '0'):
    nbsite_gallery_conf['galleries']['reference'] = {
        'title': 'Reference Gallery',
        'path': 'reference',
        'sections': [
            'elements',
            'containers',
            'streams',
            'apps'
        ]
    }

MAIN_SITE = '//holoviews.org'

html_context.update({
    'PROJECT': project,
    'DESCRIPTION': description,
    'AUTHOR': authors,
    'VERSION': version,
    'WEBSITE_SERVER': 'https:',
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
        ('About', '/about.html')
    ),
    # Social links
    'SOCIAL': (
        ('Gitter', '//gitter.im/pyviz/pyviz'),
        ('Twitter', '//twitter.com/holoviews'),
        ('Github', '//github.com/pyviz/holoviews'),
    ),
    # Links for the docs sub navigation
    'NAV': (
        ('Getting started', 'getting_started/index'),
        ('User Guide', 'user_guide/index'),
        ('Gallery', 'gallery/index'),
        ('Reference Gallery', 'reference/index'),
        ('Releases', 'releases'),
        ('API', 'Reference_Manual/index'),
        ('FAQ', 'FAQ')
    ),
    'js_includes': html_context['js_includes']+['holoviews.js']
})
