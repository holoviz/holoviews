# -*- coding: utf-8 -*-

from nbsite.shared_conf import *

# Declare information specific to this project.
project = u'HoloViews'
authors = u'HoloViz developers'
copyright = u'2020 ' + authors
description = 'Stop plotting your data - annotate your data and let it visualize itself.'

import param

param.parameterized.docstring_signature = False
param.parameterized.docstring_describe_params = False

import holoviews
version = release = holoviews.__version__

holoviews.extension.inline = False

html_theme = 'sphinx_holoviz_theme'
html_static_path += ['_static']
html_theme_options = {
    'logo': 'logo.png',
    'favicon': 'favicon.ico',
    'custom_css': 'holoviews.css',
    'include_logo_text': True,
    'second_nav': True,
    'footer': False
}
nbbuild_cell_timeout = 360

extensions += ['nbsite.gallery']

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
    'WEBSITE_URL': 'https://holoviews.org', # for canonical link
    'GOOGLE_SEARCH_ID': '006807479272082416678:p6n_f0d8taw',
    'GOOGLE_ANALYTICS_UA': 'UA-61554933-1',
    # Links
    'LINKS': (
        ('Getting started', '/getting_started/index'),
        ('User Guide', '/user_guide/index'),
        ('Gallery', '/gallery/index'),
        ('Reference Gallery', '/reference/index'),
        ('API Docs', '/reference_manual/index'),
        ('FAQ', '/FAQ'),
        ('About', '/about')
    ),
    # About Links
    'ABOUT': (
        ('About', '/about.html')
    ),
    # Social links
    'SOCIAL': (
        ('Discourse', '//discourse.holoviz.org/c/holoviews'),
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
        ('API', 'reference_manual/index'),
        ('FAQ', 'FAQ')
    ),
    'js_includes': html_context['js_includes']+['holoviews.js']
})
