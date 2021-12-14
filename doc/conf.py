# -*- coding: utf-8 -*-

from nbsite.shared_conf import *

# Declare information specific to this project.
project = u'HoloViews'
authors = u'HoloViz developers'
copyright = u'2022 ' + authors
description = 'Stop plotting your data - annotate your data and let it visualize itself.'

import param

param.parameterized.docstring_signature = False
param.parameterized.docstring_describe_params = False

import holoviews
version = release = base_version(holoviews.__version__)

holoviews.extension.inline = False

html_theme = 'pydata_sphinx_theme'
html_logo = '_static/logo_horizontal_theme.png'
html_favicon = '_static/favicon.ico'

html_static_path += ['_static']

html_css_files = [
    'nbsite.css',
    'css/custom.css'
]

html_theme_options = {
    'github_url': 'https://github.com/holoviz/holoviews',
    'icon_links': [
        {
            'name': 'Twitter',
            'url': 'https://twitter.com/holoviews',
            'icon': 'fab fa-twitter-square',
        },
        {
            'name': 'Discourse',
            'url': 'https://discourse.holoviz.org/',
            'icon': 'fab fa-discourse',
        },
    ],
    "footer_items": [
        "copyright",
        "last-updated",
    ],
    'google_analytics_id': 'UA-61554933-1',
}

nbbuild_cell_timeout = 360

extensions += [
    'nbsite.gallery',
    'sphinx_copybutton',
]

nbsite_gallery_conf = {
    'backends': ['bokeh', 'matplotlib', 'plotly'],
    'galleries': {},
    'github_org': 'holoviz',
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

templates_path = [
    '_templates'
]

html_context.update({
    # Used to add binder links to the latest released tag.
    'last_release': release,
    'github_user': 'holoviz',
    'github_repo': 'holoviews',
})

# Override the Sphinx default title that appends `documentation`
html_title = f'{project} v{version}'
# Format of the last updated section in the footer
html_last_updated_fmt = '%Y-%m-%d'
