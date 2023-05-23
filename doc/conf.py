# -*- coding: utf-8 -*-

import param

param.parameterized.docstring_signature = False
param.parameterized.docstring_describe_params = False

from nbsite.shared_conf import *

# Declare information specific to this project.
project = 'HoloViews'
authors = 'HoloViz developers'
copyright = '2005 ' + authors
description = 'Stop plotting your data - annotate your data and let it visualize itself.'

import holoviews
version = release = base_version(holoviews.__version__)

holoviews.extension.inline = False

html_theme = 'pydata_sphinx_theme'
html_logo = '_static/logo_horizontal_theme.png'
html_favicon = '_static/favicon.ico'

html_static_path += ['_static']

html_css_files += [
    'css/custom.css'
]

html_theme_options = {
    'github_url': 'https://github.com/holoviz/holoviews',
    'icon_links': [
        {
            'name': 'Twitter',
            'url': 'https://twitter.com/holoviews',
            'icon': 'fa-brands fa-twitter-square',
        },
        {
            'name': 'Discourse',
            'url': 'https://discourse.holoviz.org/',
            'icon': 'fa-brands fa-discourse',
        },
        {
            "name": "Discord",
            "url": "https://discord.gg/AXRHnJU6sP",
            "icon": "fa-brands fa-discord",
        },
    ],
    "footer_items": [
        "copyright",
        "last-updated",
    ],
    "analytics": {"google_analytics_id": 'UA-61554933-1'}
}

nbbuild_cell_timeout = 360

extensions += [
    'nbsite.gallery',
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

html_context.update({
    # Used to add binder links to the latest released tag.
    "last_release": f"v{'.'.join(holoviews.__version__.split('.')[:3])}",
    'github_user': 'holoviz',
    'github_repo': 'holoviews',
    "default_mode": "light"
})

# Override the Sphinx default title that appends `documentation`
html_title = f'{project} v{version}'
# Format of the last updated section in the footer
html_last_updated_fmt = '%Y-%m-%d'
