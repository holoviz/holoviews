import param

param.parameterized.docstring_signature = False
param.parameterized.docstring_describe_params = False

from nbsite.shared_conf import *  # noqa: F403

# Declare information specific to this project.
project = 'HoloViews'
authors = 'HoloViz developers'
copyright = '2005 ' + authors
description = 'Stop plotting your data - annotate your data and let it visualize itself.'

# Setting this to not error out if no install is done
root_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, root_path)
os.environ["PYTHONPATH"] = root_path

import holoviews as hv

version = release = base_version(hv.__version__)

hv.extension.inline = False

html_theme = 'pydata_sphinx_theme'
html_logo = '_static/logo_horizontal.png'
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
}

nbbuild_cell_timeout = 360

extensions += [
    'nbsite.gallery',
    'sphinx_copybutton',
    'nbsite.analytics',
]

myst_enable_extensions = ["colon_fence", "deflist"]

nbsite_analytics = {
    'goatcounter_holoviz': True,
}

rediraffe_redirects = {
    'gallery/demos/bokeh/eeg_viewer': 'gallery/demos/bokeh/multichannel_timeseries_viewer',
}

nbsite_gallery_conf = {
    'backends': ['bokeh', 'matplotlib', 'plotly'],
    'galleries': {},
    'github_org': 'holoviz',
    'github_project': 'holoviews'
}

if os.environ.get('HV_DOC_GALLERY') not in ('False', 'false', '0'):
    nbsite_gallery_conf['galleries']['gallery'] = {
        'title': 'Gallery',
        'intro': (
            'Also visit the `Examples HoloViz Gallery <https://examples.holoviz.org>`_ to '
            'discover a curated collection of domain-specific narrative examples using '
            'HoloViews and various HoloViz projects.'
        ),
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
            'apps',
            'features',
        ]
    }

html_context.update({
    # Used to add binder links to the latest released tag.
    "last_release": f"v{'.'.join(hv.__version__.split('.')[:3])}",
    'github_user': 'holoviz',
    'github_repo': 'holoviews',
    "default_mode": "light"
})

# Override the Sphinx default title that appends `documentation`
html_title = f'{project} v{version}'
# Format of the last updated section in the footer
html_last_updated_fmt = '%Y-%m-%d'
