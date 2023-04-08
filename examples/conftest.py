import sys

import pandas as pd
from packaging.version import Version

collect_ignore_glob = [
    # Needs selenium, phantomjs, firefox, and geckodriver to save a png picture
    "user_guide/Plotting_with_Bokeh.ipynb",
    # Possible timeout error
    "user_guide/17-Dashboards.ipynb",
    # Give file not found here.
    "user_guide/Plots_and_Renderers.ipynb",
]

if sys.version_info[:2] == (3, 11):
    collect_ignore_glob += [
        # numba not supported on Python 3.11
        "user_guide/15-Large_Data.ipynb",
        "user_guide/16-Streaming_Data.ipynb",
        "user_guide/Network_Graphs.ipynb",
        "user_guide/Linked_Brushing.ipynb",
    ]

if Version(pd.__version__) >= Version("2.0"):
    collect_ignore_glob += [
        # Xarray incompatibility: https://github.com/pydata/xarray/issues/7716
        # "user_guide",
        # Pandas bug: https://github.com/pandas-dev/pandas/issues/52451
        "reference/elements/*/TriMesh.ipynb",
        "gallery/demos/bokeh/point_draw_triangulate.ipynb",
    ]
