import sys

# import pandas as pd
# from packaging.version import Version

# PD2 = Version(pd.__version__) >= Version("2.0")

collect_ignore_glob = [
    # Needs selenium, phantomjs, firefox, and geckodriver to save a png picture
    "user_guide/Plotting_with_Bokeh.ipynb",
    # Possible timeout error
    "user_guide/17-Dashboards.ipynb",
    # Give file not found
    "user_guide/Plots_and_Renderers.ipynb",
]

# Pandas bug: https://github.com/pandas-dev/pandas/issues/52451
if sys.platform == "win32":
    collect_ignore_glob += [
        "gallery/demos/bokeh/point_draw_triangulate.ipynb",
        "reference/elements/*/TriMesh.ipynb",
        "user_guide/15-Large_Data.ipynb",
    ]


# try:
#     from dask.context import config
#     config.set(scheduler='synchronous')
# except ImportError:
#     pass
