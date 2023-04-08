import pandas as pd
from packaging.version import Version

collect_ignore_glob = [
    "Plotting_with_Bokeh",  # needs selenium, phantomjs, firefox, and geckodriver to save a png picture.
    "17-Dashboards",  # can give a timeout error.
    "Plots_and_Renderers",  # give file not found here.
]

# Add sys.version for numba
# if sys.version_info == (3, 11):

if Version(pd.__version__) >= Version("2.0"):
    collect_ignore_glob += [
        # Xarray incompatibility: https://github.com/pydata/xarray/issues/7716
        "user_guide",
        # Pandas bug: https://github.com/pandas-dev/pandas/issues/52451
        "reference/elements/*/TriMesh.ipynb",
    ]
