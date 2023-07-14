import platform
import sys

import pandas as pd
from packaging.version import Version

PD2 = Version(pd.__version__) >= Version("2.0")

# Having "OMP_NUM_THREADS"=1, set as an environment variable, can be needed
# to avoid crashing when running tests with pytest-xdist on Windows.
# This is set in the .github/workflows/test.yaml file.
# https://github.com/holoviz/holoviews/pull/5720

collect_ignore_glob = [
    # Needs selenium, phantomjs, firefox, and geckodriver to save a png picture
    "user_guide/Plotting_with_Bokeh.ipynb",
    # Possible timeout error
    "user_guide/17-Dashboards.ipynb",
    # Give file not found
    "user_guide/Plots_and_Renderers.ipynb",
]


# 2023-07-14 with following error:
# ValueError: Buffer dtype mismatch, expected 'const int64_t' but got 'int'
if PD2 and platform.system() == "Windows":
    collect_ignore_glob += [
        "gallery/demos/bokeh/point_draw_triangulate.ipynb",
        "reference/elements/*/TriMesh.ipynb",
        "user_guide/15-Large_Data.ipynb",
    ]


# 2023-07-14 with following error:
# 'from matplotlib.cbook import get_sample_data' cannot find file
if sys.version_info[:2] == (3, 8) and platform.system() == "Linux":
    collect_ignore_glob += [
        "gallery/demos/*/bachelors_degrees_by_gender.ipynb",
        "gallery/demos/*/topographic_hillshading.ipynb",
    ]
