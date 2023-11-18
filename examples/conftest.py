import platform
import sys

import bokeh
import pandas as pd
from packaging.version import Version

system = platform.system()
py_version = sys.version_info[:2]
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
if PD2 and system == "Windows":
    collect_ignore_glob += [
        "gallery/demos/bokeh/point_draw_triangulate.ipynb",
        "reference/elements/*/TriMesh.ipynb",
        "user_guide/15-Large_Data.ipynb",
    ]


# 2023-10-25, flaky on CI with timeout
if system == "Darwin":
    collect_ignore_glob += [
        "user_guide/16-Streaming_Data.ipynb",
    ]


# First available in Bokeh 3.2.0
if Version(bokeh.__version__) < Version("3.2.0"):
    collect_ignore_glob += [
        "reference/elements/bokeh/HLines.ipynb",
        "reference/elements/bokeh/HSpans.ipynb",
        "reference/elements/bokeh/VLines.ipynb",
        "reference/elements/bokeh/VSpans.ipynb",
    ]
