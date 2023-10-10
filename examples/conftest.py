import platform
import sys

import bokeh
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


# First available in Bokeh 3.2.0
if Version(bokeh.__version__) < Version("3.2.0"):
    collect_ignore_glob += [
        "reference/elements/bokeh/HLines.ipynb",
        "reference/elements/bokeh/HSpans.ipynb",
        "reference/elements/bokeh/VLines.ipynb",
        "reference/elements/bokeh/VSpans.ipynb",
    ]


def pytest_runtest_makereport(item, call):
    """
    Skip tests that fail because "the kernel died before replying to kernel_info"
    this is a common error when running the example tests in CI.

    Inspired from: https://stackoverflow.com/questions/32451811

    """
    from _pytest.runner import pytest_runtest_makereport

    tr = pytest_runtest_makereport(item, call)

    if call.excinfo is not None:
        msgs = [
            "Kernel died before replying to kernel_info",
            "Kernel didn't respond in 60 seconds",
        ]
        for msg in msgs:
            if call.excinfo.type == RuntimeError and call.excinfo.value.args[0] in msg:
                tr.outcome = "skipped"
                tr.wasxfail = f"reason: {msg}"

    return tr
