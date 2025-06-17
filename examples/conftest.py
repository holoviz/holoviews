import os
import platform
import sys
from importlib.util import find_spec

import bokeh
import pandas as pd
from packaging.version import Version

# Setting this to not error out if no install is done.
os.environ["PYTHONPATH"] = os.path.dirname(os.path.dirname(__file__))

system = platform.system()
py_version = sys.version_info[:2]
PANDAS_GE_2_0_0 = Version(pd.__version__).release >= (2, 0, 0)

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
if PANDAS_GE_2_0_0 and system == "Windows":
    collect_ignore_glob += [
        "gallery/demos/bokeh/point_draw_triangulate.ipynb",
        "reference/elements/*/TriMesh.ipynb",
        "user_guide/15-Large_Data.ipynb",
    ]


# 2024-01-15: See https://github.com/holoviz/holoviews/issues/6069
if system == "Windows":
    collect_ignore_glob += [
        "user_guide/Deploying_Bokeh_Apps.ipynb",
    ]

# First available in Bokeh 3.2.0
if Version(bokeh.__version__).release < (3, 2, 0):
    collect_ignore_glob += [
        "reference/elements/bokeh/HLines.ipynb",
        "reference/elements/bokeh/HSpans.ipynb",
        "reference/elements/bokeh/VLines.ipynb",
        "reference/elements/bokeh/VSpans.ipynb",
    ]

# 2024-03-27: ffmpeg errors on Windows CI
if system == "Windows" and os.environ.get("GITHUB_RUN_ID"):
    collect_ignore_glob += [
        "user_guide/Plotting_with_Matplotlib.ipynb",
    ]

# 2024-05: Numpy 2.0
if find_spec("datashader") is None:
    collect_ignore_glob += [
        "reference/elements/matplotlib/ImageStack.ipynb",
        "reference/elements/plotly/ImageStack.ipynb",
        "user_guide/15-Large_Data.ipynb",
        "user_guide/16-Streaming_Data.ipynb",
        "user_guide/Linked_Brushing.ipynb",
        "user_guide/Network_Graphs.ipynb",
    ]

if find_spec("scikit-image") is None:
    collect_ignore_glob += [
        "user_guide/Network_Graphs.ipynb",
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
            if call.excinfo.type is RuntimeError and call.excinfo.value.args[0] in msg:
                tr.outcome = "skipped"
                tr.wasxfail = f"reason: {msg}"

    return tr
