import platform
import sys
from importlib.util import find_spec

from holoviews.core.util.dependencies import _no_import_version

system = platform.system()
py_version = sys.version_info[:2]

collect_ignore_glob = [
    # Needs selenium, phantomjs, firefox, and geckodriver to save a png picture
    "user_guide/Plotting_with_Bokeh.ipynb",
]

# First available in Bokeh 3.2.0
if _no_import_version("bokeh") < (3, 2, 0):
    collect_ignore_glob += [
        "reference/elements/bokeh/HLines.ipynb",
        "reference/elements/bokeh/HSpans.ipynb",
        "reference/elements/bokeh/VLines.ipynb",
        "reference/elements/bokeh/VSpans.ipynb",
    ]

# 2024-05: Numpy 2.0
if find_spec("datashader") is None:
    collect_ignore_glob += [
        "reference/elements/matplotlib/ImageStack.ipynb",
        "reference/elements/plotly/ImageStack.ipynb",
        "user_guide/15-Large_Data.ipynb",
        "user_guide/16-Streaming_Data.ipynb",
        "user_guide/Interactive_Hover_for_Big_Data.ipynb",
        "user_guide/Linked_Brushing.ipynb",
        "user_guide/Network_Graphs.ipynb",
    ]

if find_spec("scikit-image") is None:
    collect_ignore_glob += [
        "user_guide/Network_Graphs.ipynb",
    ]

if find_spec("tsdownsample") is None:
    collect_ignore_glob += [
        "gallery/demos/bokeh/multichannel_timeseries_viewer.ipynb",
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
