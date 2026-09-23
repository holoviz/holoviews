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

if find_spec("datashader") is None:
    collect_ignore_glob += [
        "reference/elements/matplotlib/ImageStack.ipynb",
        "reference/elements/plotly/ImageStack.ipynb",
        "user_guide/15-Large_Data.ipynb",
        "user_guide/16-Streaming_Data.ipynb",
        "user_guide/17-Dashboards.ipynb",
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

if find_spec("scipy") is None:
    collect_ignore_glob += [
        "gallery/demos/bokeh/autompg_violins.ipynb",
        "gallery/demos/bokeh/hextile_movie_ratings.ipynb",
        "gallery/demos/bokeh/histogram_example.ipynb",
        "gallery/demos/bokeh/iris_density_grid.ipynb",
        "gallery/demos/bokeh/iris_grouped_grid.ipynb",
        "gallery/demos/bokeh/life_expectancy_split_violin.ipynb",
        "gallery/demos/bokeh/lorenz_attractor_example.ipynb",
        "gallery/demos/bokeh/point_draw_triangulate.ipynb",
        "gallery/demos/matplotlib/hextile_movie_ratings.ipynb",
        "gallery/demos/matplotlib/histogram_example.ipynb",
        "gallery/demos/matplotlib/iris_density_grid.ipynb",
        "gallery/demos/matplotlib/iris_grouped_grid.ipynb",
        "gallery/demos/matplotlib/lorenz_attractor_example.ipynb",
        "reference/elements/bokeh/Bivariate.ipynb",
        "reference/elements/bokeh/Dendrogram.ipynb",
        "reference/elements/bokeh/Distribution.ipynb",
        "reference/elements/bokeh/HexTiles.ipynb",
        "reference/elements/bokeh/TriMesh.ipynb",
        "reference/elements/bokeh/Violin.ipynb",
        "reference/elements/bokeh/Waterfall.ipynb",
        "reference/elements/matplotlib/Bivariate.ipynb",
        "reference/elements/matplotlib/Dendrogram.ipynb",
        "reference/elements/matplotlib/Distribution.ipynb",
        "reference/elements/matplotlib/HexTiles.ipynb",
        "reference/elements/matplotlib/TriMesh.ipynb",
        "reference/elements/matplotlib/Waterfall.ipynb",
        "reference/elements/plotly/Distribution.ipynb",
        "reference/streams/bokeh/Selection1D_tap.ipynb",
        "user_guide/09-Gridded_Datasets.ipynb",
        "user_guide/11-Transforming_Elements.ipynb",
    ]

if find_spec("polars") is None:
    collect_ignore_glob += [
        "reference/elements/bokeh/Waterfall.ipynb",
        "reference/elements/matplotlib/Waterfall.ipynb",
        "reference/elements/plotly/Waterfall.ipynb",
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
