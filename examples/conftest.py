import sys
import platform

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


if sys.version_info == (3, 8) and platform.system() == "Linux":
    # from matplotlib.cbook import get_sample_data has problem
    # on Linux with Python 3.8.
    collect_ignore_glob += [
        "gallery/demos/*/bachelors_degrees_by_gender.ipynb",
        "gallery/demos/*/topographic_hillshading.ipynb",
    ]
