import sys
import platform

__all__ = ("show_versions",)

PACKAGES = [
    # Data
    "cudf",
    "dask",
    "ibis",
    "networkx",
    "numpy",
    "pandas",
    "pyarrow",
    "spatialpandas",
    "streamz",
    "xarray",
    # Processing
    "numba",
    "skimage",
    "scipy",
    # Plotting
    "bokeh",
    "colorcet",
    "datashader",
    "geoviews",
    "hvplot",
    "matplotlib",
    "PIL",
    "plotly",
    # Jupyter
    "IPython",
    "jupyter_bokeh",
    "jupyterlab",
    "notebook",
    # Misc
    "panel",
    "param",
]


def show_versions():
    print(f"Python              :  {sys.version}")
    print(f"Operating system    :  {platform.platform()}")
    _panel_comms()
    print()
    _package_version("holoviews")
    print()
    for p in sorted(PACKAGES, key=lambda x: x.lower()):
        _package_version(p)


def _package_version(p):
    try:
        __import__(p)
        print(f"{p:20}:  {sys.modules[p].__version__}")
    except ImportError:
        print(f"{p:20}:  -")


def _panel_comms():
    import panel as pn

    print(f"{'Panel comms':20}:  {pn.config.comms}")
