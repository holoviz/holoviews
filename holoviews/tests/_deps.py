from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

import pytest

from holoviews.core.util.dependencies import _is_installed

_mods = {}


def _optional_dependencies(name: str, package: str | None = None):
    """Return a lazy-loading proxy for the first item in names if all `names` are installed else `None`."""
    if not _is_installed(package or name):
        return None

    @staticmethod
    def __getattr__(attr):
        if name not in _mods:
            _mods[name] = import_module(name)
        return getattr(_mods[name], attr)

    return type(name, (), {"__getattr__": __getattr__})()


optional_dependencies = import_module if TYPE_CHECKING else _optional_dependencies
cftime = optional_dependencies("cftime")
contourpy = optional_dependencies("contourpy")
dask = optional_dependencies("dask")
da = optional_dependencies("dask.array")
dd = optional_dependencies("dask.dataframe")
ds = optional_dependencies("datashader")
duckdb = optional_dependencies("duckdb")
ibis = optional_dependencies("ibis")
IPython = optional_dependencies("IPython")
jupyter = optional_dependencies("jupyter")
mpl = optional_dependencies("matplotlib")
nx = optional_dependencies("networkx")
pd = optional_dependencies("pandas")
plotly = optional_dependencies("plotly")
pl = optional_dependencies("polars")
pa = optional_dependencies("pyarrow")
scipy = optional_dependencies("scipy")
shapely = optional_dependencies("shapely")
spd = optional_dependencies("spatialpandas")
tsdownsample = optional_dependencies("tsdownsample")
xr = optional_dependencies("xarray")
xyzservices = optional_dependencies("xyzservices")


def _skip(module, name):
    modules = module if isinstance(module, list) else [module]
    return pytest.mark.skipif(any(m is None for m in modules), reason=f"{name} is not installed")


cftime_skip = _skip(cftime, "cftime")
contourpy_skip = _skip(contourpy, "contourpy")
dask_skip = _skip(dask, "dask")
da_skip = _skip(da, "dask.array")
dd_skip = _skip([dd, pa], "dask.dataframe")
ds_skip = _skip(ds, "datashader")
duckdb_skip = _skip(duckdb, "duckdb")
ibis_skip = _skip(ibis, "ibis")
ipython_skip = _skip(IPython, "IPython")
mpl_skip = _skip(mpl, "matplotlib")
nx_skip = _skip(nx, "networkx")
jupyter_skip = _skip(jupyter, "jupyter")
pd_skip = _skip(pd, "pandas")
plotly_skip = _skip(plotly, "plotly")
pl_skip = _skip(pl, "polars")
pa_skip = _skip(pa, "pyarrow")
scipy_skip = _skip(scipy, "scipy")
shapely_skip = _skip(shapely, "shapely")
spd_skip = _skip(spd, "spatialpandas")
tsdownsample_skip = _skip(tsdownsample, "tsdownsample")
xr_skip = _skip(xr, "xarray")
xyzservices_skip = _skip(xyzservices, "xyzservices")


if spd:
    # Will import _posixshmem on Linux + Python 3.14 + spatialpandas
    # which does not work with our pytest.fixture unimport
    import multiprocessing.resource_tracker  # noqa: F401
