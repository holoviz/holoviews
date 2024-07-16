from contextlib import suppress

import bokeh
from packaging.version import Version

if Version(Version(bokeh.__version__).base_version) < Version("3.5"):
    import bokeh.sampledata

    bokeh.sampledata.download()

with suppress(ImportError):
    import pooch  # noqa: F401
    import scipy  # noqa: F401
    import xarray as xr

    xr.tutorial.open_dataset("air_temperature")
    xr.tutorial.open_dataset("rasm")
