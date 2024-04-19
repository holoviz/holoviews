import bokeh.sampledata

bokeh.sampledata.download()

try:
    import pooch  # noqa: F401
    import scipy  # noqa: F401
    import xarray as xr
except ImportError:
    pass
else:
    xr.tutorial.open_dataset("air_temperature")
    xr.tutorial.open_dataset("rasm")
