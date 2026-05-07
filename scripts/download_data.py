from __future__ import annotations

import sys
import time
from contextlib import suppress

from holoviews.core.util.dependencies import _no_import_version


def retry(func, *args, **kwargs):
    for i in range(5):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            wait = 10 * 2**i
            print(f"Attempt {i + 1} failed: {e}. Retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
    return func(*args, **kwargs)


if _no_import_version("bokeh") < (3, 5, 0):
    import bokeh.sampledata

    retry(bokeh.sampledata.download)  # ty:ignore[unresolved-attribute]
    print("Downloaded bokeh sampledata")

with suppress(ImportError):
    import pooch  # noqa: F401
    import scipy  # noqa: F401
    import xarray as xr

    retry(xr.tutorial.open_dataset, "air_temperature")
    retry(xr.tutorial.open_dataset, "rasm", decode_times=False)
    print("Downloaded xarray tutorial datasets")

with suppress(ImportError):
    from scipy.datasets import ascent

    retry(ascent)
    print("Downloaded scipy dataset")
